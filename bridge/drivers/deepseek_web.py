"""Драйвер `chat.deepseek.com` через управляемый Playwright Chromium.

Как это работает и почему именно так
------------------------------------
API-ключа у нас нет, поэтому «точкой входа» в DeepSeek остаётся обычная страница
чата. Мост держит один Chromium с persistent-профилем (папка
`data/browser-profile`), где один раз сохраняются cookies после ручного логина.
Дальше он делает ровно то, что делает человек: вставляет текст в поле ввода,
отправляет его и читает `innerText` нового блока ответа, отдавая его нарастающими
снапшотами.

Что из этого следует для кода:
  * один браузер = одна одновременная генерация (очередь — в `bridge/engine.py`);
  * селекторы вынесены в `selectors.json`, потому что вёрстка чата меняется;
  * `send()` отдаёт накопленный текст, а не дельты — diff считает движок;
  * падение страницы отличается от «надо перелогиниться» — у клиента разные действия.

Три правила, на которых держится надёжность (без них были симптомы «отправляет
через раз» и «ответ не возвращается»):

  1. **Отправку надо ПРОВЕРЯТЬ, а не считать её успешной по факту нажатия Enter.**
     React-поле контролируемое: `fill()` без корректного `input`-события оставляет
     текст в поле, и сообщение просто не уходит — а мы молча ждём ответ 4 минуты.
     Поэтому после отправки ждём, пока поле опустеет (это единственный достоверный
     сигнал «сообщение приняли»), с явными фолбэками и внятной ошибкой.
  2. **Отвечный блок ищем диффом по содержимому, а не по «числу блоков».**
     Счётчик блоков не меняется, когда сайт переиспользует контейнер, — и тогда
     читается ПРЕДЫДУЩИЙ ответ. Плюс тот же селектор матчит бабл пользователя,
     поэтому эхо промпта нельзя принимать за ответ.
  3. **После любой неудачи страницу надо привести в покой.**
     Иначе оборванная генерация держит «стоп» и заблокированное поле ввода, и
     следующий запрос физически не может отправиться (лечилось ручным «новым
     чатом»). `ensure_idle()` перед отправкой и `recover()` после сбоя закрывают
     это сами.

Playwright импортируется лениво: без него запускаются API, тесты и `fake`-драйвер.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from ..config import Settings
from ..drivers.base import DriverStatus, Selectors
from ..errors import DriverCrashed, DriverNotReady, GenerationTimeout


def clean(text: str | None) -> str:
    return (text or "").replace("\u200b", "").strip()


def pick_answer(current: list[str], previous: list[str], prompt: str) -> str | None:
    """Текст НОВОГО ответа или None, если нового ещё нет.

    Два сигнала, по порядку:

    1. блоков стало больше → новый ответ последний из «выросших» (пропуская
       эхо нашего же промпта: тот же селектор матчит и бабл пользователя);
    2. блоков столько же, но текст последнего изменился → сайт переиспользует
       контейнер, и это и есть растущий ответ.

    Именно «по позиции + по изменению», а не «отличается ли текст от любого
    прошлого блока»: иначе повторный ответ, совпадающий с предыдущим
    («42», «да», «ок»), был бы неотличим от старого и мост ждал бы его до
    таймаута. При этом «ничего не изменилось» по-прежнему даёт None — это и
    лечило баг «вернул предыдущий ответ».
    """
    cur = [clean(t) for t in current]
    prev = [clean(t) for t in previous]
    echo = clean(prompt)
    if len(cur) > len(prev):
        for text in reversed(cur[len(prev) :]):
            if text and text != echo:
                return text
        return None
    if cur and prev and cur[-1] != prev[-1] and cur[-1] != echo:
        return cur[-1]
    return None


def settled_answer(current: list[str], prompt: str) -> str | None:
    """Последний непустой блок, когда генерация точно закончилась.

    Нужен для случая «новый ответ совпал текстом со старого»: позиции говорят,
    что блок новый, а содержимое — что он как предыдущий. После честного конца
    генерации последний блок и есть ответ.
    """
    echo = clean(prompt)
    for text in reversed([clean(t) for t in current]):
        if text and text != echo:
            return text
    return None


def is_settled(
    text: str,
    stable_polls: int,
    *,
    generating: bool,
    ever_generating: bool,
    needed: int,
) -> bool:
    """Считаем ответ готовым, только когда страница спокойна достаточно долго.

    `ever_generating` — мы хоть раз видели кнопку Stop. Если нет (селектор
    устарел), доверять «не генерирует» нельзя, и окно стабильности наращивается:
    иначе обрывок на полуслове ушёл бы клиенту как полный ответ.
    """
    if not text or generating:
        return False
    return stable_polls >= (needed if ever_generating else needed * 4)


class DeepSeekWebChatDriver:
    """Одна вкладка чата DeepSeek, управляемая как обычный пользователь."""

    name = "deepseek_web"
    truncated = False

    def __init__(self, settings: Settings, selectors: Selectors | None = None) -> None:
        self.settings = settings
        self.selectors = selectors or Selectors.load(settings.selectors_file)
        self._pw: Any = None
        self._ctx: Any = None
        self._page: Any = None
        self._logged_in: bool | None = None
        self._headless: bool | None = None
        self._detail: str | None = "не запущен"
        self._reasoning_on: bool = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ жизненный цикл

    async def start(self, *, headless: bool | None = None) -> None:
        async with self._lock:
            if self._page is not None and not self._page.is_closed():
                return
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:  # pragma: no cover - зависит от окружения
                raise DriverCrashed(
                    "playwright не установлен. `pip install playwright && playwright install chromium`"
                ) from exc

            self.settings.ensure_dirs()
            want_headless = self.settings.headless if headless is None else headless
            try:
                self._pw = await async_playwright().start()
                self._ctx = await self._pw.chromium.launch_persistent_context(
                    user_data_dir=str(self.settings.profile_dir),
                    headless=want_headless,
                    viewport={"width": 1360, "height": 900},
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception as exc:  # noqa: BLE001 - причину знает только Playwright
                await self._shutdown()
                reason = (str(exc).strip().splitlines() or [repr(exc)])[0]
                raise DriverCrashed(
                    f"не удалось запустить Chromium: {reason[:200]}",
                    hint=(
                        "Обычно это не скачанный браузер: выполни `pip install playwright && "
                        "playwright install chromium`. Реже — профиль занят другим экземпляром моста."
                    ),
                ) from exc

            self._headless = want_headless
            self._page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
            self._page.on("close", lambda *_: setattr(self, "_page", None))
            await self._open_chat()

    async def stop(self) -> None:
        async with self._lock:
            await self._shutdown()

    async def _shutdown(self) -> None:
        for closer in (getattr(self, "_ctx", None), getattr(self, "_pw", None)):
            if closer is None:
                continue
            try:
                await (closer.close() if hasattr(closer, "close") else closer.stop())
            except Exception:
                pass
        self._ctx = None
        self._pw = None
        self._page = None
        self._logged_in = None
        self._detail = "не запущен"

    def _require_page(self) -> Any:
        if self._page is None or self._page.is_closed():
            raise DriverNotReady(
                "Браузер моста не запущен.",
                hint="Нужен `python -m bridge.cli login` (он же поднимает браузер).",
            )
        return self._page

    # ------------------------------------------------------------------ навигация/состояние

    async def _open_chat(self) -> None:
        page = self._require_page()
        try:
            await page.goto(self.settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
        except Exception as exc:
            raise DriverCrashed(f"страница {self.settings.deepseek_url} не открылась: {exc}") from exc
        self._logged_in = await self._wait_for_input(self.settings.ready_timeout_s)
        self._detail = "готов" if self._logged_in else "нужен логин"

    async def _wait_for_input(self, timeout_s: int) -> bool:
        """Поле ввода = признак залогиненного чата. На экране логина его нет."""
        page = self._page
        if page is None or page.is_closed():
            return False
        candidates = self.selectors.candidates("input") or ["textarea"]
        per = max(400, int(timeout_s * 1000 / len(candidates)))
        for selector in candidates:
            try:
                await page.locator(selector).first.wait_for(state="visible", timeout=per)
                return True
            except Exception:
                continue
        return False

    async def status(self) -> DriverStatus:
        page = self._page
        running = bool(page is not None and not page.is_closed())
        url = None
        if running:
            try:
                url = page.url
            except Exception:
                url = None
        if running and url and ("/sign-in" in url or "login" in url):
            self._logged_in = False
        if running and self._logged_in is None:
            self._logged_in = await self._wait_for_input(2)
        return DriverStatus(
            running=running,
            logged_in=self._logged_in,
            url=url,
            headless=self._headless,
            detail=self._detail if self._detail else ("готов" if self._logged_in else "нужен логин"),
        )

    async def login(self, timeout_s: int | None = None) -> bool:
        """Поднять headful-окно и подождать, пока человек введёт аккаунт."""
        limit = timeout_s or self.settings.login_timeout_s
        await self.start(headless=False)
        page = self._require_page()
        try:
            await page.goto(self.settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
        except Exception as exc:
            raise DriverCrashed(f"не удалось открыть страницу логина: {exc}") from exc
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if await self._wait_for_input(min(5, max(1, int(deadline - time.monotonic())))):
                self._logged_in = True
                self._detail = "готов"
                return True
            await asyncio.sleep(1.0)
        self._logged_in = False
        self._detail = "логин не завершён (таймаут)"
        return False

    # ------------------------------------------------------------------ элементы страницы

    async def _locator(self, key: str, required: bool = True) -> Any:
        """Первый сработавший вариант селектора; `None`, если ничего нет и не обязательно."""
        page = self._require_page()
        for selector in self.selectors.candidates(key):
            try:
                locator = page.locator(selector).first
                if await locator.count() > 0:
                    return locator
            except Exception:
                continue
        if required:
            raise DriverCrashed(
                f"на странице не найден элемент {key!r} (пробовали: {self.selectors.get(key)})",
                hint="Сохрани `python -m bridge.cli probe --save` и поправь selectors.json.",
            )
        return None

    async def _answer_texts(self) -> tuple[list[str], int]:
        """Тексты последних N блоков ответа и их смещение в DOM.

        Читаем «окном», а не всем списком: в длинном диалоге 200 блоков
        `inner_text` каждого poll превратил бы опрос DOM в тормоз.
        """
        page = self._require_page()
        size = max(2, self.settings.answer_window)
        for selector in self.selectors.candidates("assistant_message"):
            try:
                locator = page.locator(selector)
                count = await locator.count()
            except Exception:
                continue
            if not count:
                continue
            base = max(0, count - size)
            texts: list[str] = []
            for index in range(base, count):
                try:
                    texts.append(await locator.nth(index).inner_text(timeout=1_500))
                except Exception:
                    texts.append("")
            return texts, base
        return [], 0

    async def _is_generating(self) -> bool:
        page = self._page
        if page is None or page.is_closed():
            return False
        for selector in self.selectors.candidates("stop_button"):
            try:
                if await page.locator(selector).count() > 0:
                    return True
            except Exception:
                continue
        return False

    async def _field_text(self, box: Any) -> str:
        """Содержимое поля ввода для textarea и contenteditable одновременно."""
        if box is None:
            return ""
        try:
            value = await box.input_value(timeout=800)
            if value is not None:
                return str(value)
        except Exception:
            pass
        try:
            return await box.inner_text(timeout=800)
        except Exception:
            return ""

    async def _wait_field_cleared(self, box: Any, timeout_s: float) -> bool:
        """Пока поле не опустел, сообщение не отправлено. Это и есть проверка отправки."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not clean(await self._field_text(box)):
                return True
            await asyncio.sleep(0.12)
        return not clean(await self._field_text(box))

    # ------------------------------------------------------------------ покой/санация

    async def ensure_idle(self, timeout_s: float | None = None) -> bool:
        """Привести страницу в покой: снять «генерация идёт», вычистить поле ввода.

        Возвращает True, если страница свободна. Вызывается перед отправкой
        (чтобы не толкаться в заблокированное поле) и после сбоя (чтобы следующий
        запрос не надо было начинать ручным «новым чатом»).
        """
        page = self._page
        if page is None or page.is_closed():
            return False
        limit = self.settings.idle_wait_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + limit
        stopped = False
        while await self._is_generating():
            if not stopped:
                stop_button = await self._locator("stop_button", required=False)
                if stop_button is not None:
                    try:
                        await stop_button.click(timeout=1_500)
                    except Exception:
                        pass
                stopped = True
            await asyncio.sleep(0.2)
            if time.monotonic() > deadline:
                break
        if await self._is_generating():
            self._detail = "страница всё ещё генерирует — принудительная пауза не сработала"
            return False
        box = await self._locator("input", required=False)
        if box is not None and clean(await self._field_text(box)):
            await self.clear_input()
        return True

    async def clear_input(self) -> None:
        page = self._page
        if page is None or page.is_closed():
            return
        box = await self._locator("input", required=False)
        if box is None:
            return
        try:
            await box.click(timeout=1_500)
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Delete")
            if clean(await self._field_text(box)):  # macOS-раскладка
                await page.keyboard.press("Meta+A")
                await page.keyboard.press("Delete")
        except Exception:
            pass

    async def recover(self) -> None:
        """Вызывается движком после неудачной генерации. Ничего не бросает."""
        try:
            await self.ensure_idle()
            await self._check_login_state()
        except Exception:
            pass

    async def _check_login_state(self) -> None:
        page = self._page
        if page is None or page.is_closed():
            return
        try:
            url = page.url
        except Exception:
            return
        if url and ("/sign-in" in url or "login" in url):
            self._logged_in = False
            self._detail = "слетел логин"

    # ------------------------------------------------------------------ диалог

    async def begin_chat(self) -> None:
        """Новый диалог: клик по кнопке с ПРОВЕРКОЙ, что он получился, иначе — перезагрузка.

        Раньше мы верили клику на слово: если он сглатывался, запрос уходил в
        старый диалог, а читались ответы оттуда.
        """
        page = self._require_page()
        for selector in self.selectors.candidates("new_chat_button"):
            try:
                button = page.locator(selector).first
                if await button.count() == 0:
                    continue
                await button.click(timeout=3_000)
                if await self._wait_until_fresh(min(4.0, self.settings.ready_timeout_s)):
                    return
            except Exception:
                continue
        await self._open_chat()
        await self._wait_until_fresh(self.settings.ready_timeout_s)

    async def _wait_until_fresh(self, timeout_s: float) -> bool:
        """Чистый диалог = нет блоков ответа и пустое поле ввода."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            texts, _ = await self._answer_texts()
            box = await self._locator("input", required=False)
            field_empty = True if box is None else not clean(await self._field_text(box))
            if not any(clean(text) for text in texts) and field_empty:
                return True
            await asyncio.sleep(0.15)
        return False

    async def _ensure_reasoning(self, want: bool) -> bool:
        """Поставить режим «думать» в нужное состояние, а не щёлкать вслепую.

        Слепой toggle был источником «через раз» на deepseek-reasoner: второй запрос
        выключал то, что включил первый, и режим уезжал вперемешку. Теперь состояние
        хранится локально и кликаем только при расхождении.
        """
        if self._reasoning_on == want:
            return True
        if self._page is None or self._page.is_closed():
            return False
        for selector in self.selectors.candidates("think_toggle"):
            try:
                button = self._page.locator(selector).first
                if await button.count() == 0:
                    continue
                await button.click(timeout=2_000)
                self._reasoning_on = want
                return True
            except Exception:
                continue
        # кнопки нет (аккаунт/язык/редизайн) — честно говорим, что режим не переключали
        self._detail = f"переключатель режима мышления не найден (запрошен reasoning={want})"
        return False

    # ------------------------------------------------------------------ отправка

    async def _write_prompt(self, box: Any, prompt: str) -> None:
        """Ввести текст так, чтобы React его увидел, и ПРОВЕРИТЬ, что увидел.

        Порядок: fill (деснативное событие + мгновенно) → insert_text (для
        contenteditable) → посылимвольно (самое совместимое, самое медленное).
        Каждый шаг проверяется чтением поля, потому что «исключение» тут плохой
        индикатор: поле может остаться пустым абсолютно молча.
        """
        page = self._require_page()
        await box.click(timeout=5_000)
        try:
            await box.fill("", timeout=1_500)
        except Exception:
            pass

        attempts = (
            lambda: box.fill(prompt, timeout=3_000),
            lambda: page.keyboard.insert_text(prompt),
            lambda: page.keyboard.type(prompt, delay=6),
        )
        for attempt in attempts:
            try:
                await attempt()
            except Exception:
                continue
            if clean(await self._field_text(box)) == clean(prompt):
                return
            try:  # поле заполнилось «не полностью» — чистим и пробуем следующий способ
                await box.press("Control+A")
                await box.press("Delete")
            except Exception:
                pass
        await self._dump("prompt-not-typed", prompt)
        raise DriverCrashed(
            "текст не появился в поле ввода чата",
            hint="Скорее всего, поменялся селектор `input`. Смотри `python -m bridge.cli probe --save`.",
        )

    async def _submit(self, box: Any) -> None:
        """Отправить и убедиться, что поле опустело (= сообщение приняли).

        Это чинит «через раз»: нажатие Enter в незафиксированное React-поле не
        отправка. Ждём очищения поля, при неудаче пробуем кнопку, потом Enter ещё
        раз, и только потом падаем — но уже с конкретикой, а не с таймаутом ответа.
        """
        page = self._require_page()
        wait = float(self.settings.send_wait_s)

        if self.settings.submit in {"auto", "enter"}:
            try:
                await box.press("Enter")
            except Exception:
                await page.keyboard.press("Enter")
            if await self._wait_field_cleared(box, wait):
                return

        if self.settings.submit in {"auto", "button"}:
            button = await self._locator("send_button", required=False)
            if button is not None:
                try:
                    if await button.is_enabled():
                        await button.click(timeout=2_500)
                        if await self._wait_field_cleared(box, wait):
                            return
                except Exception:
                    pass

        try:
            await box.press("Enter")
        except Exception:
            await page.keyboard.press("Enter")
        if await self._wait_field_cleared(box, wait):
            return

        leftover = clean(await self._field_text(box))
        await self.clear_input()
        await self._dump("send-rejected", leftover)
        raise DriverCrashed(
            "сообщение не отправилось: поле ввода не очистилось после Enter и клика по кнопке",
            hint=(
                "Страница, вероятно, всё ещё генерирует предыдущий ответ или поле "
                "заблокировано. Смотри data/debug/ и `python -m bridge.cli probe --save`; "
                "помогает `python -m bridge.cli reset`."
            ),
        )

    async def send(self, prompt: str, *, reasoning: bool = False) -> AsyncIterator[str]:
        """Вставить текст, отправить с проверкой и отдавать накопленный текст ответа."""
        self.truncated = False
        self._detail = None
        page = self._require_page()

        # 1. странице надо быть свободной — иначе отправка физически невозможна
        if not await self.ensure_idle():
            raise DriverCrashed(
                "страница чата не переходит в состояние «свободна»",
                hint="Нажми Stop в открытом окне чата или выполни `python -m bridge.cli reset`.",
            )

        box = await self._locator("input")
        if reasoning or self._reasoning_on:
            await self._ensure_reasoning(reasoning)

        # 2. снимок «что уже есть», чтобы отличить новый ответ от старого
        previous, _ = await self._answer_texts()

        # 3. ввод + отправка с проверкой
        await self._write_prompt(box, prompt)
        await self._submit(box)

        # 4. читаем новый блок
        async for snapshot in self._read_answer(previous, prompt):
            yield snapshot

    async def _read_answer(self, previous: list[str], prompt: str) -> AsyncIterator[str]:
        """Ждём появления нового блока ответа и отдаём его рост.

        Три фазы в одном цикле: «дождаться старта генерации» → «дождаться нового
        текста» → «читать, пока не уляжется». Разделение нужно, чтобы не отдать
        клиенту предыдущий ответ (это и был главный баг «не возвращает ответ»).
        """
        timeout = float(self.settings.generation_timeout_s)
        started = time.monotonic()
        deadline = started + timeout
        poll = max(0.08, self.settings.poll_ms / 1000)
        wait_start = min(timeout, float(self.settings.start_timeout_s))

        candidate = ""
        stable = 0
        ever_generating = False
        while time.monotonic() < deadline:
            await asyncio.sleep(poll)
            generating = await self._is_generating()
            ever_generating = ever_generating or generating
            texts, _ = await self._answer_texts()
            new = pick_answer(texts, previous, prompt)

            if new is not None and new != candidate:
                candidate = new
                stable = 0
                yield candidate
                continue

            stable += 1
            if candidate and is_settled(
                candidate,
                stable,
                generating=generating,
                ever_generating=ever_generating,
                needed=self.settings.stable_polls,
            ):
                self._detail = "готов"
                return

            if new is None and not candidate:
                # Ответ пришёл, но текстом совпал с прошлым — позиции при этом новые.
                if ever_generating and not generating and stable >= self.settings.stable_polls:
                    final = settled_answer(texts, prompt)
                    if final:
                        self._detail = "ответ совпал с предыдущим по тексту"
                        yield final
                        return
                # Ни нового текста, ни генерации: ждём разумный срок, а не весь таймаут.
                if not generating and time.monotonic() - started > wait_start:
                    break

        if not candidate:
            await self._dump("no-answer", prompt)
            raise GenerationTimeout(
                timeout,
                message=(
                    f"DeepSeek не вернул текст ответа за {timeout:.0f} с: поле опустело "
                    "(сообщение принято), но новый блок ответа не появился"
                ),
                hint=(
                    "Обычно это сменившийся селектор `assistant_message`. Выполни "
                    "`python -m bridge.cli probe --save` — дамп в data/debug/ покажет, "
                    "как теперь называются блоки, и добавь их в selectors.json."
                ),
            )
        # Дождались снапшотов, но генерация не завершилась: отдаём накопленное и
        # вешаем флаг обрезки — движок сделает из этого finish_reason="length".
        self.truncated = True
        self._detail = f"генерация прервана по таймауту {timeout:.0f}s"
        yield candidate

    # ------------------------------------------------------------------ диагностика

    async def _dump(self, reason: str, note: str = "") -> Path | None:
        """Снимок страницы в момент сбоя: без него починка селекторов — угадывание.

        Пишем именно в момент ошибки (а не потом, «на глазок» из probe), потому
        что через пару секунд состояние DOM уже другое.
        """
        if not self.settings.debug_dump:
            return None
        try:
            directory = self.settings.debug_dir
            directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            meta = directory / f"{stamp}-{reason}.txt"
            meta.write_text(
                f"reason={reason}\nurl={getattr(self._page, 'url', '?')}\ndetail={self._detail}\n"
                f"note={note[:500]}\nselectors={json.dumps(self.selectors.as_dict(), ensure_ascii=False)}\n",
                encoding="utf-8",
            )
            data = await self.probe(reason=reason, save=False)
            out = directory / f"{stamp}-{reason}.json"
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return out
        except Exception:
            return None

    async def probe(self, *, reason: str = "manual", save: bool | None = None) -> dict[str, Any]:
        """Что реально видно на странице: счётчики по каждому варианту селектора.

        Нужен, когда DeepSeek меняет вёрстку: по `matched: false` сразу видно,
        какой ключ править в `selectors.json`.
        """
        page = self._page
        if page is None or page.is_closed():
            return {"driver": self.name, "running": False}

        selectors: dict[str, Any] = {}
        for key, raw in self.selectors.as_dict().items():
            detail: Any = {"matched": False}
            for selector in self.selectors.candidates(key):
                try:
                    count = await page.locator(selector).count()
                except Exception as exc:
                    detail = {"error": str(exc)[:120], "selector": selector}
                    break
                if count:
                    detail = {"selector": selector, "count": count}
                    break
            selectors[key] = detail

        buttons: list[str] = []
        try:
            buttons = await page.eval_on_selector_all(
                "button",
                "els => els.slice(0, 60).map(e => e.getAttribute('aria-label') || e.textContent.trim().slice(0,40))",
            )
        except Exception:
            pass

        answers, _ = await self._answer_texts()
        box = await self._locator("input", required=False)
        payload: dict[str, Any] = {
            "driver": self.name,
            "running": True,
            "reason": reason,
            "url": page.url,
            "title": await page.title(),
            "generating": await self._is_generating(),
            "reasoning_on": self._reasoning_on,
            "field_text": (clean(await self._field_text(box))[:160] if box is not None else None),
            "answer_tail": [clean(text)[:300] for text in answers[-3:]],
            "selectors": selectors,
            "buttons": [b for b in buttons if b],
            "suggested_fix": [key for key, value in selectors.items() if isinstance(value, dict) and not value.get("count")],
        }
        if save or (save is None and self.settings.debug_dump):
            try:
                directory = self.settings.debug_dir
                directory.mkdir(parents=True, exist_ok=True)
                out = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-probe.json"
                out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                payload["saved_to"] = str(out)
                try:
                    html = await page.content()
                    (out.with_suffix(".html")).write_text(html, encoding="utf-8")
                except Exception:
                    pass
            except Exception:
                pass
        return payload
