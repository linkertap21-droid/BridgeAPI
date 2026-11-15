"""Драйвер `chat.deepseek.com` через управляемый Playwright Chromium.

Как это работает и почему именно так
------------------------------------
Никакого API-ключа у нас нет, поэтому «точкой входа» в DeepSeek остаётся
обычная веб-страница чата. Мост держит один Chromium с persistent-профилем
(папка `data/browser-profile`), в котором один раз сохраняются cookies после
ручного логина. Дальше он делает ровно то, что делает человек:
вставляет текст в поле ввода, жмёт Enter и читает innerText последнего
ответа, отдавая его нарастающими снапшотами.

Что из этого следует для кода:
  * один браузер = одна одновременная генерация (очередь — в `bridge/engine.py`);
  * селекторы вынесены в `selectors.json`, потому что вёрстка чата меняется;
  * `send()` отдаёт накопленный текст, а не дельты — diff считает движок;
  * падение страницы отличается от «надо перелогиниться» — у клиента разные действия.

Playwright импортируется лениво: без него можно запускать API, тесты и `fake`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from ..config import Settings
from ..drivers.base import DriverStatus, Selectors
from ..errors import DriverCrashed, DriverNotReady, GenerationTimeout, NeedLogin


class DeepSeekWebChatDriver:
    """Одна вкладка чата DeepSeek, управляемая как обычный пользователь."""

    name = "deepseek_web"

    def __init__(self, settings: Settings, selectors: Selectors | None = None) -> None:
        self.settings = settings
        self.truncated = False
        self.selectors = selectors or Selectors.load(settings.selectors_file)
        self._pw: Any = None
        self._ctx: Any = None
        self._page: Any = None
        self._logged_in: bool | None = None
        self._headless: bool | None = None
        self._detail: str | None = "не запущен"
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
            except Exception as exc:  # noqa: BLE001 - источник ошибки знает только Playwright
                await self._shutdown()
                reason = (str(exc).strip().splitlines() or [repr(exc)])[0]
                raise DriverCrashed(
                    f"не удалось запустить Chromium: {reason[:200]}",
                    hint=(
                        "Обычно это не скачанный браузер: выполни `pip install playwright && "
                        "playwright install chromium`. Реже — порт/профиль заняты другим экземпляром моста."
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
                await closer.close() if hasattr(closer, "close") else await closer.stop()
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
        page = self._require_page()
        for selector in self.selectors.candidates("input"):
            try:
                await page.locator(selector).first.wait_for(
                    state="visible", timeout=max(500, timeout_s * 1000 // max(1, len(self.selectors.candidates("input"))))
                )
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

    # ------------------------------------------------------------------ диалог

    async def begin_chat(self) -> None:
        """Новый диалог: клик по кнопке, иначе — перезагрузка страницы чата."""
        page = self._require_page()
        for selector in self.selectors.candidates("new_chat_button"):
            try:
                button = page.locator(selector).first
                if await button.count() > 0:
                    await button.click(timeout=3_000)
                    await asyncio.sleep(0.4)
                    return
            except Exception:
                continue
        await self._open_chat()

    async def _locate(self, key: str, *, required: bool = True) -> Any:
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
                f"на странице не найден элемент {key!r} (селекторы: {self.selectors.get(key)})"
            )
        return None

    async def _message_count(self) -> int:
        page = self._require_page()
        for selector in self.selectors.candidates("assistant_message"):
            try:
                count = await page.locator(selector).count()
                if count:
                    return count
            except Exception:
                continue
        return 0

    async def _last_answer_text(self) -> str:
        page = self._require_page()
        for selector in self.selectors.candidates("assistant_message"):
            try:
                locator = page.locator(selector)
                count = await locator.count()
                if count:
                    return (await locator.nth(count - 1).inner_text(timeout=2_000)) or ""
            except Exception:
                continue
        return ""

    async def _is_generating(self) -> bool:
        page = self._require_page()
        for selector in self.selectors.candidates("stop_button"):
            try:
                if await page.locator(selector).count() > 0:
                    return True
            except Exception:
                continue
        return False

    async def _toggle_reasoning(self) -> bool:
        for selector in self.selectors.candidates("think_toggle"):
            try:
                button = self._page.locator(selector).first
                if await button.count() > 0:
                    await button.click(timeout=2_000)
                    return True
            except Exception:
                continue
        return False

    async def send(self, prompt: str, *, reasoning: bool = False) -> AsyncIterator[str]:
        """Вставить текст, отправить и отдавать накопленный текст ответа."""
        self.truncated = False
        page = self._require_page()
        box = await self._locate("input")
        if reasoning:
            await self._toggle_reasoning()

        before = await self._message_count()
        try:
            await box.click(timeout=5_000)
            try:
                await box.fill(prompt, timeout=3_000)
                if (await box.input_value(timeout=1_000) if _is_inputy(box) else None) != prompt:
                    raise RuntimeError("fill не сработал")
            except Exception:
                # contenteditable: fill не умеет — вставляем через клавиатуру
                await page.keyboard.insert_text(prompt)
        except Exception as exc:
            raise DriverCrashed(f"не удалось ввести текст в поле чата: {exc}") from exc

        sent = False
        try:
            button = await self._locate("send_button", required=False)
            if button is not None and await button.is_enabled():
                await button.click(timeout=2_000)
                sent = True
        except Exception:
            sent = False
        if not sent:
            await page.keyboard.press("Enter")

        async for snapshot in self._stream_answer(before):
            yield snapshot
        if reasoning:
            await self._toggle_reasoning()

    async def _stream_answer(self, before: int) -> AsyncIterator[str]:
        """Ждём новый пузырёк ответа и читаем его, пока генерация не остановится."""
        timeout = self.settings.generation_timeout_s
        deadline = time.monotonic() + timeout
        poll = max(0.08, self.settings.poll_ms / 1000)

        last = ""
        stable = 0
        appeared = False
        while time.monotonic() < deadline:
            await asyncio.sleep(poll)
            count = await self._message_count()
            if not appeared and count <= before and not await self._is_generating():
                # ещё не начал отвечать (страница могла перезагрузиться)
                continue
            appeared = True
            text = await self._last_answer_text()
            if len(text) > len(last):
                last = text
                stable = 0
                yield last
            elif text:
                stable += 1
            if appeared and last and stable >= self.settings.stable_polls and not await self._is_generating():
                return
        if not last:
            raise GenerationTimeout(timeout)
        # Снапшоты были, но генерация не завершилась: отдаём накопленное и вешаем
        # флаг обрезки — движок сделает из этого finish_reason="length".
        self.truncated = True
        self._detail = f"генерация прервана по таймауту {timeout}s"
        yield last

    # ------------------------------------------------------------------ диагностика

    async def probe(self) -> dict[str, Any]:
        """Что видно на странице: нужны, когда DeepSeek поменяет вёрстку."""
        page = self._page
        if page is None or page.is_closed():
            return {"driver": self.name, "running": False}
        selectors: dict[str, Any] = {}
        for key in self.selectors.as_dict():
            found = None
            for selector in self.selectors.candidates(key):
                try:
                    count = await page.locator(selector).count()
                except Exception as exc:
                    found = {"error": str(exc)[:120]}
                    break
                if count:
                    found = {"selector": selector, "count": count}
                    break
            selectors[key] = found or {"matched": False}
        aria_labels: list[str] = []
        try:
            aria_labels = await page.eval_on_selector_all(
                "button",
                "els => els.slice(0, 60).map(e => e.getAttribute('aria-label') || e.textContent.trim().slice(0,40))",
            )
        except Exception:
            pass
        return {
            "driver": self.name,
            "running": True,
            "url": page.url,
            "title": await page.title(),
            "selectors": selectors,
            "buttons": [a for a in aria_labels if a],
        }


def _is_inputy(locator: Any) -> bool:
    return locator is not None
