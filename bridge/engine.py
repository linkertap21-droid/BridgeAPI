"""Движок моста: очередь генераций, контекст диалога, OpenAI-совместимый результат.

Почему тут вообще есть очередь: у нас одна вкладка браузера, а значит одна
одновременная генерация. Параллельные HTTP-запросы невозможно мультиплексировать —
клиент либо ждёт своей очереди, либо честно получает 429.

Второй нетривиальный момент: у веб-чата нет понятия «отправь массив messages»,
есть только поле ввода. Поэтому решение «что именно отправить» принимает движок:
  * если сессия «жива» в браузере и история совпадает — отправляем только
    последнее сообщение (контекст уже есть на странице);
  * иначе — начинаем новый диалог и отправляем один промпт с транскриптом.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

from .config import Settings
from .drivers.base import ConversationDriver, DriverStatus
from .errors import Busy, DriverNotReady, NeedLogin
from .sessions import Session, SessionStore

OPENAI_MODELS = (
    {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek-web", "reasoning": False},
    {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek-web", "reasoning": True},
)


def estimate_tokens(text: str) -> int:
    """Порядковая оценка токенов — точного токенизатора у нас нет намеренно."""
    return max(1, round(len(text) / 4)) if text else 0


@dataclass
class Completion:
    text: str
    session_id: str
    model: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rebuilt_context: bool = False  # True = мост начал новый диалог и переслал историю
    elapsed_s: float = 0.0

    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


@dataclass
class StreamHandle:
    """Сюда движок кладёт итоговый Completion по окончании стрима."""

    completion: Completion | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _norm(role: str, content: str) -> tuple[str, str]:
    return (role or "user", (content or "").strip())


def _norm_message(message: dict[str, str]) -> tuple[str, str]:
    return _norm(message.get("role", "user"), message.get("content", ""))


def diverges(client_history: list[dict[str, str]], stored: list[dict[str, str]]) -> bool:
    """Клиент прислал историю, которая расходится с тем, что реально на странице.

    Сравниваем только пересечение. Клиент может историю не присылать вовсе (тогда
    расходиться нечему) либо прислать полный массив messages — тогда он обязан
    совпадать с нашей записью, иначе страница «помнит» не то, что мы думаем.
    """
    for sent, stored_turn in zip(client_history, stored):
        if _norm_message(sent) != _norm(stored_turn["role"], stored_turn["content"]):
            return True
    return False


def plan(session: Session, messages: list[dict[str, str]]) -> tuple[str, bool]:
    """Что отправить в поле ввода и надо ли начинать в браузере новый диалог.

    Три случая, которые реально встречаются:
      1. страница живая и история не противоречит нашей → отправляем последнее
         сообщение дословно (весь контекст уже на странице);
      2. страница не живая (после рестарта моста / new chat) → донести приходится
         нашу сохранённую историю + новое сообщение одним промптом;
      3. клиент прислал другую историю (редактировал сообщения, как в UI ChatGPT) →
         доверять странице нельзя, пересобираем с нуля.
    """
    if not messages:
        return "", True
    last = messages[-1]
    if session.chat_live and not diverges(messages[:-1], session.transcript()):
        return (last.get("content") or "").strip(), False

    stored = session.transcript()
    if stored and not diverges(messages[:-1], stored):
        # клиент экономный (прислал только новое) — восстанавливаем историю сами
        replay = stored + [last]
    else:
        replay = list(messages)
    return build_prompt(replay, with_history=len(replay) > 1), True


def build_prompt(messages: list[dict[str, str]], *, with_history: bool) -> str:
    """Свернуть messages в один текст для поля ввода.

    Важно: если пересылать нечего (первое сообщение в новом диалоге), отправляем
    его дословно. Оборачивать в «продолжи диалог» имеет смысл только тогда, когда
    внутри действительно есть история или системные инструкции — иначе модель
    начинает объяснять, что она прочитала историю.
    """
    if not messages:
        return ""
    last = (messages[-1].get("content") or "").strip()
    if not with_history:
        return last

    system = "\n\n".join(
        (m.get("content") or "").strip() for m in messages[:-1] if m.get("role") == "system"
    ).strip()
    history = [m for m in messages[:-1] if m.get("role") in {"user", "assistant"}]
    if not system and not history:
        return last

    parts: list[str] = []
    if system:
        parts.append(f"Инструкции (учитывай их во всём ответе):\n{system}")
    if history:
        lines = []
        for message in history:
            who = "Пользователь" if message.get("role") == "user" else "Ассистент"
            lines.append(f"{who}: {(message.get('content') or '').strip()}")
        parts.append("Вот история нашего диалога:\n\n" + "\n\n".join(lines))
        parts.append(
            "Продолжи диалог: ответь только на последнее сообщение пользователя, "
            "без пояснений про историю.\n\nПоследнее сообщение пользователя:\n" + last
        )
    else:
        parts.append(last)
    return "\n\n---\n\n".join(parts)


class Engine:
    """Связка «HTTP-клиент ↔ драйвер браузера ↔ локальная история»."""

    def __init__(
        self,
        settings: Settings,
        driver: ConversationDriver,
        store: SessionStore | None = None,
    ) -> None:
        self.settings = settings
        self.driver = driver
        self.store = store or SessionStore(settings.sessions_dir)
        self._lock = asyncio.Lock()
        self._waiters = 0
        self._tasks: set = set()  # fire-and-forget задачи санации/дампов
        self._stats: dict[str, Any] = {
            "requests": 0,      # сколько раз просили сгенерировать
            "completed": 0,     # ...из них дошло до клиента непустым ответом
            "failed": 0,        # бросили ошибку (включая обрезку по таймауту)
            "started_at": time.time(),
        }

    # ------------------------------------------------------------------ жизненный цикл

    async def start(self) -> None:
        self.settings.ensure_dirs()
        await self.driver.start()

    async def stop(self) -> None:
        await self.driver.stop()

    async def status(self) -> dict[str, Any]:
        driver_status: DriverStatus = await self.driver.status()
        return {
            "ok": bool(driver_status.running and driver_status.logged_in),
            "driver": self.driver.name,
            "browser": driver_status.to_dict(),
            "queue": {
                "busy": self._lock.locked(),
                "waiting": self._waiters,
                "max_waiters": self.settings.max_waiters,
            },
            "sessions": len(list(self.store.iter_all())),
            "stats": {**self._stats, "uptime_s": round(time.time() - self._stats["started_at"], 1)},
            "config": self.settings.public_dict(),
        }

    async def probe(self, **kwargs: Any) -> dict[str, Any]:
        return await self.driver.probe(**kwargs)

    async def reset(self) -> dict[str, Any]:
        """Принудительно вернуть страницу в покой (кнопка Stop + чистка поля ввода).

        Ручная кнопка «всё застыло»: после обрыва или упавшей вёрстки мост умеет
        восстановиться сам, но этот эндпоинт позволяет сделать это по требованию.
        """
        ensure = getattr(self.driver, "ensure_idle", None)
        idle = bool(await ensure()) if ensure is not None else False
        status = await self.driver.status()
        return {"idle": idle, "browser": status.to_dict(), "driver": self.driver.name}

    async def login(self) -> dict[str, Any]:
        ok = await self.driver.login(self.settings.login_timeout_s)
        if not ok:
            raise NeedLogin()
        return {"logged_in": True, "detail": "профиль сохранён, можно пользоваться"}

    # ------------------------------------------------------------------ очередь

    async def _acquire(self) -> None:
        if self._waiters >= self.settings.max_waiters and self._lock.locked():
            raise Busy(0.0)
        self._waiters += 1
        try:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=self.settings.queue_wait_s)
            except asyncio.TimeoutError as exc:
                raise Busy(self.settings.queue_wait_s) from exc
        finally:
            self._waiters -= 1

    def _release(self) -> None:
        if self._lock.locked():
            self._lock.release()

    # ------------------------------------------------------------------ основной API

    def resolve_model(self, model: str | None) -> tuple[str, bool]:
        name = (model or "deepseek-chat").strip()
        reasoning = name in {"deepseek-reasoner", "deepseek-thinking", "reasoner"}
        return ("deepseek-reasoner" if reasoning else "deepseek-chat"), reasoning

    def prepare(
        self, messages: list[dict[str, str]], *, session_id: str | None, model: str
    ) -> tuple[Session, str, bool]:
        """(сессия, промпт для поля ввода, нужен ли новый диалог в браузере)."""
        session = self.store.get_or_create(session_id, model=model)
        prompt, needs_new_chat = plan(session, messages)
        if not prompt:
            from .errors import InvalidRequest

            raise InvalidRequest("Пустой запрос: messages[-1].content должен быть непустым.")
        return session, prompt, needs_new_chat

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str | None = None,
        model: str | None = None,
        handle: StreamHandle | None = None,
    ) -> AsyncIterator[str]:
        """Отдаёт дельты текста; итоговый Completion кладёт в `handle`."""
        handle = handle or StreamHandle()
        model_name, reasoning = self.resolve_model(model)
        session, prompt, with_new_chat = self.prepare(messages, session_id=session_id, model=model_name)

        await self._acquire()
        started = time.monotonic()
        text = ""
        sent_len = 0
        completed = False
        needs_recover = False
        try:
            status = await self.driver.status()
            if not status.running:
                raise DriverNotReady(
                    "Браузер моста не запущен.", hint="`python -m bridge.cli login` или `serve`."
                )
            if status.logged_in is False:
                raise NeedLogin()

            if with_new_chat:
                await self.driver.begin_chat()
            self._stats["requests"] += 1

            async for snapshot in self._send(prompt, reasoning=reasoning):
                if not isinstance(snapshot, str):
                    continue
                if len(snapshot) > sent_len:
                    delta = snapshot[sent_len:]
                    sent_len = len(snapshot)
                    text = snapshot
                    yield delta
            completed = True
        except (asyncio.CancelledError, GeneratorExit):
            # Клиент ушёл или поток закрыли: ждать здесь нельзя (генератор уже
            # закрывается), поэтому санацию ставим задачей и сразу отдаём отмену.
            self._stats["failed"] += 1
            self._schedule_recover()
            raise
        except Exception:
            self._stats["failed"] += 1
            needs_recover = True
            raise
        finally:
            elapsed = time.monotonic() - started
            final_text = text.strip()
            # Неудачный обмен в историю не пишем: фантомная реплика ассистента
            # потом ушла бы в модель при пересборке контекста.
            if completed and final_text:
                self._stats["completed"] += 1
                self.store.append(session, "user", (messages[-1].get("content") or "").strip() or prompt)
                self.store.append(session, "assistant", final_text)
            self.store.mark_live(session, live=bool(completed and final_text), model=model_name)
            # движок страницы не присылает finish-сигнал, поэтому «обрезано таймаутом»
            # сообщает сам драйвер — иначе клиент получил бы честное-looking "stop".
            truncated = bool(getattr(self.driver, "truncated", False))
            handle.completion = Completion(
                text=final_text,
                session_id=session.id,
                model=model_name,
                finish_reason="length" if (truncated or not completed) else "stop",
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(final_text),
                rebuilt_context=with_new_chat,
                elapsed_s=round(elapsed, 2),
            )
            handle.meta = {"driver": self.driver.name, "completed": completed}
            if needs_recover:
                # Страница после сбоя остаётся «говорящей» и с текстом в поле:
                # без this шаг следующий запрос пришлось бы начинать вручную.
                await self._recover()
            self._release()

    async def _recover(self) -> None:
        """Попросить драйвер привести страницу в покой. Ошибки глотаем: это уже починка."""
        recover = getattr(self.driver, "recover", None)
        if recover is None:
            return
        try:
            await recover()
        except Exception:
            pass

    def _schedule_recover(self) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self._recover())
        except RuntimeError:  # нет цикла — например, синхронный вызов из теста
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self, prompt: str, *, reasoning: bool) -> AsyncIterator[str]:
        stream = self.driver.send(prompt, reasoning=reasoning)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str | None = None,
        model: str | None = None,
    ) -> Completion:
        handle = StreamHandle()
        chunks: list[str] = []
        async for delta in self.stream(messages, session_id=session_id, model=model, handle=handle):
            chunks.append(delta)
        if handle.completion is None:  # pragma: no cover - страховка
            return Completion(
                text="".join(chunks), session_id=session_id or "", model=model or "deepseek-chat"
            )
        return handle.completion


def messages_from(messages: Iterable[dict[str, str]] | None, prompt: str | None) -> list[dict[str, str]]:
    """Нормализовать вход: `messages` (OpenAI) или `prompt` (упрощённый /v1/chat)."""
    if messages:
        out = []
        for item in messages:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, list):  # мультимодальный content: берём текстовые части
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                out.append({"role": str(item.get("role") or "user"), "content": str(content or "")})
        return out
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    from .errors import InvalidRequest

    raise InvalidRequest("Нужно передать либо `messages`, либо `prompt`.")
