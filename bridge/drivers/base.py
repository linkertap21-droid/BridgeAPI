"""Интерфейс драйвера диалога + селекторы страницы.

Мост не знает, ЧТО происходит с запросом. Он знает, что у него есть объект,
который умеет `start/stop/status/begin_chat/send`. Это даёт:
  * `deepseek_web` — реальный драйвер (управляемый Chromium);
  * `fake` — скриптовый драйвер для тестов и отладки API без браузера;
  * место под будущий драйвер «официальный API по ключу» одним новым файлом.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, runtime_checkable

# Ключевые точки страницы. A/B-эксперименты и редизайны ломают вёрстку — поэтому
# значение может перечислять несколько вариантов через "||": пробуем по очереди.
# Переопределить без правки кода можно файлом selectors.json.
DEFAULT_SELECTORS: dict[str, str] = {
    "input": 'textarea#chat-input || div[contenteditable="true"][id="chat-input"] || #chat-input || textarea[placeholder]',
    "send_button": 'button[aria-label="Send"] || button[aria-label*="Send"i] || button[aria-label="Отправить"] '
    '|| button[data-testid="send"] || [id*="input-box"] button:last-child',
    "stop_button": 'button[aria-label="Stop"] || button[aria-label*="Stop"i] || button[aria-label*="Останов"] '
    '|| button[data-testid="stop"] || [class*="stop-generating"]',
    # Тот же селектор матчит и бабл пользователя: потому ответ определяется диффом
    # по содержимому (`pick_answer`), а не «сколько блоков стало».
    "assistant_message": '.message-content .ds-markdown || .chat-response .ds-markdown || [class*="ds-markdown"] '
    "|| [class*='markdown-body'] || .markdown-body",
    "user_message": ".chat-box .ds-markdown || .chat-prompt .ds-markdown",
    "new_chat_button": 'button[aria-label="Create a new chat"] || button[aria-label="New chat"] || .new-chat-button',
    "sign_in_marker": 'input[type="password"] || button:has-text("Log in") || div[class*="sign-in"]',
    "think_toggle": 'button[aria-label*="DeepThink"] || div:has-text("深度思考")',
}

_MISSING = object()


@dataclass
class DriverStatus:
    """Ответ /v1/auth/status и часть /health."""

    running: bool = False
    logged_in: bool | None = None
    url: str | None = None
    headless: bool | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "logged_in": self.logged_in,
            "url": self.url,
            "headless": self.headless,
            "detail": self.detail,
        }


class Selectors:
    """Обёртка над словарём селекторов: разбор "||"-альтернатив и merged-override."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._map = dict(DEFAULT_SELECTORS)
        if mapping:
            self._map.update({k: str(v) for k, v in mapping.items() if v})

    @classmethod
    def load(cls, path: Path | str | None) -> "Selectors":
        if not path:
            return cls()
        file = Path(path)
        if not file.is_file():
            return cls()
        raw = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("selectors.json должен быть объектом {ключ: селектор}")
        # ключи с подчёркиванием — комментарии в JSON, их не надо искать на странице
        return cls({k: v for k, v in raw.items() if not k.startswith("_")})

    def candidates(self, key: str) -> list[str]:
        raw = self._map.get(key, "")
        return [part.strip() for part in raw.split("||") if part.strip()]

    def get(self, key: str, default: Any = _MISSING) -> str:
        if key in self._map:
            return self._map[key]
        if default is _MISSING:
            raise KeyError(key)
        return default

    def as_dict(self) -> dict[str, str]:
        return dict(self._map)


@runtime_checkable
class ConversationDriver(Protocol):
    """Всё, что нужно движку от «браузера с чатом»."""

    name: str

    async def start(self) -> None:
        """Поднять браузер/страницу. Идемпотентно."""

    async def stop(self) -> None:
        """Закрыть всё. Идемпотентно, не бросает."""

    async def status(self) -> DriverStatus:
        ...

    async def login(self, timeout_s: int | None = None) -> bool:
        """Открыть headful-окно и дождаться ручного логина. True — залогинено."""

    async def begin_chat(self) -> None:
        """Начать новый диалог на странице (или перезагрузить чистый)."""

    async def send(self, prompt: str, *, reasoning: bool = False) -> AsyncIterator[str]:
        """Отправить промпт и отдавать ТЕКСТ НАКОПИТЕЛЬНО (не дельты).

        Движок сам превращает снапшоты в дельты — драйверу не надо считать diff.
        """

    async def ensure_idle(self, timeout_s: float | None = None) -> bool:
        """Привести страницу в покой (снять «генерирует», вычистить поле ввода)."""

    async def recover(self) -> None:
        """Вызывается движком ПОСЛЕ неудачной генерации.

        Без этого шага оборванная генерация оставляет страницу занятой, и
        следующий запрос упирается в заблокированное поле ввода — лечится только
        ручным «новым чатом». Никогда не бросает.
        """

    async def probe(self, *, reason: str = "manual", save: bool | None = None) -> dict[str, Any]:
        """Диагностика: какие селекторы находятся, что на странице."""

    # Необязательный, но уважаемый движком атрибут: `True`, если ответ оборвался
    # не потому, что модель закончила, а потому что истёк таймаут ожидания.
    # По нему finish_reason становится "length", а не вральным "stop".
    truncated: bool


class ScriptedDriver:
    """Драйвер-заглушка, воспроизводящий заданный ответ по кускам.

    Используется тестами и `BRIDGE_PROVIDER=fake` для проверки API без браузера.
    """

    name = "fake"

    truncated = False

    def __init__(self, reply: str = "Это ответ от фейкового драйвера.", chunk_size: int = 12) -> None:
        self.reply = reply
        self.idle_calls = 0
        self.recover_calls = 0
        self.chunk_size = chunk_size
        self.prompts: list[str] = []
        self.begin_chat_calls = 0
        self.running = False
        self.logged_in: bool | None = True
        self.reasoning_flags: list[bool] = []

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False

    async def status(self) -> DriverStatus:
        return DriverStatus(running=self.running, logged_in=self.logged_in, url="fake://chat")

    async def login(self, timeout_s: int | None = None) -> bool:
        self.logged_in = True
        return True

    async def begin_chat(self) -> None:
        self.begin_chat_calls += 1

    async def ensure_idle(self, timeout_s: float | None = None) -> bool:
        self.idle_calls += 1
        return True

    async def recover(self) -> None:
        # как в живом драйвере: восстановление = «успокоить страницу»
        self.recover_calls += 1
        await self.ensure_idle()

    async def send(self, prompt: str, *, reasoning: bool = False) -> AsyncIterator[str]:
        import asyncio

        self.prompts.append(prompt)
        self.reasoning_flags.append(reasoning)
        text = self.reply if "{prompt}" not in self.reply else self.reply.format(prompt=prompt)
        emitted = 0
        for end in range(self.chunk_size, len(text) + 1, self.chunk_size):
            emitted = end
            yield text[:end]
            await asyncio.sleep(0)
        if emitted < len(text):
            yield text  # хвост короче шага; дубликат «последнего снапшота» не отдаём

    async def probe(self, *, reason: str = "manual", save: bool | None = None) -> dict[str, Any]:
        return {"driver": self.name, "url": "fake://chat", "reason": reason, "selectors": {}}
