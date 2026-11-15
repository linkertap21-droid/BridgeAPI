"""Реестр драйверов.

Новый бэкенд (например, «DeepSeek API по ключу» или другой веб-чат) = один файл
+ одна строка в `_REGISTRY`. Ни HTTP-слой, ни движок править не надо.
"""

from __future__ import annotations

from ..config import Settings
from .base import DEFAULT_SELECTORS, ConversationDriver, DriverStatus, Selectors, ScriptedDriver


def build_driver(settings: Settings) -> ConversationDriver:
    name = (settings.provider or "deepseek_web").strip().lower()
    if name in _SCRIPTED:
        return _SCRIPTED[name](settings)
    if name in {"deepseek_web", "deepseek", "web", "webchat"}:
        from .deepseek_web import DeepSeekWebChatDriver

        return DeepSeekWebChatDriver(settings, Selectors.load(settings.selectors_file))
    known = ", ".join(sorted(set(_SCRIPTED) | {"deepseek_web"}))
    raise ValueError(f"неизвестный BRIDGE_PROVIDER={name!r}. Доступно: {known}")


FAKE_REPLY_DEFAULT = (
    "Это ответ фейкового драйвера. Он нужен, чтобы проверять HTTP-контракт моста "
    "без браузера и без лимитов DeepSeek."
)


def _fake_reply_driver(settings: Settings) -> ScriptedDriver:
    from ..config import _str  # env читается здесь, чтобы не таскать os.environ в Settings

    return ScriptedDriver(reply=_str("BRIDGE_FAKE_REPLY", FAKE_REPLY_DEFAULT))


_SCRIPTED = {
    "fake": _fake_reply_driver,
    "scripted": _fake_reply_driver,
}

__all__ = [
    "build_driver",
    "ConversationDriver",
    "DriverStatus",
    "Selectors",
    "ScriptedDriver",
    "DEFAULT_SELECTORS",
]
