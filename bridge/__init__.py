"""Мост между локальным API и DeepSeek (веб-чат без ключа)."""

from __future__ import annotations

from .config import REPO_ROOT, Settings, load_dotenv
from .engine import Completion, Engine, build_prompt, estimate_tokens, messages_from
from .errors import (
    BridgeError,
    Busy,
    DriverCrashed,
    DriverNotReady,
    GenerationTimeout,
    InvalidRequest,
    NeedLogin,
)
from .sessions import Session, SessionStore

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "load_dotenv",
    "REPO_ROOT",
    "Engine",
    "Completion",
    "SessionStore",
    "Session",
    "build_prompt",
    "messages_from",
    "estimate_tokens",
    "BridgeError",
    "NeedLogin",
    "Busy",
    "DriverNotReady",
    "DriverCrashed",
    "GenerationTimeout",
    "InvalidRequest",
    "__version__",
]
