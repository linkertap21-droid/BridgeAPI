"""Конфигурация моста.

Всё настраивается переменными окружения (или файлом `.env` в корне репозитория),
чтобы мост можно было запускать без правки кода: `python -m bridge.cli serve`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | str = REPO_ROOT / ".env", *, override: bool = False) -> int:
    """Минимальный загрузчик `.env` — чтобы не тянуть зависимость python-dotenv.

    Поддерживает `KEY=value`, `export KEY=value`, кавычки и комментарии.
    Уже выставленные переменные окружения не перезаписывает (если не `override=True`).
    """
    file = Path(path)
    if not file.is_file():
        return 0
    loaded = 0
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # не роняем процесс ради опечатки в конфиге
        raise ValueError(f"{name} должно быть целым числом, получено: {value!r}") from exc


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    token: str | None = None
    provider: str = "deepseek_web"
    log_level: str = "info"

    data_dir: Path = REPO_ROOT / "data"
    selectors_file: Path | None = None

    deepseek_url: str = "https://chat.deepseek.com"
    headless: bool = False
    login_timeout_s: int = 600
    ready_timeout_s: int = 20

    generation_timeout_s: int = 240
    start_timeout_s: int = 25
    send_wait_s: int = 10
    idle_wait_s: int = 45
    poll_ms: int = 350
    stable_polls: int = 3
    answer_window: int = 10
    submit: str = "auto"
    debug_dump: bool = True
    max_waiters: int = 8
    queue_wait_s: int = 120
    keep_browser_open: bool = True

    cors_origin_regex: str = r"^http://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$"
    cors_allow_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, dotenv: Path | str | None = None) -> "Settings":
        if dotenv is not None:
            load_dotenv(dotenv)
        origins = _str("BRIDGE_CORS_ALLOW_ORIGINS", "")
        return cls(
            host=_str("BRIDGE_HOST", "127.0.0.1"),
            port=_int("BRIDGE_PORT", 8787),
            token=_str("BRIDGE_TOKEN", "") or None,
            provider=_str("BRIDGE_PROVIDER", "deepseek_web").strip().lower(),
            log_level=_str("BRIDGE_LOG_LEVEL", "info"),
            data_dir=_path("BRIDGE_DATA_DIR", REPO_ROOT / "data"),
            selectors_file=_path("BRIDGE_SELECTORS_FILE", REPO_ROOT / "selectors.json")
            if (REPO_ROOT / "selectors.json").is_file() or os.environ.get("BRIDGE_SELECTORS_FILE")
            else None,
            deepseek_url=_str("DEEPSEEK_CHAT_URL", "https://chat.deepseek.com").rstrip("/"),
            headless=_bool("BRIDGE_HEADLESS", False),
            login_timeout_s=_int("BRIDGE_LOGIN_TIMEOUT_S", 600),
            ready_timeout_s=_int("BRIDGE_READY_TIMEOUT_S", 20),
            generation_timeout_s=_int("BRIDGE_GENERATION_TIMEOUT_S", 240),
            start_timeout_s=_int("BRIDGE_START_TIMEOUT_S", 25),
            send_wait_s=_int("BRIDGE_SEND_WAIT_S", 10),
            idle_wait_s=_int("BRIDGE_IDLE_WAIT_S", 45),
            poll_ms=_int("BRIDGE_POLL_MS", 350),
            stable_polls=_int("BRIDGE_STABLE_POLLS", 3),
            answer_window=_int("BRIDGE_ANSWER_WINDOW", 10),
            submit=_str("BRIDGE_SUBMIT", "auto").lower(),
            debug_dump=_bool("BRIDGE_DEBUG_DUMP", True),
            max_waiters=_int("BRIDGE_MAX_WAITERS", 8),
            queue_wait_s=_int("BRIDGE_QUEUE_WAIT_S", 120),
            keep_browser_open=_bool("BRIDGE_KEEP_BROWSER_OPEN", True),
            cors_origin_regex=_str(
                "BRIDGE_CORS_ORIGIN_REGEX",
                r"^http://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$",
            ),
            cors_allow_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )

    def override(self, **changes: Any) -> "Settings":
        clean = {k: v for k, v in changes.items() if v is not None}
        return replace(self, **clean) if clean else self

    @property
    def profile_dir(self) -> Path:
        """Профиль Chromium: здесь живут cookies DeepSeek, логиниться нужно один раз."""
        return self.data_dir / "browser-profile"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def debug_dir(self) -> Path:
        """Дампы страницы при сбоях: без них правка селекторов — угадывание."""
        return self.data_dir / "debug"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        """Для /health: то, что безопасно показывать (токена тут нет)."""
        return {
            "provider": self.provider,
            "target_url": self.deepseek_url,
            "headless": self.headless,
            "generation_timeout_s": self.generation_timeout_s,
            "poll_ms": self.poll_ms,
            "stable_polls": self.stable_polls,
            "submit": self.submit,
            "debug_dump": self.debug_dump,
            "data_dir": str(self.data_dir),
            "auth_required": bool(self.token),
        }


def default_settings(**changes: Any) -> Settings:
    """Settings для тестов/CLI без чтения `.env`."""
    base = Settings()
    return base.override(**changes)


def as_dict(settings: Settings) -> dict[str, Any]:
    data = {f.name: getattr(settings, f.name) for f in settings.__dataclass_fields__.values()}
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    data["cors_allow_origins"] = list(data["cors_allow_origins"])
    return data
