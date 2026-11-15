"""CLI моста: `python -m bridge.cli <команда>`.

Команды разбиты по ролям: один раз `login`, дальше `serve`, а `chat`/`status`/
`probe` — чтобы проверять мост без curl и без сторонних клиентов.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Settings


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(REPO_ROOT / ".env")
    return settings.override(
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        provider=getattr(args, "provider", None),
        headless=getattr(args, "headless", None),
        log_level=getattr(args, "log_level", None),
        token=getattr(args, "token", None),
    )


# --------------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _settings(args)
    if settings.host not in {"127.0.0.1", "localhost", "::1"} and not settings.token:
        print(
            "⚠️  мост слушает НЕ loopback и при этом без токена — любой в сети получит "
            "доступ к твоему аккаунту DeepSeek. Задай BRIDGE_TOKEN.",
            file=sys.stderr,
        )
    print(f"BridgeAPI → http://{settings.host}:{settings.port}  (docs: /docs, provider: {settings.provider})")
    print(f"    чат-мишень: {settings.deepseek_url}   профиль: {settings.profile_dir}")
    if settings.provider == "deepseek_web":
        print("    если ещё не логинился: сначала `python -m bridge.cli login`")
    uvicorn.run(
        "bridge.http:app_factory",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
    return 0


# --------------------------------------------------------------------------- login


async def _login(settings: Settings) -> int:
    from .drivers import build_driver

    driver = build_driver(settings)
    try:
        print("Открываю Chromium с чатом DeepSeek — войди в аккаунт в этом окне.")
        print(f"Профиль: {settings.profile_dir}  (логин сохраняется, повторять не нужно)")
        ok = await driver.login(settings.login_timeout_s)
    finally:
        if not settings.keep_browser_open:
            await driver.stop()
    if ok:
        print("✅ залогинено — можно запускать `python -m bridge.cli serve`")
        return 0
    print("❌ логин не завершён. Проверь, что окно не закрылось раньше времени.", file=sys.stderr)
    return 1


def cmd_login(args: argparse.Namespace) -> int:
    settings = _settings(args)
    return asyncio.run(_login(settings))


# --------------------------------------------------------------------------- status / probe


async def _engine_call(settings: Settings, what: str) -> Any:
    from .drivers import build_driver
    from .engine import Engine

    engine = Engine(settings, build_driver(settings))
    await engine.start()
    try:
        if what == "status":
            return await engine.status()
        if what == "probe":
            return await engine.probe()
        raise ValueError(what)
    finally:
        await engine.stop()


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_status(args: argparse.Namespace) -> int:
    settings = _settings(args)
    client = _maybe_remote(settings, args)
    if client is not None:
        _print_json(client.get("/health").json())
        return 0
    _print_json(asyncio.run(_engine_call(settings, "status")))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _print_json(asyncio.run(_engine_call(settings, "probe")))
    return 0


# --------------------------------------------------------------------------- chat


def cmd_chat(args: argparse.Namespace) -> int:
    import httpx

    settings = _settings(args)
    base = args.url or f"http://{settings.host}:{settings.port}"
    headers = {"Authorization": f"Bearer {args.token}"} if args.token or settings.token else {}
    session_id = args.session
    prompt = " ".join(args.message).strip() if args.message else ""

    with httpx.Client(base_url=base, timeout=httpx.Timeout(settings.generation_timeout_s + 30, connect=10)) as client:
        if prompt:
            text = _ask_once(client, headers, prompt, session_id, args.model, args.stream)
            print(text)
            return 0

        print("Интерактивный режим: пустая строка — выход, /new — новый диалог.")
        while True:
            try:
                line = input("ты > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                return 0
            if line == "/new":
                session_id = None
                print("(новая сессия)")
                continue
            try:
                answer, session_id = _ask_once(client, headers, line, session_id, args.model, args.stream, return_session=True)
                print(f"ds > {answer}")
            except httpx.HTTPStatusError as exc:
                print(f"ошибка {exc.response.status_code}: {exc.response.text[:400]}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"мост недоступен: {exc}", file=sys.stderr)
                return 1


def _ask_once(client: Any, headers: dict[str, str], prompt: str, session_id: str | None, model: str, stream: bool, *, return_session: bool = False) -> Any:
    body: dict[str, Any] = {"prompt": prompt, "model": model, "stream": stream}
    if session_id:
        body["session_id"] = session_id
    if not stream:
        response = client.post("/v1/chat", json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        if return_session:
            return data["reply"], data.get("session_id")
        return data["reply"]

    pieces: list[str] = []
    with client.stream("POST", "/v1/chat", json=body, headers=headers) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    pieces.append(delta)
                    print(delta, end="", flush=True)
    print()
    text = "".join(pieces)
    return (text, session_id) if return_session else text


# --------------------------------------------------------------------------- selftest


async def _selftest(settings: Settings) -> int:
    from fastapi.testclient import TestClient

    from .drivers.base import ScriptedDriver
    from .engine import Engine
    from .http import build_app
    from .sessions import SessionStore
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="bridge-selftest-"))
    test_settings = settings.override(data_dir=work, provider="fake")
    driver = ScriptedDriver(reply="Мост работает: {prompt}")
    app = build_app(
        test_settings,
        engine=Engine(test_settings, driver, SessionStore(test_settings.sessions_dir)),
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        first = client.post("/v1/chat", json={"prompt": "привет"}).json()
        sid = first["session_id"]
        second = client.post("/v1/chat", json={"prompt": "ещё раз", "session_id": sid}).json()
    print("health.ok        =", health["ok"], "| driver =", health["driver"])
    print("первый ответ     =", first["reply"][:70])
    print("второй ответ     =", second["reply"][:70])
    print("prompts драйвера =", driver.prompts)
    ok = bool(health["ok"]) and first["reply"] and "ещё раз" in driver.prompts[-1]
    print("✅ самопроверка пройдена" if ok else "❌ самопроверка провалилась")
    return 0 if ok else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    return asyncio.run(_selftest(_settings(args)))


def _maybe_remote(settings: Settings, args: argparse.Namespace) -> Any:
    """Если сервер уже запущен — спрашиваем его, а не лезем в браузер (иначе два Chrome)."""
    if getattr(args, "local", False):
        return None
    try:
        import httpx

        base = getattr(args, "url", None) or f"http://{settings.host}:{settings.port}"
        client = httpx.Client(base_url=base, timeout=3.0)
        client.get("/health")
        return client
    except Exception:
        return None


# --------------------------------------------------------------------------- парсер


def _add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Общие флаги работают и до подкоманды, и после неё.

    После подкоманды значения добавляются с default=SUPPRESS, иначе subparser
    затёр бы уже распознанные главным парсером аргументы (классическая ловушка
    argparse с parents=).
    """
    default: Any = argparse.SUPPRESS if suppress else None
    parser.add_argument("--url", default=default, help="адрес запущенного моста (для chat/status)")
    parser.add_argument(
        "--token",
        default=default if suppress else os.environ.get("BRIDGE_TOKEN"),
        help="BRIDGE_TOKEN для Authorization",
    )
    parser.add_argument("--host", type=str, default=default, help="адрес для serve (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=default, help="порт для serve (по умолчанию 8787)")
    parser.add_argument("--provider", default=default, help="deepseek_web | fake")
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=default,
        help="не показывать окно Chromium (нельзя при первом логине)",
    )
    parser.add_argument(
        "--no-headless", dest="headless", action="store_false", default=default, help="показывать окно браузера"
    )
    parser.add_argument("--log-level", dest="log_level", default=default, help="debug | info | warning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bridge.cli",
        description="Локальный мост к DeepSeek без API-ключа: браузер как бэкенд, наружу — OpenAI-совместимый REST",
    )
    _add_common(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)
    parents = argparse.ArgumentParser(add_help=False)
    _add_common(parents, suppress=True)

    def add(name: str, help_text: str, func: Any, **extra: Any) -> argparse.ArgumentParser:
        parser = sub.add_parser(name, help=help_text, parents=[parents], description=help_text, **extra)
        parser.set_defaults(func=func)  # без этого args.func отсутствует и serve падает в AttributeError
        return parser

    add("serve", "поднять HTTP-мост", cmd_serve)
    add("login", "один раз залогиниться в чат (сохраняет профиль)", cmd_login)
    add("status", "состояние моста/браузера", cmd_status).add_argument(
        "--local", action="store_true", help="не ходить на сервер, поднять свой браузер"
    )
    add("probe", "что видят селекторы на странице (для починки после редизайна)", cmd_probe)
    chat = add("chat", "отправить сообщение через мост", cmd_chat)
    chat.add_argument("message", nargs="*", help="без аргумента — интерактивный режим")
    chat.add_argument("--model", default="deepseek-chat")
    chat.add_argument("--session", help="продолжить диалог")
    chat.add_argument("--stream", action="store_true", help="печатать ответ по мере генерации")
    add("selftest", "проверить HTTP-контракт на фейковом драйвере (без браузера)", cmd_selftest)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ошибки моста показываем текстом, а не трейсбеком: у них уже есть message + hint."""
    from .errors import BridgeError

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nостановлено", file=sys.stderr)
        return 130
    except BridgeError as exc:
        print(f"ошибка {exc.status_code} ({exc.code}): {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"  что делать: {exc.hint}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
