"""HTTP-слой моста (FastAPI).

Здесь нет бизнес-логики — только контракт: OpenAI-совместимые эндпоинты,
стриминг через SSE, диагностика браузера и CRUD локальных диалогов.
Всё «умное» живёт в `bridge/engine.py` и в драйвере.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .config import REPO_ROOT, Settings
from .drivers.base import ConversationDriver
from .engine import OPENAI_MODELS, Completion, Engine, StreamHandle
from .errors import BridgeError, DriverNotReady, InvalidRequest, NeedLogin
from .models import (
    ChatCompletionRequest,
    ChatRequest,
    Choice,
    ChoiceMessage,
    chunk,
    created_ts,
    new_id,
)
from .sessions import SessionStore


def error_body(exc: BridgeError) -> dict[str, Any]:
    """Единая форма ошибки. `error.type` — машиночитаемый код (`login_required`, ...),
    `hint` — что делать; дублировать одно и тем же полем незачем."""
    return exc.to_payload()


def require_token(request: Request) -> None:
    """Токен опционален: по умолчанию мост слушает только loopback.

    Как только меняешь BRIDGE_HOST на 0.0.0.0 — выставь и BRIDGE_TOKEN,
    иначе любой в сети сможет пользоваться твоим аккаунтом DeepSeek.
    """
    settings: Settings = request.app.state.settings
    if not settings.token:
        return
    header = request.headers.get("authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    supplied = supplied or request.query_params.get("token", "")
    if supplied != settings.token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "нужен заголовок Authorization: Bearer <BRIDGE_TOKEN>", "type": "unauthorized"}},
        )


def build_app(
    settings: Settings | None = None,
    *,
    driver: ConversationDriver | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    """Фабрика приложения. `driver`/`engine` нужны тестам, чтобы не поднимать браузер."""
    settings = settings or Settings.from_env(REPO_ROOT / ".env")
    store = SessionStore(settings.sessions_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal engine
        if engine is None:
            from .drivers import build_driver

            engine = Engine(settings, driver or build_driver(settings), store)
        app.state.engine = engine
        app.state.startup_error = None
        try:
            await engine.start()
        except BridgeError as exc:
            # Браузер не поднялся (нет профиля / не скачан chromium) — API всё равно
            # полезен: /health и /v1/auth/* объяснят, что делать, вместо 500 на ровном месте.
            app.state.startup_error = exc.message
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(
        title="BridgeAPI — локальный мост к DeepSeek",
        version="0.2.0",
        description="REST/OpenAI-совместимый API поверх твоего браузера с chat.deepseek.com, без API-ключа.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=settings.cors_origin_regex,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(BridgeError)
    async def bridge_error_handler(_: Request, exc: BridgeError) -> JSONResponse:
        return JSONResponse(error_body(exc), status_code=exc.status_code)

    def get_engine(request: Request) -> Engine:
        current = request.app.state.engine
        if current is None:  # pragma: no cover - вызов вне lifespan
            raise DriverNotReady("Мост ещё не инициализирован.")
        return current

    # ------------------------------------------------------------------ служебное

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        data = await get_engine(request).status()
        error = getattr(request.app.state, "startup_error", None)
        if error:
            data["startup_error"] = error
        return data

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "BridgeAPI",
            "generate": ["POST /v1/chat/completions", "POST /v1/chat"],
            "diagnostics": ["GET /health", "POST /v1/auth/login", "GET /v1/auth/status", "GET /v1/debug/probe"],
            "docs": "/docs",
            "config": settings.public_dict(),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [dict(m) for m in OPENAI_MODELS]}

    # ------------------------------------------------------------------ генерация

    @app.post("/v1/chat/completions", dependencies=[Depends(require_token)])
    async def chat_completions(request: Request) -> Any:
        payload = _parse(ChatCompletionRequest, await _body(request))
        return await _run(request, payload.as_messages(), payload.session_id, payload.model, payload.stream, openai=True)

    @app.post("/v1/chat", dependencies=[Depends(require_token)])
    async def chat_simple(request: Request) -> Any:
        payload = _parse(ChatRequest, await _body(request))
        messages = [{"role": "user", "content": payload.prompt}]
        if payload.system:
            messages.insert(0, {"role": "system", "content": payload.system})
        return await _run(request, messages, payload.session_id, payload.model, payload.stream, openai=False)

    async def _run(
        request: Request,
        messages: list[dict[str, str]],
        session_id: str | None,
        model: str,
        stream: bool,
        *,
        openai: bool,
    ) -> Any:
        engine = get_engine(request)
        if stream:
            return StreamingResponse(
                _sse(engine, messages, session_id=session_id, model=model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
            )
        completion = await engine.complete(messages, session_id=session_id, model=model)
        if openai:
            return _openai_response(completion)
        return {
            "reply": completion.text,
            "session_id": completion.session_id,
            "model": completion.model,
            "finish_reason": completion.finish_reason,
            "rebuilt_context": completion.rebuilt_context,
            "elapsed_s": completion.elapsed_s,
            "usage": _usage(completion),
        }

    # ------------------------------------------------------------------ логин/диагностика

    @app.get("/v1/auth/status", dependencies=[Depends(require_token)])
    async def auth_status(request: Request) -> dict[str, Any]:
        return (await get_engine(request).status())["browser"]

    @app.post("/v1/auth/login", dependencies=[Depends(require_token)])
    async def auth_login(request: Request, timeout_s: int | None = Query(None, ge=5, le=3600)) -> dict[str, Any]:
        ok = await get_engine(request).driver.login(timeout_s)
        if not ok:
            exc = NeedLogin()
            exc.message = f"Логин не завершён за {timeout_s or 'отведённое'} времени."
            raise exc
        return {"logged_in": True, "detail": "cookies сохранены в профиль моста, можно пользоваться"}

    @app.post("/v1/auth/logout", dependencies=[Depends(require_token)])
    async def auth_logout(request: Request) -> dict[str, Any]:
        await get_engine(request).stop()
        return {"ok": True, "detail": "браузер закрыт; чтобы забыть аккаунт — удали data/browser-profile"}

    @app.api_route("/v1/debug/probe", methods=["GET", "POST"], dependencies=[Depends(require_token)])
    async def debug_probe(request: Request, save: bool = Query(False)) -> dict[str, Any]:
        """Что нашли селекторы; `?save=true` — сохранить дамп страницы в data/debug/."""
        return await get_engine(request).probe(save=save)

    @app.post("/v1/debug/recover", dependencies=[Depends(require_token)])
    async def debug_recover(request: Request) -> dict[str, Any]:
        """Снять залипшую генерацию и освободить поле ввода (чинит «надо вручную открыть новый чат»)."""
        return await get_engine(request).reset()

    # ------------------------------------------------------------------ диалоги

    @app.get("/v1/sessions", dependencies=[Depends(require_token)])
    async def sessions_list(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return {"object": "list", "data": get_engine(request).store.list(limit=limit)}

    @app.post("/v1/sessions/clear", dependencies=[Depends(require_token)])
    async def sessions_clear(request: Request) -> dict[str, Any]:
        return {"cleared": get_engine(request).store.clear()}

    @app.get("/v1/sessions/{session_id}", dependencies=[Depends(require_token)])
    async def session_get(request: Request, session_id: str) -> dict[str, Any]:
        session = get_engine(request).store.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"error": {"message": f"сессия {session_id} не найдена", "type": "not_found"}},
            )
        return session.to_dict()

    @app.delete("/v1/sessions/{session_id}", dependencies=[Depends(require_token)])
    async def session_delete(request: Request, session_id: str) -> dict[str, Any]:
        return {"deleted": get_engine(request).store.delete(session_id), "id": session_id}

    return app


def app_factory() -> FastAPI:
    """Точка входа uvicorn(`--factory`) / `uvicorn.run(..., factory=True)`."""
    return build_app()


def _parse(model: Any, body: dict[str, Any]) -> Any:
    """Ошибка схемы = 400 с внятным текстом, а не 500 или «непонятное 422»."""
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'тело'}: {item['msg']}"
            for item in exc.errors()
        )
        raise InvalidRequest(f"некорректный запрос — {problems}") from exc


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise InvalidRequest(f"тело запроса не JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise InvalidRequest("тело запроса должно быть JSON-объектом")
    return body


def _usage(completion: Completion) -> dict[str, Any]:
    return {**completion.usage(), "estimated": True}


def _openai_response(completion: Completion) -> dict[str, Any]:
    return {
        "id": new_id(),
        "object": "chat.completion",
        "created": created_ts(),
        "model": completion.model,
        "choices": [
            Choice(
                message=ChoiceMessage(content=completion.text),
                finish_reason=completion.finish_reason,
            ).model_dump()
        ],
        "usage": _usage(completion),
        "session_id": completion.session_id,
        "bridge": {
            "rebuilt_context": completion.rebuilt_context,
            "elapsed_s": completion.elapsed_s,
            "note": "usage оценён по длине текста: у веб-чата нет точного счётчика токенов; "
            "temperature/max_tokens не применяются — их нечем управлять на стороне страницы",
        },
    }


def sse_event(data: dict[str, Any] | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _sse(
    engine: Engine,
    messages: list[dict[str, str]],
    *,
    session_id: str | None,
    model: str,
) -> AsyncIterator[str]:
    """SSE: role-чанк → дельты → финальный чанк с finish_reason/usage → [DONE]."""
    completion_id = new_id()
    created = created_ts()
    handle = StreamHandle()
    yield sse_event(chunk(completion_id, model, created, delta={"role": "assistant", "content": ""}))
    try:
        async for delta in engine.stream(messages, session_id=session_id, model=model, handle=handle):
            yield sse_event(chunk(completion_id, model, created, delta={"content": delta}))
    except asyncio.CancelledError:
        raise  # клиент закрыл соединение — отдаём отмену, генератор отработает finally
    except BridgeError as exc:
        # Заголовок 200 уже отправлен, поэтому ошибку несём телом потока.
        yield sse_event(error_body(exc))
        yield sse_event("[DONE]")
        return
    except Exception as exc:  # noqa: BLE001 - поток не должен ронять сервер
        yield sse_event({"error": {"message": f"внутренняя ошибка моста: {exc}", "type": "bridge_error"}})
        yield sse_event("[DONE]")
        return

    final = handle.completion
    usage = {
        "prompt_tokens": final.prompt_tokens if final else 0,
        "completion_tokens": final.completion_tokens if final else 0,
        "total_tokens": (final.prompt_tokens + final.completion_tokens) if final else 0,
        "estimated": True,
    }
    yield sse_event(
        chunk(
            completion_id,
            model,
            created,
            delta={},
            finish_reason=final.finish_reason if final else "stop",
            usage=usage,
        )
    )
    yield sse_event("[DONE]")
