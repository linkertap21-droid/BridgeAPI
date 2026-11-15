"""HTTP-контракт моста: то, что проверяет любой клиент — коды, форма ответа, SSE."""

from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from bridge.config import Settings
from bridge.drivers.base import ScriptedDriver
from bridge.engine import Engine
from bridge.http import build_app
from bridge.sessions import SessionStore


def test_health_and_root(client):
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["driver"] == "fake"
    assert health["queue"]["busy"] is False
    assert health["config"]["auth_required"] is False

    root = client.get("/").json()
    assert "POST /v1/chat" in root["generate"]


def test_models_lists_both_modes(client):
    data = client.get("/v1/models").json()["data"]
    ids = {item["id"] for item in data}
    assert ids == {"deepseek-chat", "deepseek-reasoner"}


def test_openai_completion_shape(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "_ping_"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Привет! Я эхо-мост: _ping_"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["estimated"] is True
    assert body["usage"]["total_tokens"] > 0
    assert body["session_id"]


def test_simple_endpoint_and_reasoning_flag(client, driver):
    body = client.post("/v1/chat", json={"prompt": "реши", "model": "deepseek-reasoner"}).json()
    assert body["reply"].startswith("Привет!")
    assert body["model"] == "deepseek-reasoner"
    assert driver.reasoning_flags == [True]


def test_streaming_sse_reassembles_and_ends_with_done(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "abcd"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = list(response.iter_lines())

    events = [line[6:] for line in lines if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(item) for item in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Привет! Я эхо-мост: abcd"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] > 0


def test_login_required_maps_to_503(tmp_settings):
    driver = ScriptedDriver()
    driver.logged_in = False
    app = build_app(tmp_settings, engine=Engine(tmp_settings, driver, SessionStore(tmp_settings.sessions_dir)))
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"prompt": "привет"})
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "login_required"
    assert "login" in error["hint"]


def test_stream_reports_error_inside_sse(tmp_settings):
    driver = ScriptedDriver()
    driver.logged_in = False
    app = build_app(tmp_settings, engine=Engine(tmp_settings, driver, SessionStore(tmp_settings.sessions_dir)))
    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat", json={"prompt": "привет", "stream": True}) as response:
            raw = "".join(response.iter_lines())
    assert '"type": "login_required"' in raw
    assert "[DONE]" in raw


def test_validation_errors(tmp_settings):
    app = build_app(tmp_settings, engine=Engine(tmp_settings, ScriptedDriver(), SessionStore(tmp_settings.sessions_dir)))
    with TestClient(app) as client:
        bad = client.post("/v1/chat/completions", json={"messages": []})
        assert bad.status_code == 400
        assert bad.json()["error"]["type"] == "invalid_request"
        assert "messages" in bad.json()["error"]["message"]
        assert client.post("/v1/chat", json={}).status_code == 400
        assert client.post("/v1/chat", content="не json", headers={"content-type": "application/json"}).status_code == 400


def test_token_auth_required_when_configured(tmp_path):
    settings = Settings(provider="fake", data_dir=tmp_path / "d", token="s3cret", poll_ms=1, stable_polls=1)
    driver = ScriptedDriver(reply="ок")
    app = build_app(settings, engine=Engine(settings, driver, SessionStore(settings.sessions_dir)))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200  # диагностика открыта
        assert client.post("/v1/chat", json={"prompt": "хай"}).status_code == 401
        assert client.post("/v1/chat", json={"prompt": "хай"}, headers={"Authorization": "Bearer nope"}).status_code == 401
        ok = client.post("/v1/chat", json={"prompt": "хай"}, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert ok.json()["reply"] == "ок"


def test_probe_and_auth_endpoints(client):
    assert client.get("/v1/debug/probe").json()["driver"] == "fake"
    assert client.get("/v1/auth/status").json()["running"] is True
    assert client.post("/v1/auth/login").json()["logged_in"] is True
    assert client.post("/v1/auth/logout").json()["ok"] is True


def test_busy_returns_429_with_hint(tmp_settings):
    from bridge.drivers.base import ScriptedDriver

    settings = tmp_settings.override(queue_wait_s=0.05, max_waiters=4)
    engine = Engine(settings, ScriptedDriver(), SessionStore(settings.sessions_dir))

    async def hold():
        await engine._acquire()

    asyncio.run(hold())  # «браузер» занят навсегда
    with TestClient(build_app(settings, engine=engine)) as client:
        response = client.post("/v1/chat", json={"prompt": "пока ты занят"})
    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "bridge_busy" and "429" not in error["message"]
    engine._release()


def test_sessions_crud(client):
    first = client.post("/v1/chat", json={"prompt": "первый"}).json()
    sid = first["session_id"]
    client.post("/v1/chat", json={"prompt": "второй", "session_id": sid})

    listing = client.get("/v1/sessions").json()["data"]
    assert any(item["id"] == sid and item["turn_count"] == 4 for item in listing)

    detail = client.get(f"/v1/sessions/{sid}").json()
    assert [t["role"] for t in detail["turns"]] == ["user", "assistant", "user", "assistant"]
    assert detail["chat_live"] is True

    assert client.delete(f"/v1/sessions/{sid}").json()["deleted"] is True
    assert client.get(f"/v1/sessions/{sid}").status_code == 404


def test_stateful_session_does_not_replay_history(client):
    """Регресс: «только новое сообщение + session_id» не должно выглядеть как потеря контекста.

    Это самый частый путь (curl, CLI, простой клиент). Мост уже отправил историю
    на страницу, значит второй запрос уходит дословно, а новый диалог не начинается.
    """
    first = client.post("/v1/chat", json={"prompt": "запомни число 42"}).json()
    assert first["rebuilt_context"] is True  # новый диалог — ожидаемо

    second = client.post(
        "/v1/chat", json={"prompt": "какое число?", "session_id": first["session_id"]}
    ).json()
    assert second["rebuilt_context"] is False
    detail = client.get(f"/v1/sessions/{first['session_id']}").json()
    assert detail["chat_live"] is True
    assert detail["turn_count"] == 4
