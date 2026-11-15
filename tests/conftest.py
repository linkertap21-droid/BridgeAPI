"""Общие фикстуры: приложение на скриптовом драйвере, без браузера."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bridge.config import Settings
from bridge.drivers.base import ScriptedDriver
from bridge.engine import Engine
from bridge.http import build_app
from bridge.sessions import SessionStore


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8787,
        provider="fake",
        data_dir=tmp_path / "data",
        headless=True,
        poll_ms=1,
        stable_polls=1,
        generation_timeout_s=5,
        queue_wait_s=5,
    )


@pytest.fixture()
def driver() -> ScriptedDriver:
    return ScriptedDriver(reply="Привет! Я эхо-мост: {prompt}", chunk_size=7)


@pytest.fixture()
def engine(tmp_settings: Settings, driver: ScriptedDriver) -> Engine:
    driver.logged_in = True
    return Engine(tmp_settings, driver, SessionStore(tmp_settings.sessions_dir))


@pytest.fixture()
def client(tmp_settings: Settings, engine: Engine):
    app = build_app(tmp_settings, engine=engine)
    with TestClient(app) as test_client:
        yield test_client
