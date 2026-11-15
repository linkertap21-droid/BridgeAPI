"""Драйверы и селекторы: то, что ломается вместе с вёрсткой DeepSeek."""

from __future__ import annotations

import asyncio
import json

import pytest

from bridge.config import Settings
from bridge.drivers import build_driver
from bridge.drivers.base import DEFAULT_SELECTORS, ScriptedDriver, Selectors
from bridge.drivers.deepseek_web import DeepSeekWebChatDriver


def test_selectors_split_alternatives():
    selectors = Selectors({"input": "  textarea#chat-input ||  #alt  "})
    assert selectors.candidates("input") == ["textarea#chat-input", "#alt"]
    assert selectors.candidates("unknown_key") == []


def test_selectors_file_overrides_defaults(tmp_path):
    path = tmp_path / "selectors.json"
    path.write_text(json.dumps({"input": "#my-own-input"}), encoding="utf-8")
    selectors = Selectors.load(path)
    assert selectors.get("input") == "#my-own-input"
    # незаменённые ключи остаются дефолтными
    assert selectors.get("assistant_message") == DEFAULT_SELECTORS["assistant_message"]


def test_selectors_load_missing_file_is_silent(tmp_path):
    assert Selectors.load(tmp_path / "нет.json").get("input") == DEFAULT_SELECTORS["input"]


def test_build_driver_registry(tmp_path):
    assert isinstance(build_driver(Settings(data_dir=tmp_path / "d", provider="fake")), ScriptedDriver)
    assert isinstance(
        build_driver(Settings(data_dir=tmp_path / "d", provider="deepseek_web")), DeepSeekWebChatDriver
    )
    with pytest.raises(ValueError, match="неизвестный BRIDGE_PROVIDER"):
        build_driver(Settings(data_dir=tmp_path / "d", provider="nope"))


def test_scripted_driver_streams_cumulative_snapshots():
    driver = ScriptedDriver(reply="0123456789", chunk_size=3)

    async def go():
        return [chunk async for chunk in driver.send("привет")]

    snapshots = asyncio.run(go())
    assert snapshots == ["012", "012345", "012345678", "0123456789"]
    assert driver.prompts == ["привет"]


def test_scripted_driver_can_echo_prompt():
    driver = ScriptedDriver(reply="эхо: {prompt}")

    async def go():
        return [chunk async for chunk in driver.send("данные")]

    assert "".join(asyncio.run(go())).startswith("эхо: данные")


def test_web_driver_reports_not_ready_without_browser(tmp_path):
    """Без запущенного Chromium драйвер обязан дать понятную ошибку, а не AttributeError."""
    driver = DeepSeekWebChatDriver(Settings(data_dir=tmp_path / "d", headless=True))
    from bridge.errors import DriverNotReady

    async def go():
        return [chunk async for chunk in driver.send("привет")]

    with pytest.raises(DriverNotReady, match="не запущен"):
        asyncio.run(go())
    # probe не бросает — он и нужен, чтобы диагностировать «браузера нет»
    assert asyncio.run(driver.probe())["running"] is False
