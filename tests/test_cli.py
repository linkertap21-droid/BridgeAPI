"""CLI: парсер не должен терять подкоманды и общие флаги.

Здесь ровно то, что ломается молча на первом же `serve`: func и позиции флагов.
"""

from __future__ import annotations

import pytest

from bridge.cli import build_parser


def test_every_subcommand_resolves_to_handler():
    parser = build_parser()
    for command in ["serve", "login", "status", "probe", "selftest"]:
        args = parser.parse_args([command])
        assert callable(getattr(args, "func", None)), f"у команды {command} не назначен func"


def test_common_flags_work_before_and_after_subcommand():
    parser = build_parser()
    before = parser.parse_args(["--port", "9000", "--provider", "fake", "serve"])
    after = parser.parse_args(["serve", "--port", "9000", "--provider", "fake"])
    for args in (before, after):
        assert args.port == 9000 and args.provider == "fake"

    chat = parser.parse_args(["chat", "привет", "мир", "--stream"])
    assert chat.message == ["привет", "мир"] and chat.stream is True


def test_headless_toggle(tmp_path, monkeypatch):
    monkeypatch.setenv("BRIDGE_DATA_DIR", str(tmp_path))
    parser = build_parser()
    assert parser.parse_args(["serve", "--headless"]).headless is True
    assert parser.parse_args(["serve", "--no-headless"]).headless is False
    assert parser.parse_args(["serve"]).headless is None  # пусть решает .env/дефолт


def test_cli_prints_bridge_error_instead_of_traceback(tmp_path, monkeypatch, capsys):
    """Без установленного Chromium `status --local` должен дать подсказку, а не стектрейс."""
    from bridge import cli

    monkeypatch.setenv("BRIDGE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_engine_call", _boom)
    code = cli.main(["status", "--local", "--provider", "deepseek_web"])
    err = capsys.readouterr().err
    assert code == 1
    assert "driver_crashed" in err and "что делать" in err


async def _boom(settings, what):
    from bridge.errors import DriverCrashed

    raise DriverCrashed("browser type has not been downloaded")
