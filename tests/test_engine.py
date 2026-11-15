"""Логика движка: что именно уходит в поле ввода и как работает очередь."""

from __future__ import annotations

import asyncio

import pytest

from bridge.drivers.base import ScriptedDriver
from bridge.engine import Engine, StreamHandle, build_prompt, diverges, estimate_tokens, plan
from bridge.errors import Busy, NeedLogin
from bridge.sessions import Session, SessionStore, Turn


def messages(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in pairs]


# --------------------------------------------------------------------- сборка промпта


def test_single_message_goes_as_is():
    prompt = build_prompt(messages(("user", "сколько будет 2+2?")), with_history=False)
    assert prompt == "сколько будет 2+2?"


def test_system_only_no_history_wrapper():
    prompt = build_prompt(messages(("system", "Отвечай кратко"), ("user", "привет")), with_history=True)
    assert "Инструкции" in prompt and "Отвечай кратко" in prompt
    assert "история нашего диалога" not in prompt.lower()
    assert prompt.rstrip().endswith("привет")


def test_history_is_flattened_into_one_prompt():
    prompt = build_prompt(
        messages(("user", "первое"), ("assistant", "ответ1"), ("user", "второе")),
        with_history=True,
    )
    assert "Пользователь: первое" in prompt
    assert "Ассистент: ответ1" in prompt
    assert "Последнее сообщение пользователя:\nвторое" in prompt


def test_plan_three_cases():
    # 1. новый диалог: промпт дословный, но страницу надо обновить
    prompt, fresh = plan(Session(id="s"), messages(("user", "а")))
    assert (prompt, fresh) == ("а", True)

    # 2. страница живая, клиент прислал только новое → шлём дословно, диалог не трогаем
    alive = Session(id="s", chat_live=True, turns=[Turn("user", "а"), Turn("assistant", "б")])
    assert plan(alive, messages(("user", "в"))) == ("в", False)
    #    то же самое, если клиент переслал всю историю (OpenAI-клиенты)
    assert plan(alive, messages(("user", "а"), ("assistant", "б"), ("user", "в"))) == ("в", False)

    # 3. страница мертва → историю доносим сами, даже если клиент о ней молчит
    dead = Session(id="s", chat_live=False, turns=[Turn("user", "а"), Turn("assistant", "б")])
    prompt, fresh = plan(dead, messages(("user", "в")))
    assert fresh is True and "Пользователь: а" in prompt and prompt.rstrip().endswith("в")

    # разошедшаяся история клиента → пересборка с нуля
    prompt, fresh = plan(alive, messages(("user", "СОВСЕМ ДРУГОЕ"), ("assistant", "б"), ("user", "в")))
    assert fresh is True and "СОВСЕМ ДРУГОЕ" in prompt


def test_diverges_only_on_overlap():
    assert diverges([], []) is False
    assert diverges(messages(("user", "а")), messages(("user", "а"))) is False
    assert diverges(messages(("user", "а")), messages(("user", "б"))) is True


def test_estimate_tokens_is_sane():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


# --------------------------------------------------------------------- движок


def run(coro):
    return asyncio.run(coro)


def make_engine(tmp_path, *, reply="ответ", logged_in=True):
    from bridge.config import Settings

    settings = Settings(provider="fake", data_dir=tmp_path / "data", poll_ms=1, stable_polls=1, generation_timeout_s=5)
    driver = ScriptedDriver(reply=reply, chunk_size=5)
    driver.logged_in = logged_in
    engine = Engine(settings, driver, SessionStore(settings.sessions_dir))
    run(engine.start())  # поднимает «браузер» (у скриптового драйвера он фиктивный)
    return engine, driver


def test_stream_yields_deltas_not_snapshots(tmp_path):
    engine, driver = make_engine(tmp_path, reply="0123456789")
    handle = StreamHandle()

    async def go():
        chunks = [c async for c in engine.stream(messages(("user", "пинг")), handle=handle)]
        return chunks

    chunks = run(go())
    assert "".join(chunks) == "0123456789"
    assert chunks[0] == "01234"  # дельта, а не накопленный снапшот
    assert handle.completion is not None
    assert handle.completion.finish_reason == "stop"
    assert driver.prompts == ["пинг"]


def test_second_turn_sends_only_new_message(tmp_path):
    engine, driver = make_engine(tmp_path, reply="раз-ответ")
    session_id = run(engine.complete(messages(("user", "первое")))).session_id

    run(engine.complete(messages(("user", "первое"), ("assistant", "раз-ответ"), ("user", "второе")), session_id=session_id))
    assert driver.prompts == ["первое", "второе"]  # историю не пересылали
    assert driver.begin_chat_calls == 1


def test_history_replayed_when_page_lost_context(tmp_path):
    engine, driver = make_engine(tmp_path, reply="раз-ответ")
    completion = run(engine.complete(messages(("user", "первое"))))
    session = engine.store.get(completion.session_id)
    engine.store.mark_live(session, live=False)  # как после рестарта моста

    run(engine.complete(messages(("user", "первое"), ("assistant", "раз-ответ"), ("user", "второе")), session_id=completion.session_id))
    assert "Вот история нашего диалога" in driver.prompts[-1]
    assert driver.begin_chat_calls == 2


def test_need_login_raised_before_touching_page(tmp_path):
    engine, driver = make_engine(tmp_path, logged_in=False)
    with pytest.raises(NeedLogin):
        run(engine.complete(messages(("user", "привет"))))
    assert driver.prompts == []


def test_session_marked_dead_after_error(tmp_path):
    class ExplodingDriver(ScriptedDriver):
        async def send(self, prompt, *, reasoning=False):
            raise RuntimeError("страница упала")
            yield  # pragma: no cover

    from bridge.config import Settings

    settings = Settings(provider="fake", data_dir=tmp_path / "data", poll_ms=1, stable_polls=1)
    engine = Engine(settings, ExplodingDriver(), SessionStore(settings.sessions_dir))
    run(engine.start())
    with pytest.raises(Exception):
        run(engine.complete(messages(("user", "привет"))))
    sessions = engine.store.list()
    assert sessions and sessions[0]["chat_live"] is False
    # и никакого «ответа-заглушки» в историю записано не было
    assert sessions[0]["turn_count"] == 0


def test_busy_when_lock_held(tmp_path):
    from bridge.config import Settings

    settings = Settings(provider="fake", data_dir=tmp_path / "data", queue_wait_s=0.05, max_waiters=4)
    engine = Engine(settings, ScriptedDriver(), SessionStore(settings.sessions_dir))

    async def go():
        await engine._acquire()  # держим «браузер занятым»
        try:
            with pytest.raises(Busy):
                await engine._acquire()
        finally:
            engine._release()

    run(go())


def test_waiters_are_capped(tmp_path):
    from bridge.config import Settings

    settings = Settings(provider="fake", data_dir=tmp_path / "data", queue_wait_s=10, max_waiters=0)
    engine = Engine(settings, ScriptedDriver(), SessionStore(settings.sessions_dir))

    async def go():
        await engine._acquire()
        try:
            with pytest.raises(Busy):
                await engine._acquire()
        finally:
            engine._release()

    run(go())


def test_successful_turn_after_failure_rebuilds_context(tmp_path):
    """Упавшую генерацию не пишем в историю, но мост обязан остаться рабочим."""
    from bridge.config import Settings

    class FlakyDriver(ScriptedDriver):
        calls = 0

        async def send(self, prompt, *, reasoning=False):
            FlakyDriver.calls += 1
            if FlakyDriver.calls == 1:
                raise RuntimeError("страница моргнула")
                yield  # pragma: no cover
            async for snapshot in super().send(prompt, reasoning=reasoning):
                yield snapshot

    settings = Settings(provider="fake", data_dir=tmp_path / "data", poll_ms=1, stable_polls=1)
    driver = FlakyDriver(reply="готово")
    engine = Engine(settings, driver, SessionStore(settings.sessions_dir))
    run(engine.start())

    with pytest.raises(RuntimeError):
        run(engine.complete(messages(("user", "первое"))))
    assert engine.store.list()[0]["turn_count"] == 0

    completion = run(engine.complete(messages(("user", "первое"))))
    assert completion.text == "готово"
    assert completion.rebuilt_context is True  # история пуста → новый диалог


def test_truncated_generation_reports_length_finish_reason(tmp_path):
    """Если ответ оборвался по таймауту, клиент обязан получить finish_reason="length".

    Иначе молчаливый «stop» выглядит как полноценный ответ, и обрыв не заметить.
    """
    from bridge.config import Settings

    class TruncatingDriver(ScriptedDriver):
        truncated = False

        async def send(self, prompt, *, reasoning=False):
            self.truncated = True
            yield "всё, что успели прочитать"

    settings = Settings(provider="fake", data_dir=tmp_path / "data", poll_ms=1, stable_polls=1)
    engine = Engine(settings, TruncatingDriver(), SessionStore(settings.sessions_dir))
    run(engine.start())

    completion = run(engine.complete(messages(("user", "длинный рассказ"))))
    assert completion.finish_reason == "length"
    assert completion.text == "всё, что успели прочитать"
    assert completion.rebuilt_context is True
    assert engine._stats["requests"] == 1 and engine._stats["completed"] == 1


def test_engine_recovers_page_after_failed_generation(tmp_path):
    """После сбоя движок обязан попросить драйвер «успокоиться» — иначе следующий запрос не уйдёт.

    Это и есть «надо вручную открыть новый чат»: страница оставалась в режиме
    генерации с заблокированным полем ввода.
    """
    class ExplodingDriver(ScriptedDriver):
        async def send(self, prompt, *, reasoning=False):
            raise RuntimeError("страница упала")
            yield  # pragma: no cover

    from bridge.config import Settings

    settings = Settings(provider="fake", data_dir=tmp_path / "data", poll_ms=1, stable_polls=1)
    driver = ExplodingDriver()
    engine = Engine(settings, driver, SessionStore(settings.sessions_dir))
    run(engine.start())

    with pytest.raises(RuntimeError):
        run(engine.complete(messages(("user", "привет"))))
    assert driver.recover_calls == 1
    assert driver.idle_calls >= 1  # ensure_idle вызывается и recovery-путём
    assert engine._stats["failed"] == 1


def test_engine_does_not_recover_on_success(tmp_path):
    from bridge.config import Settings

    settings = Settings(provider="fake", data_dir=tmp_path / "data", poll_ms=1, stable_polls=1)
    driver = ScriptedDriver(reply="ok")
    engine = Engine(settings, driver, SessionStore(settings.sessions_dir))
    run(engine.start())
    run(engine.complete(messages(("user", "привет"))))
    assert driver.recover_calls == 0
    assert engine._stats["completed"] == 1 and engine._stats["failed"] == 0
