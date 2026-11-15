"""Машина состояний драйвера против фейковой страницы чата.

Эти тесты — регресс на три реальных симптома: «отправка через раз», «ответ не
вернулся», «надо вручную открыть новый чат». Каждый тест описывает ситуацию,
в которой старая реализация вела себя иначе, чем должна.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from bridge.config import Settings
from bridge.drivers.deepseek_web import DeepSeekWebChatDriver, clean, is_settled, pick_answer, settled_answer
from bridge.errors import DriverCrashed, GenerationTimeout

from tests.fake_page import FakeChatPage


def make(page: FakeChatPage, **overrides) -> DeepSeekWebChatDriver:
    settings = Settings(
        poll_ms=1,
        stable_polls=overrides.pop("stable_polls", 1),
        generation_timeout_s=overrides.pop("generation_timeout_s", 5),
        start_timeout_s=overrides.pop("start_timeout_s", 3),
        send_wait_s=0.4,
        idle_wait_s=0.6,
        ready_timeout_s=0.4,
        login_timeout_s=1,
        debug_dump=False,
        **overrides,
    )
    driver = DeepSeekWebChatDriver(settings)
    driver._page = page
    return driver


def collect(driver, prompt="привет", **kwargs):
    async def go():
        return [chunk async for chunk in driver.send(prompt, **kwargs)]

    return asyncio.run(go())


# --------------------------------------------------------------------- отправка


def test_send_returns_final_answer_text():
    page = FakeChatPage()
    driver = make(page)
    snapshots = collect(driver)
    assert snapshots[-1] == "Это новый ответ модели"
    assert driver.truncated is False
    assert page.sends == 1  # ровно одна отправка, без дублей
    # каждое следующее значение — рост предыдущего (движок считает diff по длине)
    assert all(b.startswith(a) for a, b in zip(snapshots, snapshots[1:]))


def test_send_waits_out_a_stuck_generation_instead_of_failing():
    """«Через раз»: страница ещё генерирует прошлый ответ → надо снять это, а не молчать."""
    page = FakeChatPage(generating=True)
    driver = make(page)
    snapshots = collect(driver)
    assert page.stops_clicked >= 1, "драйвер не нажал Stop на залипшей генерации"
    assert snapshots[-1] == "Это новый ответ модели"
    assert page.sends == 1


def test_send_is_verified_not_assumed():
    """Если поле не очистилось — сообщение НЕ ушло: падает сразу и с подсказкой."""
    page = FakeChatPage(accepts_send=False)
    driver = make(page)
    with pytest.raises(DriverCrashed) as exc:
        collect(driver)
    assert "не отправилось" in exc.value.message
    assert exc.value.hint and "reset" in exc.value.hint
    assert page.input_text == ""  # поле вычищено, следующий запрос не упрётся в него


def test_prompt_is_written_before_submit_and_cleaned_after_failure():
    page = FakeChatPage(accepts_input=False)
    driver = make(page)
    with pytest.raises(DriverCrashed, match="не появился в поле ввода"):
        collect(driver)
    assert page.sends == 0  # не отправляли то, что не удалось ввести


# --------------------------------------------------------------------- чтение ответа


def test_previous_answer_is_not_returned_as_new():
    """Главный баг «не возвращает ответ»: счётчик блоков не рос, читался старый текст."""
    page = FakeChatPage(blocks=["предыдущий вопрос", "СТАРЫЙ ответ модели"])
    driver = make(page)
    snapshots = collect(driver)
    assert snapshots
    assert "СТАРЫЙ" not in snapshots[-1]
    assert snapshots[-1] == "Это новый ответ модели"


def test_user_echo_is_never_treated_as_answer():
    """Тот же селектор матчит бабл пользователя: без фильтра клиенту уехал бы его же текст."""
    page = FakeChatPage(answer="", thinking_polls=50)  # блок ответа так и не наполнился
    page.pending_answer = ""
    driver = make(page, generation_timeout_s=1, start_timeout_s=0.2)
    with pytest.raises(GenerationTimeout):
        collect(driver, prompt="привет")


def test_no_answer_block_raises_generation_timeout_with_actionable_hint():
    """Сообщение приняли, а нового блока нет → 504, а не «ответ = промпт» и не вечное ожидание."""
    page = FakeChatPage(answer="", thinking_polls=50)
    page.pending_answer = ""
    driver = make(page, generation_timeout_s=1, start_timeout_s=0.2)
    with pytest.raises(GenerationTimeout) as exc:
        collect(driver)
    assert exc.value.status_code == 504
    assert "assistant_message" in exc.value.hint
    assert "probe" in exc.value.hint


def test_answer_is_read_while_thinking_pauses():
    """Пауза «думает» не должна считаться концом ответа (иначе получим обрывок)."""
    page = FakeChatPage(thinking_polls=3, answer="полный ответ после раздумий")
    driver = make(page, stable_polls=1)
    snapshots = collect(driver)
    assert snapshots[-1] == "полный ответ после раздумий"


# --------------------------------------------------------------------- новый диалог


def test_begin_chat_falls_back_to_reload_when_click_is_swallowed():
    """Раньше клик по «новому чату» принимался на веру, и запрос уезжал в старый диалог."""
    page = FakeChatPage(blocks=["старый блок"], new_chat_works=False)
    driver = make(page)

    async def go():
        await driver.begin_chat()

    asyncio.run(go())
    assert page.gotos >= 1, "не было перезагрузки страницы, когда клик по «новому чату» не сработал"
    assert page.blocks == []


def test_begin_chat_uses_button_when_it_works():
    page = FakeChatPage(blocks=["старый блок"])
    driver = make(page)
    asyncio.run(driver.begin_chat())
    assert page.new_chats == 1 and page.gotos == 0


# --------------------------------------------------------------------- санация/статус


def test_recover_never_raises_and_uses_stop():
    page = FakeChatPage(generating=True)
    page.input_text = "остаток"
    driver = make(page)
    asyncio.run(driver.recover())
    assert page.stops_clicked >= 1
    assert page.input_text == ""


def test_status_reports_login_loss():
    page = FakeChatPage()
    page.url = "https://chat.deepseek.com/sign-in"
    driver = make(page)
    status = asyncio.run(driver.status())
    assert status.logged_in is False
    assert "sign-in" in status.url


def test_diagnostics_dump_is_written_on_failure(tmp_path):
    """Без дампа правка селекторов — угадывание; проверяем, что он реально появляется."""
    page = FakeChatPage(answer="", thinking_polls=50)
    page.pending_answer = ""
    driver = make(page, generation_timeout_s=1, start_timeout_s=0.2)
    driver.settings = driver.settings.override(debug_dump=True, data_dir=tmp_path / "d")
    with pytest.raises(GenerationTimeout):
        collect(driver)
    dumps = list((tmp_path / "d" / "debug").glob("*no-answer*"))
    assert dumps, "дамп диагностики не записан"
    assert any(path.suffix == ".json" for path in dumps)
    payload = json.loads(next(p for p in dumps if p.suffix == ".json").read_text(encoding="utf-8"))
    assert payload["reason"] == "no-answer" and "selectors" in payload


def test_reasoning_mode_is_stateful_not_blind_toggle():
    """Режим «думать» переключается один раз и возвращается при смене модели.

    Слепой toggle давал «через раз» на deepseek-reasoner: каждый второй запрос
    выключал то, что включил первый.
    """
    page = FakeChatPage()
    driver = make(page)
    collect(driver, reasoning=True)
    assert page.think_clicks == 1
    collect(driver, reasoning=True)
    assert page.think_clicks == 1  # состояние уже верное — не трогаем
    collect(driver, reasoning=False)
    assert page.think_clicks == 2  # выключили обратно


def test_default_selectors_have_generic_fallbacks():
    """После редизайна должен остаться хотя бы один общий вариант на ключ."""
    from bridge.drivers.base import DEFAULT_SELECTORS

    for key in ("input", "send_button", "stop_button", "assistant_message"):
        candidates = [c.strip() for c in DEFAULT_SELECTORS[key].split("||")]
        assert len(candidates) >= 2, f"у {key} должен быть запасной селектор"


# --------------------------------------------------------------------- чистые функции


def test_pick_answer_ignores_stale_blocks_and_echo():
    prompt = "сколько будет 2+2"
    previous = ["вопрос ранее", "ответ ранее"]
    # только эхо пользователя + пустой бабл ассистента — ответа ещё нет
    assert pick_answer([*previous, prompt, ""], previous, prompt) is None
    # блоков стало больше → берём последний непустой из новых
    assert pick_answer([*previous, prompt, "Четы"], previous, prompt) == "Четы"
    # ничего не изменилось → None (это и есть защита от «верни предыдущий ответ»)
    assert pick_answer(previous, previous, prompt) is None
    # контейнер переиспользован: счётчик не растёт, но текст меняется
    assert pick_answer(["ответ ранее 2"], ["ответ ранее"], prompt) == "ответ ранее 2"
    # повторный ответ, совпадающий с прошлым, — по позиции новый, значит принимаем
    assert pick_answer(["ответ ранее", prompt, "ответ ранее"], ["ответ ранее"], prompt) == "ответ ранее"


def test_pick_answer_handles_reused_container():
    """Сайт переиспользует тот же блок → счётчик не растёт, а текст меняется."""
    assert pick_answer(["растущ"], ["расту"], "промпт") == "растущ"


def test_is_settled_requires_quiet_page():
    assert is_settled("текст", 5, generating=True, ever_generating=True, needed=1) is False
    assert is_settled("текст", 1, generating=False, ever_generating=True, needed=1) is True
    # stop-селектор не сработал ни разу → доверять «тишине» нельзя, окно растёт в 4 раза
    assert is_settled("текст", 3, generating=False, ever_generating=False, needed=1) is False
    assert is_settled("текст", 4, generating=False, ever_generating=False, needed=1) is True
    assert is_settled("", 100, generating=False, ever_generating=True, needed=1) is False


def test_settled_answer_prefers_last_non_echo_block():
    assert settled_answer(["вопрос", "ответ"], "вопрос") == "ответ"
    assert settled_answer(["вопрос", "  "], "вопрос") is None


def test_identical_repeat_answer_is_returned_fast():
    """Второй запрос с тем же ответом не должен ждать полный таймаут."""
    page = FakeChatPage(answer="42", chunk=2)
    driver = make(page, generation_timeout_s=4)
    first = collect(driver, prompt="сколько будет 6*7?")
    assert first[-1] == "42"
    started = time.monotonic()
    second = collect(driver, prompt="а ещё раз?")
    assert second and "42" in second[-1]
    assert time.monotonic() - started < 3, "повтор ответа завис до таймаута"


def test_clean_strips_zero_width_junk():
    assert clean("  текст\u200b\n") == "текст"
    assert clean(None) == ""
