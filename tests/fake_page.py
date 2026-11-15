"""Фейковая страница чата: позволяет прогнать состояние драйвера без браузера.

Смысл не в том, чтобы «заменить Playwright», а в том, чтобы зафиксировать
поведение, на котором ловились реальные баги:
  * отправка не подтверждена → сообщение не ушло, а мы ждём ответ 4 минуты;
  * счётчик блоков не растёт → читается ПРЕДЫДУЩИЙ ответ;
  * тот же селектор матчит бабл пользователя → эхо промпта уезжает как «ответ»;
  * оборванная генерация оставляет страницу занятой → следующий запрос не уходит.
"""

from __future__ import annotations

from typing import Any


class FakeChatPage:
    """Минимальная модель страницы чата: поле ввода, блоки сообщений, генерация."""

    def __init__(
        self,
        *,
        blocks: list[str] | None = None,
        generating: bool = False,
        accepts_input: bool = True,
        accepts_send: bool = True,
        new_chat_works: bool = True,
        answer: str = "Это новый ответ модели",
        chunk: int = 8,
        thinking_polls: int = 0,
    ) -> None:
        self.blocks: list[str] = list(blocks or [])
        self.generating = generating
        self.accepts_input = accepts_input
        self.accepts_send = accepts_send
        self.new_chat_works = new_chat_works
        self.pending_answer = answer
        self.chunk = chunk
        self.thinking_polls = thinking_polls
        self.input_text = ""
        self.step = 0
        self.url = "https://chat.deepseek.com/"
        self.stops_clicked = 0
        self.sends = 0
        self.new_chats = 0
        self.think_clicks = 0
        self.gotos = 0
        self.keys: list[str] = []
        self._closed = False

    # ------------------------------------------------------------------ playright-ish API

    def is_closed(self) -> bool:
        return self._closed

    def on(self, *_args: Any, **_kw: Any) -> None:
        return None

    @property
    def keyboard(self) -> "FakeKeyboard":
        if not hasattr(self, "_keyboard"):
            self._keyboard = FakeKeyboard(self)
        return self._keyboard

    def locator(self, selector: str) -> "FakeLocator":
        return FakeLocator(self, selector)

    async def goto(self, url: str, **_: Any) -> None:
        self.gotos += 1
        self.blocks = []
        self.input_text = ""
        self.generating = False
        self.step = 0
        self.url = url

    async def title(self) -> str:
        return "DeepSeek Chat"

    async def content(self) -> str:
        return "<html><body>fake</body></html>"

    async def eval_on_selector_all(self, selector: str, expr: str) -> list[str]:
        return ["Send", "Stop", "Create a new chat"]

    # ------------------------------------------------------------------ внутренности

    def _matches(self, selector: str) -> str:
        """Селектор → роль. Неизвестный селектор обязан дать `none`, как это делает
        живой Playwright (count() == 0), а не «матчится на всякий случай»."""
        low = selector.lower()
        if "chat-input" in low or ("textarea" in low and "placeholder" not in low):
            return "input"
        if "send" in low or "отправить" in low:
            return "send"
        if "stop" in low or "останов" in low:
            return "stop"
        if "new chat" in low or "create a new" in low or "новый чат" in low:
            return "new_chat"
        if "deepthink" in low or "мышлен" in low or "思考" in low:
            return "think"
        if "markdown" in low or "message-content" in low:
            return "block"
        return "none"

    def _count(self, role: str) -> int:
        if role == "input":
            return 1
        if role == "send":
            return 0 if not self.accepts_send else 1
        if role == "stop":
            return 1 if self.generating else 0
        if role == "new_chat":
            return 1
        if role == "think":
            return 1
        if role == "none":
            return 0
        return len(self.blocks)

    def _advance_generation(self) -> None:
        if not self.generating or self.pending_answer is None:
            return
        if self.thinking_polls > 0:
            self.thinking_polls -= 1
            return
        self.step += 1
        text = self.pending_answer[: self.step * self.chunk]
        if self.blocks and self.blocks[-1] in ("", self.pending_answer[: len(self.blocks[-1])]):
            self.blocks[-1] = text
        else:
            self.blocks.append(text)
        if len(text) >= len(self.pending_answer):
            self.generating = False

    def _submit(self) -> None:
        """Отправка: единственный честный признак успеха — поле опустело."""
        self.sends += 1
        if not self.accepts_input or not self.accepts_send or not self.input_text.strip():
            return
        self.blocks.append(self.input_text)  # эхо пользователя матчится тем же селектором
        self.input_text = ""
        self.generating = True
        self.step = 0
        self.blocks.append("")  # пустой бабл ассистента появляется до первого символа


class FakeKeyboard:
    def __init__(self, page: FakeChatPage) -> None:
        self.page = page

    async def press(self, key: str) -> None:
        self.page.keys.append(key)
        if key in {"Enter", "Return"}:
            self.page._submit()
        elif key in {"Control+A", "Meta+A"}:
            pass
        elif key == "Delete":
            self.page.input_text = ""

    async def insert_text(self, text: str) -> None:
        if self.page.accepts_input:
            self.page.input_text = text

    async def type(self, text: str, delay: int = 0) -> None:
        await self.insert_text(text)


class FakeLocator:
    def __init__(self, page: FakeChatPage, selector: str, index: int | None = None) -> None:
        self.page = page
        self.selector = selector
        self.role = page._matches(selector)
        self.index = index

    @property
    def first(self) -> "FakeLocator":
        return self

    def nth(self, index: int) -> "FakeLocator":
        return FakeLocator(self.page, self.selector, index)

    async def count(self) -> int:
        return self.page._count(self.role)

    async def wait_for(self, *, state: str | None = None, timeout: int | None = None) -> None:
        if self.page._count(self.role) == 0:
            raise TimeoutError(f"nothing matched {self.selector}")

    async def is_enabled(self) -> bool:
        return True

    async def is_visible(self) -> bool:
        return True

    async def inner_text(self, timeout: int | None = None) -> str:
        if self.role != "block":
            return self.page.input_text
        if self.index is None:
            return self.page.blocks[-1] if self.page.blocks else ""
        if self.index == len(self.page.blocks) - 1:
            # генерация «течёт» ровно на один шаг за poll, а не на каждый прочитанный блок
            self.page._advance_generation()
        return self.page.blocks[self.index] if self.index < len(self.page.blocks) else ""

    async def input_value(self, timeout: int | None = None) -> str:
        if self.role != "input":
            raise ValueError("not an input")
        return self.page.input_text

    async def fill(self, text: str, timeout: int | None = None) -> None:
        if self.page.accepts_input:
            self.page.input_text = text

    async def click(self, timeout: int | None = None) -> None:
        if self.role == "stop":
            self.page.stops_clicked += 1
            self.page.generating = False
        elif self.role == "new_chat":
            self.page.new_chats += 1
            if self.page.new_chat_works:
                self.page.blocks = []
                self.page.input_text = ""
                self.page.generating = False
        elif self.role == "think":
            self.page.think_clicks += 1
        elif self.role == "send":
            self.page._submit()

    async def press(self, key: str, timeout: int | None = None) -> None:
        await self.page.keyboard.press(key)
