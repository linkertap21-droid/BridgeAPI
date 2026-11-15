"""Схемы запросов/ответов.

Отдельный файл, потому что OpenAI-совместимость — это контракт: её проверяют
SDK-клиенты, и она не должна «случайно» измениться вместе с роутерами.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool", "developer", "function"] = "user"
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions — подмножество OpenAI + поле `session_id`."""

    model_config = ConfigDict(extra="ignore")

    model: str = "deepseek-chat"
    messages: list[ChatMessage] = Field(default_factory=list)
    prompt: str | None = None
    stream: bool = False
    session_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    @model_validator(mode="after")
    def _require_content(self) -> "ChatCompletionRequest":
        if not self.messages and not self.prompt:
            raise ValueError("нужно передать `messages` или `prompt`")
        return self

    def as_messages(self) -> list[dict[str, str]]:
        if self.messages:
            return [_flatten(m) for m in self.messages]
        return [{"role": "user", "content": self.prompt or ""}]


def _flatten(message: ChatMessage) -> dict[str, str]:
    content = message.content
    if isinstance(content, list):  # мультимодальный формат OpenAI: [{type:"text",...}]
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    elif not isinstance(content, str):
        content = str(content)
    return {"role": message.role, "content": content}


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Честно предупреждаем: оценка по длине текста, а не настоящий tokenizer.
    estimated: bool = True


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    session_id: str | None = None
    bridge: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """POST /v1/chat — упрощённый эндпоинт: присылаешь строку, получаешь строку."""

    model_config = ConfigDict(extra="ignore")

    prompt: str
    system: str | None = None
    session_id: str | None = None
    model: str = "deepseek-chat"
    stream: bool = False


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str = "deepseek-chat"
    finish_reason: str = "stop"
    rebuilt_context: bool = False
    elapsed_s: float = 0.0
    usage: Usage = Field(default_factory=Usage)


def new_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def chunk(
    completion_id: str,
    model: str,
    created: int,
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Один SSE-чанк в формате OpenAI."""
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage:
        payload["usage"] = usage
    return payload


def created_ts() -> int:
    return int(time.time())


__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "ChoiceMessage",
    "Usage",
    "chunk",
    "created_ts",
    "new_id",
]
