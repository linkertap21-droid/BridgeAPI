#!/usr/bin/env python3
"""Мост как обычный OpenAI-эндпоинт.

Требует `pip install openai`. Запущенный BridgeAPI предполагается:
    python -m bridge.cli serve

Смысл примера в том, что клиентский код не отличается от работы с api.openai.com —
вся специфика «браузера вместо API» спрятана в мосте.
"""

from __future__ import annotations

import os

from openai import OpenAI

BASE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8787/v1")
SESSION = os.environ.get("BRIDGE_SESSION", "demo-session")

client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("BRIDGE_TOKEN") or "bridge")


def ask(prompt: str, *, stream: bool = True) -> str:
    stream_id = None
    try:
        stream_id = client.chat.completions.create(  # type: ignore[assignment]
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Отвечай по-русски и по делу."},
                {"role": "user", "content": prompt},
            ],
            extra_body={"session_id": SESSION},
            stream=stream,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"мост ответил ошибкой: {exc}")
        print("подсказка: проверь `python -m bridge.cli status` — скорее всего, нужен login")
        raise SystemExit(1) from exc

    if not stream:
        return stream_id.choices[0].message.content or ""  # type: ignore[union-attr]

    pieces: list[str] = []
    for chunk in stream_id:  # type: ignore[union-attr]
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            pieces.append(delta)
            print(delta, end="", flush=True)
    print()
    return "".join(pieces)


if __name__ == "__main__":
    print("ты > сколько будет 17*23?")
    answer = ask("Сколько будет 17*23? Только число.")
    print("мост >", answer)
    print("\nты > проверь: умножь это число на 2")
    print("мост >", ask("Умножь предыдущий результат на 2. Только число."))
    print("\n(диалог продолжался в одной сессии — контекст держит мост)")
