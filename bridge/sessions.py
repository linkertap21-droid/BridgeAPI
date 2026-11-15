"""Локальное хранилище диалогов.

Мост stateless по API, но историю держать нужно: чтобы OpenAI-клиенты могли
присылать только последнее сообщение, а полный контекст подмешивался сам.

Отдельно хранится флаг `chat_live`: он означает «в браузере сейчас открыт тот же
самый диалог, поэтому пересылать историю не надо». После рестарта моста или
`new chat` в браузере флаг сбрасывается, и следующий запрос уйдёт с полным
транскриптом — так контекст не теряется.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]")


@dataclass
class Turn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str = "deepseek-chat"
    chat_live: bool = False
    turns: list[Turn] = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def transcript(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["turn_count"] = self.turn_count
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        turns = [Turn(**t) if isinstance(t, dict) else t for t in data.get("turns", [])]
        return cls(
            id=data["id"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            model=data.get("model", "deepseek-chat"),
            chat_live=bool(data.get("chat_live", False)),
            turns=turns,
        )


class SessionStore:
    """Простое JSON-хранилище: один файл на диалог + кэш в памяти."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def _file(self, session_id: str) -> Path:
        return self.directory / f"{_SAFE_ID.sub('_', session_id)}.json"

    def new_id(self) -> str:
        return f"srv-{uuid.uuid4().hex[:12]}"

    def get(self, session_id: str) -> Session | None:
        if session_id in self._cache:
            return self._cache[session_id]
        path = self._file(session_id)
        if not path.is_file():
            return None
        try:
            session = Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
        self._cache[session_id] = session
        return session

    def get_or_create(self, session_id: str | None, *, model: str = "deepseek-chat") -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                if model and existing.model != model:
                    # сменился режим (reasoner ⇄ chat) — в браузере это другой диалог
                    existing.model = model
                    existing.chat_live = False
                    self.save(existing)
                return existing
            session = Session(id=session_id, model=model or "deepseek-chat")
        else:
            session = Session(id=self.new_id(), model=model or "deepseek-chat")
        self.save(session)
        return session

    def append(self, session: Session, role: str, content: str) -> Turn:
        turn = Turn(role=role, content=content)
        session.turns.append(turn)
        session.updated_at = turn.ts
        self.save(session)
        return turn

    def save(self, session: Session) -> None:
        self._cache[session.id] = session
        path = self._file(session.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def mark_live(self, session: Session, *, live: bool, model: str | None = None) -> None:
        if model:
            session.model = model
        session.chat_live = live
        self.save(session)

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for session in self.iter_all():
            items.append(
                {
                    "id": session.id,
                    "model": session.model,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "turn_count": session.turn_count,
                    "chat_live": session.chat_live,
                    "preview": (session.turns[-1].content[:120] if session.turns else ""),
                }
            )
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return items[:limit]

    def iter_all(self) -> Iterator[Session]:
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session.from_dict(data)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            self._cache[session.id] = session
            yield session

    def delete(self, session_id: str) -> bool:
        self._cache.pop(session_id, None)
        path = self._file(session_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        count = 0
        for path in list(self.directory.glob("*.json")):
            path.unlink()
            count += 1
        self._cache.clear()
        return count
