"""Ошибки моста и их отображение в HTTP.

Клиент (curl, SDK, расширение) должен по коду ошибки понимать, что делать:
перелогиниться, подождать, сократить запрос или это баг в мосте.
"""

from __future__ import annotations


class BridgeError(Exception):
    """База. `status_code` + машиночитаемый `code` + человекочитаемое `message`."""

    status_code: int = 500
    code: str = "bridge_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_payload(self) -> dict[str, object]:
        error: dict[str, object] = {"message": self.message, "type": self.code}
        if self.hint:
            error["hint"] = self.hint
        return {"error": error}


class InvalidRequest(BridgeError):
    status_code = 400
    code = "invalid_request"


class NeedLogin(BridgeError):
    status_code = 503
    code = "login_required"

    def __init__(self) -> None:
        super().__init__(
            "Браузер моста не залогинен в chat.deepseek.com.",
            hint=(
                "Запусти `python -m bridge.cli login`, войди в аккаунт и повтори запрос. "
                "Если чат открыт, но «залип» — сначала `python -m bridge.cli reset`."
            ),
        )


class DriverNotReady(BridgeError):
    status_code = 503
    code = "driver_not_ready"

    def __init__(self, message: str = "Драйвер браузера ещё не запущен.", **kw: object) -> None:
        super().__init__(message, **kw)  # type: ignore[arg-type]


class Busy(BridgeError):
    status_code = 429
    code = "bridge_busy"

    def __init__(self, waited_s: float) -> None:
        super().__init__(
            f"Мост занят другим запросом (ожидание {waited_s:.1f} с превышено).",
            hint="В одном браузере может генерироваться только один ответ; повтори позже.",
        )


class GenerationTimeout(BridgeError):
    status_code = 504
    code = "generation_timeout"

    def __init__(
        self,
        timeout_s: float,
        *,
        message: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message or f"DeepSeek не закончил генерацию за {timeout_s:.0f} с.",
            hint=hint or "Увеличь BRIDGE_GENERATION_TIMEOUT_S или отправь более короткий запрос.",
        )


class DriverCrashed(BridgeError):
    status_code = 502
    code = "driver_crashed"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(
            f"Страница чата недоступна: {message}",
            hint=hint
            or "Сайт мог сменить вёрстку. Проверь `python -m bridge.cli probe` и selectors.json.",
        )
