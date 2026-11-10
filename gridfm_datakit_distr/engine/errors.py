"""Errors raised by the engine layer."""

from __future__ import annotations


class DssError(RuntimeError):
    """An OpenDSS command or API call failed.

    Attributes:
        command: the command text (or a description of the API call) that failed.
        message: the engine's error message.
        code: the OpenDSS error number, if the engine reported one.
    """

    def __init__(self, command: str, message: str, code: int | None = None) -> None:
        self.command = command
        self.message = message
        self.code = code
        prefix = f"OpenDSS error {code}: " if code is not None else ""
        super().__init__(f"{prefix}{message} [command: {command!r}]")
