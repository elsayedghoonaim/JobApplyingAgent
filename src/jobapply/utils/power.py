"""Run-scoped Windows power-management support."""

from __future__ import annotations

import ctypes
import logging
import os
from contextlib import contextmanager
from functools import wraps
from threading import RLock
from typing import Any, Callable, Iterator, TypeVar

ES_SYSTEM_REQUIRED = 0x00000001
ES_CONTINUOUS = 0x80000000

_LOGGER = logging.getLogger("jobapply")
_LOCK = RLock()
_ACTIVE_GUARDS = 0
_POWER_REQUEST_ACTIVE = False

F = TypeVar("F", bound=Callable[..., Any])


def _set_thread_execution_state(flags: int) -> bool:
    """Set the current Windows thread's execution requirement."""
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    return bool(function(flags))


@contextmanager
def prevent_system_sleep() -> Iterator[bool]:
    """Prevent idle system sleep while at least one workflow guard is active.

    The display flag is deliberately omitted so Windows may turn the screen off.
    Explicit user sleep, power-button, and laptop-lid actions remain effective.
    """
    global _ACTIVE_GUARDS, _POWER_REQUEST_ACTIVE

    acquired = False
    supported = os.name == "nt"
    with _LOCK:
        if _ACTIVE_GUARDS:
            _ACTIVE_GUARDS += 1
            acquired = _POWER_REQUEST_ACTIVE
        elif supported:
            try:
                acquired = _set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            except Exception as exc:
                _LOGGER.warning("Could not prevent Windows sleep during this run: %s", exc)
            if acquired:
                _ACTIVE_GUARDS = 1
                _POWER_REQUEST_ACTIVE = True
            else:
                _LOGGER.warning("Windows rejected the request to prevent sleep during this run.")

    try:
        yield acquired
    finally:
        with _LOCK:
            if acquired and _ACTIVE_GUARDS:
                _ACTIVE_GUARDS -= 1
                if _ACTIVE_GUARDS == 0 and _POWER_REQUEST_ACTIVE:
                    try:
                        if not _set_thread_execution_state(ES_CONTINUOUS):
                            _LOGGER.warning(
                                "Windows rejected the request to restore normal sleep behavior."
                            )
                    except Exception as exc:
                        _LOGGER.warning("Could not restore normal Windows sleep behavior: %s", exc)
                    finally:
                        _POWER_REQUEST_ACTIVE = False


def keep_system_awake(function: F) -> F:
    """Decorate an async workflow so its complete lifetime prevents idle sleep."""

    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        with prevent_system_sleep():
            return await function(*args, **kwargs)

    return wrapped  # type: ignore[return-value]
