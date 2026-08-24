"""Process-owned MongoDB client manager with safe concurrency and idempotent shutdown."""

import threading
from collections.abc import Callable
from typing import ClassVar, Optional

from motor.motor_asyncio import AsyncIOMotorClient

from jobapply.settings import get_settings


class MongoClientManager:
    """Thread-safe process singleton manager for AsyncIOMotorClient with condition-synchronized shutdown."""

    _client: ClassVar[Optional[AsyncIOMotorClient]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _condition: ClassVar[threading.Condition] = threading.Condition(_lock)
    _is_closing: ClassVar[bool] = False
    _reset_hooks: ClassVar[list[Callable[[], None]]] = []

    @classmethod
    def register_reset_hook(cls, hook: Callable[[], None]) -> None:
        """Register a synchronous callback to reset repository lifecycle/index states upon shutdown."""
        with cls._lock:
            if hook not in cls._reset_hooks:
                cls._reset_hooks.append(hook)

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """Get or initialize the process-owned AsyncIOMotorClient instance."""
        with cls._condition:
            while cls._is_closing:
                cls._condition.wait()
            if cls._client is None:
                settings = get_settings()
                cls._client = AsyncIOMotorClient(settings.mongodb_url)
            return cls._client

    @classmethod
    async def close(cls) -> None:
        """Idempotently close the owned Motor client and reset all repository states exactly once, coordinating concurrent callers."""
        client_to_close = None
        hooks_to_run: list[Callable[[], None]] = []

        with cls._condition:
            while cls._is_closing:
                cls._condition.wait()
                if cls._client is None:
                    return

            cls._is_closing = True
            if cls._client is not None:
                client_to_close = cls._client
                cls._client = None
            hooks_to_run = list(cls._reset_hooks)

        errors: list[Exception] = []
        try:
            # Run reset hooks outside the manager lock so repositories can acquire their own locks
            for hook in hooks_to_run:
                try:
                    hook()
                except Exception as e:
                    errors.append(e)

            if client_to_close is not None:
                if hasattr(client_to_close, "close"):
                    try:
                        client_to_close.close()
                    except Exception as e:
                        errors.append(e)
        finally:
            with cls._condition:
                cls._is_closing = False
                cls._condition.notify_all()

        if errors:
            raise errors[0]
