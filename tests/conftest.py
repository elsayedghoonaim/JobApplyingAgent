import os
import socket
import sys
import threading
from unittest.mock import AsyncMock

import pytest
import pytest_socket

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# On Windows, asyncio ProactorEventLoop requires internal socketpair for its self-pipe.
# Under pytest-socket --disable-socket, socket.socket constructor is guarded.
# Un-guard ONLY calls originating within stdlib socket.socketpair so offline asyncio loops initialize.
# The marker must be thread-local because some tests create multiple event loops concurrently.
_socketpair_state = threading.local()
_orig_socket = socket.socket
_orig_disable = pytest_socket.disable_socket


def _inside_socketpair() -> bool:
    return bool(getattr(_socketpair_state, "active", False))


def _patch_pytest_socket():
    def _custom_disable(allow_unix_socket: bool = False):
        _orig_disable(allow_unix_socket)
        guarded = socket.socket

        class OfflineSocket(guarded):
            def __new__(cls, family=-1, type=-1, proto=-1, fileno=None):
                if _inside_socketpair():
                    obj = _orig_socket.__new__(_orig_socket)
                    if fileno is not None:
                        _orig_socket.__init__(obj, family, type, proto, fileno)
                    else:
                        _orig_socket.__init__(obj, family, type, proto)
                    return obj
                return guarded.__new__(guarded, family, type, proto, fileno)

            def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
                if _inside_socketpair():
                    return
                super().__init__(family, type, proto, fileno)

        socket.socket = OfflineSocket

    pytest_socket.disable_socket = _custom_disable
    _custom_disable()


_patch_pytest_socket()

_orig_socketpair = socket.socketpair


def _safe_socketpair(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0):
    previous = _inside_socketpair()
    _socketpair_state.active = True
    try:
        return _orig_socketpair(family, type, proto)
    finally:
        _socketpair_state.active = previous


socket.socketpair = _safe_socketpair


@pytest.fixture(autouse=True)
def _isolate_post_submit_ambiguity_tests(request, monkeypatch):
    """Keep post-submit ambiguity tests focused on the submit boundary itself.

    Those tests use deliberately minimal Playwright mocks. New pre-submit DOM and
    account-safety inspections are exercised elsewhere, so neutralize them only
    for these two tests to prevent unrelated mock behavior from short-circuiting
    before the submit/confirmation paths under test.
    """
    target_tests = {
        "test_execution_node_unconfirmed_timeout_halts_with_pause_and_notification_route",
        "test_execution_node_submit_click_exception_treated_as_ambiguous_review",
    }
    if request.node.name in target_tests:
        import jobapply.nodes.execution as execution_module
        from jobapply.execution import RequiredFieldValidationResult

        monkeypatch.setattr(
            execution_module,
            "guard_page_account_safety",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            execution_module,
            "validate_visible_required_controls",
            AsyncMock(
                return_value=RequiredFieldValidationResult(
                    is_valid=True,
                    unresolved_fields=[],
                )
            ),
        )
    yield


@pytest.fixture(autouse=True)
async def _reset_test_mongo_and_cache_state():
    from jobapply.utils.mongo import MongoClientManager
    from jobapply.utils.source_cache import reset_source_cache

    await MongoClientManager.close()
    reset_source_cache()
    try:
        yield
    finally:
        await MongoClientManager.close()
        reset_source_cache()
