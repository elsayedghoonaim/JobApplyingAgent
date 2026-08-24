import os
import socket
import sys

import pytest
import pytest_socket

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# On Windows, asyncio ProactorEventLoop requires internal socketpair for its self-pipe.
# Under pytest-socket --disable-socket, socket.socket constructor is guarded.
# Un-guard ONLY calls originating within stdlib socket.socketpair so offline asyncio loops initialize.
_in_socketpair = False
_orig_socket = socket.socket
_orig_disable = pytest_socket.disable_socket


def _patch_pytest_socket():
    def _custom_disable(allow_unix_socket: bool = False):
        _orig_disable(allow_unix_socket)
        guarded = socket.socket

        class OfflineSocket(guarded):
            def __new__(cls, family=-1, type=-1, proto=-1, fileno=None):
                global _in_socketpair
                if _in_socketpair:
                    obj = _orig_socket.__new__(_orig_socket)
                    if fileno is not None:
                        _orig_socket.__init__(obj, family, type, proto, fileno)
                    else:
                        _orig_socket.__init__(obj, family, type, proto)
                    return obj
                return guarded.__new__(guarded, family, type, proto, fileno)

            def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
                global _in_socketpair
                if _in_socketpair:
                    return
                super().__init__(family, type, proto, fileno)

        socket.socket = OfflineSocket

    pytest_socket.disable_socket = _custom_disable
    _custom_disable()


_patch_pytest_socket()

_orig_socketpair = socket.socketpair


def _safe_socketpair(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0):
    global _in_socketpair
    _in_socketpair = True
    try:
        return _orig_socketpair(family, type, proto)
    finally:
        _in_socketpair = False


socket.socketpair = _safe_socketpair


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
