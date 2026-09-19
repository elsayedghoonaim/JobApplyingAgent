"""Tests for run-scoped Windows sleep prevention."""

import asyncio
from unittest.mock import Mock

import pytest

from jobapply.utils import power


def _reset_power_state() -> None:
    power._ACTIVE_GUARDS = 0
    power._POWER_REQUEST_ACTIVE = False


def test_prevent_system_sleep_acquires_once_for_nested_guards(monkeypatch):
    _reset_power_state()
    setter = Mock(return_value=True)
    monkeypatch.setattr(power.os, "name", "nt")
    monkeypatch.setattr(power, "_set_thread_execution_state", setter)

    with power.prevent_system_sleep() as outer:
        with power.prevent_system_sleep() as inner:
            assert outer is True
            assert inner is True
            assert setter.call_count == 1

    assert [call.args[0] for call in setter.call_args_list] == [
        power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED,
        power.ES_CONTINUOUS,
    ]
    assert power._ACTIVE_GUARDS == 0
    assert power._POWER_REQUEST_ACTIVE is False


def test_prevent_system_sleep_failure_does_not_restore(monkeypatch):
    _reset_power_state()
    setter = Mock(return_value=False)
    monkeypatch.setattr(power.os, "name", "nt")
    monkeypatch.setattr(power, "_set_thread_execution_state", setter)

    with power.prevent_system_sleep() as acquired:
        assert acquired is False

    setter.assert_called_once_with(power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED)
    assert power._ACTIVE_GUARDS == 0


def test_prevent_system_sleep_is_noop_off_windows(monkeypatch):
    _reset_power_state()
    setter = Mock()
    monkeypatch.setattr(power.os, "name", "posix")
    monkeypatch.setattr(power, "_set_thread_execution_state", setter)

    with power.prevent_system_sleep() as acquired:
        assert acquired is False

    setter.assert_not_called()


@pytest.mark.asyncio
async def test_keep_system_awake_restores_state_after_cancellation(monkeypatch):
    _reset_power_state()
    setter = Mock(return_value=True)
    monkeypatch.setattr(power.os, "name", "nt")
    monkeypatch.setattr(power, "_set_thread_execution_state", setter)

    @power.keep_system_awake
    async def cancelled_run():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancelled_run()

    assert [call.args[0] for call in setter.call_args_list] == [
        power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED,
        power.ES_CONTINUOUS,
    ]
