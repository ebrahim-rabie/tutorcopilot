import pytest

from app.generate import retry_transient


def test_retry_transient_recovers_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("upstream blip")
        return "ok"

    assert retry_transient(flaky, attempts=3, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_retry_transient_raises_after_exhaustion():
    def always_fails():
        raise RuntimeError("permanently dead")

    with pytest.raises(RuntimeError, match="permanently dead"):
        retry_transient(always_fails, attempts=2, base_delay=0)


def test_retry_transient_success_is_passthrough():
    assert retry_transient(lambda: 42, attempts=3, base_delay=0) == 42
