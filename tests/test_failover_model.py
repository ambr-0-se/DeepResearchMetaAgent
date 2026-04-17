"""Unit tests for FailoverModel — primary → backup quota-exhaustion switch.

Pure-unit, no API calls. Uses fake `OpenAIServerModel`-shaped objects to drive
the wrapper's behavior. Loads `src.models.failover` via importlib to bypass the
heavy `src/__init__.py` chain (same isolation pattern as
test_reasoning_preservation.py).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FAILOVER_FILE = _REPO_ROOT / "src" / "models" / "failover.py"


def _load_failover():
    """Load failover.py without triggering src/__init__.py (which pulls
    in browser/crawl4ai deps). failover.py only uses TYPE_CHECKING imports
    of OpenAIServerModel/ChatMessage, so we don't need to stub those.
    """
    if "_failover_under_test" in sys.modules:
        return sys.modules["_failover_under_test"]
    spec = importlib.util.spec_from_file_location("_failover_under_test", _FAILOVER_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_failover_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def failover_mod():
    return _load_failover()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeResp:
    """Mimics openai.APIError-shaped exception body."""
    text: str = ""


class _QuotaExhausted(Exception):
    """Mimics a DashScope free-tier 429 carrying a quota-exceeded body."""
    def __init__(self, message="Throttling.FreeTierQuotaExceeded — quota exceeded"):
        super().__init__(message)
        self.status_code = 429
        self.message = message


class _BillingRequired(Exception):
    def __init__(self):
        super().__init__("Payment required")
        self.status_code = 402


class _TransientRateLimit(Exception):
    """A short-window 429 that the underlying retry loop already handled —
    exposed to FailoverModel only when retries are exhausted. NOT a quota
    signal, so failover should NOT trigger."""
    def __init__(self):
        super().__init__("rate limit; retry in 1s")
        self.status_code = 429


class _StubModel:
    """Stand-in for OpenAIServerModel. Records calls; emits configured outputs."""

    def __init__(self, name: str, error_seq=None, result="ok", stream_chunks=None):
        self.model_id = name
        self._last_input_token_count = 17
        self._last_output_token_count = 23
        self._calls = 0
        self._error_seq = list(error_seq or [])
        self._result = result
        self._stream_chunks = list(stream_chunks or [])

    async def generate(self, **kwargs):
        self._calls += 1
        if self._error_seq:
            err = self._error_seq.pop(0)
            if err is not None:
                raise err
        return self._result

    def generate_stream(self, **kwargs):
        if self._error_seq:
            err = self._error_seq.pop(0)
            if err is not None:
                raise err
        yield from self._stream_chunks


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detects_dashscope_free_tier_quota_message(failover_mod):
    assert failover_mod._looks_like_quota_exhaustion(_QuotaExhausted())


def test_detects_402_billing(failover_mod):
    assert failover_mod._looks_like_quota_exhaustion(_BillingRequired())


def test_does_not_treat_short_rate_limit_as_quota(failover_mod):
    """Transient 429 must NOT trigger a permanent failover — that would mask
    fixable hiccups and lock the eval onto the more expensive backup."""
    assert not failover_mod._looks_like_quota_exhaustion(_TransientRateLimit())


def test_quota_pattern_matches_response_body(failover_mod):
    class _ErrWithResp(Exception):
        status_code = 500
        message = ""
        response = _FakeResp(text="HTTP 500 — exceeded your current quota")
    assert failover_mod._looks_like_quota_exhaustion(_ErrWithResp())


# ---------------------------------------------------------------------------
# Generate (async) — single-call failover
# ---------------------------------------------------------------------------


def test_generate_uses_primary_when_healthy(failover_mod):
    primary = _StubModel("p", error_seq=[None], result="from-primary")
    backup = _StubModel("b", result="from-backup")
    fm = failover_mod.FailoverModel(primary, backup, alias="alias")

    out = asyncio.run(fm.generate(messages=[]))
    assert out == "from-primary"
    assert primary._calls == 1
    assert backup._calls == 0
    # Token counters bubble up
    assert fm._last_input_token_count == 17
    assert fm._last_output_token_count == 23


def test_generate_switches_on_quota_exhaustion(failover_mod):
    primary = _StubModel("p", error_seq=[_QuotaExhausted()])
    backup = _StubModel("b", error_seq=[None], result="from-backup")
    fm = failover_mod.FailoverModel(primary, backup, alias="alias")

    out = asyncio.run(fm.generate(messages=[]))
    assert out == "from-backup"
    assert fm._switched is True
    assert primary._calls == 1
    assert backup._calls == 1


def test_generate_does_not_switch_on_transient_rate_limit(failover_mod):
    primary = _StubModel("p", error_seq=[_TransientRateLimit()])
    backup = _StubModel("b")
    fm = failover_mod.FailoverModel(primary, backup, alias="alias")

    with pytest.raises(_TransientRateLimit):
        asyncio.run(fm.generate(messages=[]))
    # Stays on primary — backup untouched
    assert fm._switched is False
    assert backup._calls == 0


def test_switch_is_one_way_and_sticky(failover_mod):
    """After failover, subsequent calls go straight to backup without retrying primary."""
    primary = _StubModel("p", error_seq=[_QuotaExhausted(), None], result="primary-recovered")
    backup = _StubModel("b", error_seq=[None, None], result="from-backup")
    fm = failover_mod.FailoverModel(primary, backup, alias="alias")

    asyncio.run(fm.generate(messages=[]))   # Triggers switch
    out2 = asyncio.run(fm.generate(messages=[]))  # Should NOT touch primary again

    assert out2 == "from-backup"
    assert primary._calls == 1   # one initial failed call only
    assert backup._calls == 2


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_failover_before_first_chunk(failover_mod):
    primary = _StubModel("p", error_seq=[_QuotaExhausted()])
    backup = _StubModel("b", error_seq=[None], stream_chunks=["a", "b", "c"])
    fm = failover_mod.FailoverModel(primary, backup, alias="alias")

    out = list(fm.generate_stream(messages=[]))
    assert out == ["a", "b", "c"]
    assert fm._switched is True


def test_stream_no_switch_when_primary_streams_clean(failover_mod):
    primary = _StubModel("p", error_seq=[None], stream_chunks=["x", "y"])
    backup = _StubModel("b", stream_chunks=["never"])
    fm = failover_mod.FailoverModel(primary, backup, alias="alias")

    out = list(fm.generate_stream(messages=[]))
    assert out == ["x", "y"]
    assert fm._switched is False


# ---------------------------------------------------------------------------
# Attribute proxying
# ---------------------------------------------------------------------------


def test_passthrough_unknown_attribute_to_active(failover_mod):
    primary = _StubModel("p")
    primary.flatten_messages_as_text = "from-primary"
    backup = _StubModel("b")
    backup.flatten_messages_as_text = "from-backup"
    fm = failover_mod.FailoverModel(primary, backup, alias="alias")

    assert fm.flatten_messages_as_text == "from-primary"
    fm._switched = True
    assert fm.flatten_messages_as_text == "from-backup"


def test_alias_overrides_child_model_id(failover_mod):
    primary = _StubModel("p-id")
    backup = _StubModel("b-id")
    fm = failover_mod.FailoverModel(primary, backup, alias="public-alias")
    assert fm.model_id == "public-alias"
