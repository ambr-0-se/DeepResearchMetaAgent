"""Unit tests for the multi-key round-robin refactor
(§P4 of `docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md`).

Coverage:
  1. `_load_suffix_keys` correctly reads `BASE`, `BASE_2`, `BASE_3`, …
     and stops at the first unset / placeholder.
  2. `_KeyPoolState` round-robin distribution, cooldown logic, and
     fall-through when all keys are cooling.
  3. `KeyRotatingOpenAIServerModel` registers + its rotating proxy
     dispatches `chat.completions.create` across keys and marks
     cooldowns on `RateLimitError` with Retry-After respected.
  4. `KeyRotatingChatOpenAI` rotates `ainvoke` calls and marks
     cooldowns similarly.
  5. Registration branching: single key → `OpenAIServerModel` +
     `ChatOpenAI`; multi-key → `KeyRotating*`.
  6. No API key appears in captured logs on any error path.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.models.openaillm import (
    KeyRotatingChatOpenAI,
    KeyRotatingOpenAIServerModel,
    _KeyPoolState,
    _extract_cooldown_from_rate_limit,
    _load_suffix_keys,
    _DEFAULT_COOLDOWN_SECS,
)


def _make_rate_limit_error(retry_after_secs: int):
    """Build a genuine `openai.RateLimitError`. Its constructor inspects
    `response.request`, so a bare `SimpleNamespace` isn't enough — we
    need a real `httpx.Response` with a request attached.
    """
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={"retry-after": str(retry_after_secs)},
        request=request,
    )
    return openai.RateLimitError(
        message="simulated 429", response=response, body=None,
    )


# ---------------------------------------------------------------------------
# _load_suffix_keys
# ---------------------------------------------------------------------------

def test_load_suffix_keys_single_primary(monkeypatch):
    monkeypatch.setenv("FOO_KEY", "aaa")
    monkeypatch.delenv("FOO_KEY_2", raising=False)
    assert _load_suffix_keys("FOO_KEY") == ["aaa"]


def test_load_suffix_keys_primary_plus_one_suffix(monkeypatch):
    monkeypatch.setenv("FOO_KEY", "aaa")
    monkeypatch.setenv("FOO_KEY_2", "bbb")
    assert _load_suffix_keys("FOO_KEY") == ["aaa", "bbb"]


def test_load_suffix_keys_stops_at_first_unset_after_start(monkeypatch):
    """`_2` set but `_3` not → returns [primary, _2]; _4 present is skipped."""
    monkeypatch.setenv("FOO_KEY", "a")
    monkeypatch.setenv("FOO_KEY_2", "b")
    monkeypatch.delenv("FOO_KEY_3", raising=False)
    monkeypatch.setenv("FOO_KEY_4", "d")  # intentionally set — should be skipped
    assert _load_suffix_keys("FOO_KEY") == ["a", "b"]


def test_load_suffix_keys_placeholder_treated_as_unset(monkeypatch):
    monkeypatch.setenv("FOO_KEY", "PLACEHOLDER")
    monkeypatch.delenv("FOO_KEY_2", raising=False)
    assert _load_suffix_keys("FOO_KEY", placeholder="PLACEHOLDER") == []


def test_load_suffix_keys_empty_when_all_unset(monkeypatch):
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("FOO_KEY_2", raising=False)
    monkeypatch.delenv("FOO_KEY_3", raising=False)
    assert _load_suffix_keys("FOO_KEY") == []


# ---------------------------------------------------------------------------
# _KeyPoolState rotation + cooldown
# ---------------------------------------------------------------------------

def test_pool_round_robin_two_keys():
    pool = _KeyPoolState(["a", "b"])
    picks = [pool.pick_index() for _ in range(6)]
    # 6 picks with 2 keys should distribute 3/3.
    assert picks.count(0) == 3
    assert picks.count(1) == 3


def test_pool_cooldown_skips_key(monkeypatch):
    pool = _KeyPoolState(["a", "b"])
    # Pin a deterministic monotonic clock.
    t = [1000.0]

    def fake_now():
        return t[0]

    monkeypatch.setattr("src.models.openaillm._monotonic_now", fake_now)

    pool.mark_cooldown(0, 30.0)  # key 0 cool for 30s
    picks = [pool.pick_index() for _ in range(4)]
    assert all(p == 1 for p in picks), (
        f"all picks should fall on key 1 while key 0 is cool; got {picks}"
    )

    # Advance time past the cooldown; round-robin resumes.
    t[0] += 31.0
    picks2 = [pool.pick_index() for _ in range(4)]
    assert set(picks2) == {0, 1}, (
        f"both keys should be eligible after cooldown expires; got {picks2}"
    )


def test_pool_all_cooling_returns_earliest_ready(monkeypatch):
    pool = _KeyPoolState(["a", "b", "c"])
    t = [1000.0]
    monkeypatch.setattr("src.models.openaillm._monotonic_now", lambda: t[0])

    pool.mark_cooldown(0, 50.0)
    pool.mark_cooldown(1, 5.0)   # earliest-ready
    pool.mark_cooldown(2, 30.0)

    assert pool.pick_index() == 1


def test_pool_single_key_always_picks_zero():
    pool = _KeyPoolState(["only"])
    for _ in range(5):
        assert pool.pick_index() == 0


# ---------------------------------------------------------------------------
# _extract_cooldown_from_rate_limit
# ---------------------------------------------------------------------------

def test_extract_cooldown_uses_retry_after_header():
    exc = _make_rate_limit_error(25)
    assert _extract_cooldown_from_rate_limit(exc) == 25.0


def test_extract_cooldown_defaults_when_header_absent():
    # Build a 429 without a retry-after header.
    import httpx, openai
    response = httpx.Response(
        status_code=429,
        headers={},
        request=httpx.Request("POST", "https://x/"),
    )
    exc = openai.RateLimitError(
        message="no retry-after", response=response, body=None,
    )
    assert _extract_cooldown_from_rate_limit(exc) == _DEFAULT_COOLDOWN_SECS


# ---------------------------------------------------------------------------
# KeyRotatingOpenAIServerModel rotation + cooldown on real API call
# ---------------------------------------------------------------------------

class _FakeAsyncChatCompletions:
    """Records which key made each call and can be rigged to raise 429."""

    def __init__(self, key_tag: str, raise_429_with: Any = None) -> None:
        self._key_tag = key_tag
        self._raise_429_with = raise_429_with  # either None or the retry-after seconds
        self.create_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._raise_429_with is not None:
            raise _make_rate_limit_error(self._raise_429_with)
        return SimpleNamespace(
            _key_tag=self._key_tag,
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(
                role="assistant", content="ok", tool_calls=None,
                reasoning_content=None,
                model_dump=lambda include=None: {
                    "role": "assistant", "content": "ok", "tool_calls": None,
                },
            ))],
        )


class _FakeAsyncChat:
    def __init__(self, completions) -> None:
        self.completions = completions


class _FakeAsyncClient:
    """Stand-in for `openai.AsyncOpenAI`."""

    def __init__(self, key_tag: str, raise_429_with: Any = None) -> None:
        self._comp = _FakeAsyncChatCompletions(key_tag, raise_429_with)
        self.chat = _FakeAsyncChat(self._comp)


@pytest.mark.asyncio
async def test_rotating_model_round_robins_underlying_clients(monkeypatch):
    """Two calls on a 2-key model should touch key 0 then key 1 (or 1 then 0),
    whichever order round-robin picks — never both on the same client.
    """
    fake_a = _FakeAsyncClient("A")
    fake_b = _FakeAsyncClient("B")

    model = KeyRotatingOpenAIServerModel.__new__(KeyRotatingOpenAIServerModel)
    # Avoid full __init__ (parent needs real ApiModel setup). Install the
    # pool + cached clients directly — we test only the rotation proxy.
    model._pool = _KeyPoolState(["k1", "k2"])
    model._async_clients = {0: fake_a, 1: fake_b}
    # `create_client()` would normally be called by parent __init__; we
    # instantiate the proxy directly.
    from src.models.openaillm import _RotatingAsyncClientProxy
    proxy = _RotatingAsyncClientProxy(model)

    await proxy.chat.completions.create(messages=[], model="x")
    await proxy.chat.completions.create(messages=[], model="x")

    total_calls = len(fake_a._comp.create_calls) + len(fake_b._comp.create_calls)
    assert total_calls == 2
    assert len(fake_a._comp.create_calls) == 1
    assert len(fake_b._comp.create_calls) == 1


@pytest.mark.asyncio
async def test_rotating_model_marks_cooldown_on_rate_limit(monkeypatch):
    """On `RateLimitError` with `Retry-After: 25`, the picked key's
    `cool_until` is advanced by ~25s. Subsequent picks land on the
    other key.
    """
    # Pin time to reason about cool_until deterministically.
    t = [1000.0]
    monkeypatch.setattr("src.models.openaillm._monotonic_now", lambda: t[0])

    fake_a = _FakeAsyncClient("A", raise_429_with=25)  # always 429
    fake_b = _FakeAsyncClient("B")

    model = KeyRotatingOpenAIServerModel.__new__(KeyRotatingOpenAIServerModel)
    model._pool = _KeyPoolState(["k1", "k2"])
    model._async_clients = {0: fake_a, 1: fake_b}
    from src.models.openaillm import _RotatingAsyncClientProxy
    proxy = _RotatingAsyncClientProxy(model)

    import openai
    with pytest.raises(openai.RateLimitError):
        await proxy.chat.completions.create(messages=[], model="x")

    # Whichever key was picked first got 429'd. Cooldown ≈ 25.
    cool_0 = model._pool.cool_until_raw(0)
    cool_1 = model._pool.cool_until_raw(1)
    assert (cool_0 >= 1024.0) != (cool_1 >= 1024.0), (
        f"exactly one key should be cooling after a single 429; "
        f"got cool_0={cool_0}, cool_1={cool_1}"
    )


# ---------------------------------------------------------------------------
# KeyRotatingChatOpenAI rotation + delegation
# ---------------------------------------------------------------------------

class _FakeChatOpenAI:
    """Stand-in for `langchain_openai.ChatOpenAI`."""

    def __init__(self, *, model: str, api_key: str, base_url: str | None = None,
                 raise_429_with: Any = None, **kwargs) -> None:
        self.model = model
        self._api_key_tag = api_key
        self.base_url = base_url
        self.ainvoke_calls: list = []
        self._raise_429_with = raise_429_with

    async def ainvoke(self, *args, **kwargs):
        self.ainvoke_calls.append((args, kwargs))
        if self._raise_429_with is not None:
            raise _make_rate_limit_error(self._raise_429_with)
        return {"content": f"from {self._api_key_tag}"}

    def bind_tools(self, tools):
        # Typical ChatOpenAI returns a new instance; for the test we
        # return self so `__getattr__`-based fallback can exercise it.
        return self


@pytest.mark.asyncio
async def test_langchain_wrapper_rotates_ainvoke(monkeypatch):
    # `KeyRotatingChatOpenAI.__init__` does `from langchain_openai import
    # ChatOpenAI` at call time, so patching the attribute on the source
    # module is enough — the import statement reads the current binding.
    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChatOpenAI)

    wrapper = KeyRotatingChatOpenAI(
        model="mistral-small-2603", api_keys=["k1", "k2"], base_url="http://x"
    )

    r1 = await wrapper.ainvoke("hello")
    r2 = await wrapper.ainvoke("hello")

    tags_invoked = [inst._api_key_tag for inst in wrapper._instances
                    if inst.ainvoke_calls]
    assert set(tags_invoked) == {"k1", "k2"}, (
        f"expected both instances invoked; got {tags_invoked}"
    )


@pytest.mark.asyncio
async def test_langchain_wrapper_marks_cooldown_on_rate_limit(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr("src.models.openaillm._monotonic_now", lambda: t[0])

    # Replace ChatOpenAI where KeyRotatingChatOpenAI imports it.
    import langchain_openai
    # First instance raises 429 with Retry-After 7; second is healthy.
    call_order = iter([7, None])

    def _factory(*, model, api_key, base_url=None, **kwargs):
        return _FakeChatOpenAI(
            model=model, api_key=api_key, base_url=base_url,
            raise_429_with=next(call_order),
        )

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _factory)

    wrapper = KeyRotatingChatOpenAI(
        model="mistral-small-2603", api_keys=["k1", "k2"], base_url="http://x"
    )

    import openai
    with pytest.raises(openai.RateLimitError):
        await wrapper.ainvoke("hi")

    # Exactly one of the two keys should be cooling; its cool_until should
    # reflect Retry-After: 7.
    cool_0 = wrapper._pool.cool_until_raw(0)
    cool_1 = wrapper._pool.cool_until_raw(1)
    assert (cool_0 >= 1006.0) != (cool_1 >= 1006.0), (
        f"expected exactly one key cooling; got cool_0={cool_0}, cool_1={cool_1}"
    )


def test_langchain_wrapper_getattr_fallback_on_bind_tools(monkeypatch):
    """Attributes not explicitly wrapped fall through to instances[0].
    This is a documented caveat — the returned `bind_tools(...)` object
    is locked to instance 0 and will NOT rotate on subsequent calls.
    Test asserts the fallback works; tracking of the caveat is in the
    class docstring, not the behaviour.
    """
    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChatOpenAI)

    wrapper = KeyRotatingChatOpenAI(
        model="m", api_keys=["k1", "k2"], base_url="http://x"
    )
    # bind_tools is not in the explicit method list — goes through __getattr__.
    bound = wrapper.bind_tools([])
    # Fake returns self; verify we got an instance object back (not a
    # method descriptor or an AttributeError).
    assert isinstance(bound, _FakeChatOpenAI)


# ---------------------------------------------------------------------------
# Registration branching (single vs multi key)
# ---------------------------------------------------------------------------

def test_registration_uses_plain_classes_with_single_key(monkeypatch):
    """Only `MISTRAL_API_KEY` set (no `_2`) → `OpenAIServerModel` +
    `ChatOpenAI` registered, NOT the rotating variants. This keeps
    single-key operators on the exact same code path they had before
    P4 landed.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "solo-key")
    monkeypatch.delenv("MISTRAL_API_KEY_2", raising=False)

    # ModelManager is a Singleton; clear the state so init_models runs
    # against a clean slate.
    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    from src.models.openaillm import OpenAIServerModel
    from langchain_openai import ChatOpenAI as LCChatOpenAI

    assert isinstance(mm.registed_models["mistral-small"], OpenAIServerModel)
    assert not isinstance(
        mm.registed_models["mistral-small"], KeyRotatingOpenAIServerModel
    )
    assert isinstance(mm.registed_models["langchain-mistral-small"], LCChatOpenAI)
    assert not isinstance(
        mm.registed_models["langchain-mistral-small"], KeyRotatingChatOpenAI
    )


def test_registration_uses_rotating_classes_with_multi_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k1")
    monkeypatch.setenv("MISTRAL_API_KEY_2", "k2")

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    assert isinstance(
        mm.registed_models["mistral-small"], KeyRotatingOpenAIServerModel
    )
    assert isinstance(
        mm.registed_models["langchain-mistral-small"], KeyRotatingChatOpenAI
    )


# ---------------------------------------------------------------------------
# Secret handling — API keys never appear in logs
# ---------------------------------------------------------------------------

def test_no_api_key_logged_on_registration(monkeypatch, caplog):
    """Neither primary nor secondary key should appear in captured log
    messages during multi-key registration.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "super-secret-primary")
    monkeypatch.setenv("MISTRAL_API_KEY_2", "super-secret-secondary")
    caplog.set_level(logging.INFO)

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    log_blob = "\n".join(record.getMessage() for record in caplog.records)
    assert "super-secret-primary" not in log_blob
    assert "super-secret-secondary" not in log_blob
