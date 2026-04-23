import asyncio
import os
import subprocess
import atexit
import signal
from dotenv import load_dotenv
load_dotenv(verbose=True)

from browser_use import Agent

from src.tools import AsyncTool, ToolResult
from src.tools._browser_json_extractor import install_tolerant_extractor
from src.tools.browser import Controller
from src.utils import assemble_project_path
from src.registry import TOOL
from src.models import model_manager


# Per-internal-step budget for the browser-use Agent. Used to cap
# `browser_agent.run(max_steps=N)` at `N * _PER_STEP_TIMEOUT_S`. Needed
# because the LangChain `langchain-or-*` wrappers do NOT inherit the
# native-route `extra_body` (Kimi's reasoning-off pin, Gemma's provider
# pin) — observed on Kimi C0 during I2 2026-04-19: internal browser
# iterations stalled 200-650s on LangChain-routed Moonshot calls,
# consuming the full 1200s per-Q timeout despite `max_steps=3`. This cap
# surfaces a stuck iteration as a clean `TimeoutError` → tool reports
# "no extracted content" via the existing fallback path.
_PER_STEP_TIMEOUT_S = 60


# Wire-id prefixes that require `tool_calling_method='raw'` in browser_use.
# Re-exported from src/models/tool_choice.py:_AUTO_WIRE_PREFIXES so there
# is a SINGLE source of truth for "which provider family needs
# raw-mode browser path". Importing by alias (not copying the tuple)
# ensures a future addition to `_AUTO_WIRE_PREFIXES` (e.g., another
# provider family that rejects tool_choice="required") automatically
# picks up the same browser-side remediation without a second edit.
# See HANDOFF_QWEN_BROWSER_RAW_MODE.md code-review §HIGH-2.
from src.models.tool_choice import _AUTO_WIRE_PREFIXES as _RAW_MODE_WIRE_PREFIXES


def _resolve_wire_id(model) -> str:
    """Return the wire id for a LangChain-compatible model wrapper.

    Tries `model_id` first (the attribute our native `OpenAIServerModel`
    sets), falls back to `model_name` (what `ChatOpenAI` exposes), and
    finally to an empty string. `KeyRotatingChatOpenAI` delegates
    attribute lookup via `__getattr__` to instance[0], so either
    attribute works through that wrapper as well.
    """
    wire_id = getattr(model, "model_id", None)
    if not wire_id:
        wire_id = getattr(model, "model_name", None)
    return wire_id or ""


def _unwrap_for_browser_use(model):
    """Unwrap `KeyRotatingChatOpenAI` to its first `ChatOpenAI` instance
    before passing to `browser_use.Agent`.

    Why: `browser_use.AgentSettings.page_extraction_llm` is typed as
    `BaseChatModel | None`. Pydantic v2 strict-mode validation rejects
    `KeyRotatingChatOpenAI` because it's a composite object — NOT a
    `BaseChatModel` subclass — regardless of attribute delegation.
    Before this unwrap, Mistral's T3v2 smoke logged 35 ×
    ``'ChatOpenAI' object has no attribute 'get'`` silently killing
    every browser_use step (cf. `workdir/gaia_c4_mistral_20260422_T3v2smoke/log.txt`).

    Trade-off: browser_use loses key rotation — it's locked to
    instance[0] for the session. Acceptable because browser_use LLM
    calls are a small fraction of total Mistral traffic (the planner
    and sub-agents still go through the rotating wrapper via the
    native `OpenAIServerModel` path).

    Non-rotating wrappers (plain `ChatOpenAI`, P5
    `ToolChoiceDowngradingChatOpenAI._Impl`) pass through untouched.
    """
    instances = getattr(model, "_instances", None)
    if isinstance(instances, list) and instances:
        return instances[0]
    return model


def _pick_browser_tool_calling_method(wire_id: str) -> str | None:
    """Choose the `browser_use.Agent(tool_calling_method=...)` kwarg.

    Returns ``'raw'`` for provider families that break browser_use's
    default `function_calling` transport — currently Qwen via
    OpenRouter+Alibaba (verified 2026-04-23 live probe):
      - `tool_choice="required"` → HTTP 404 (P5 downgrades this to "auto")
      - `tool_choice="auto"` + `method='function_calling'` → Qwen fills
        every optional field of `AgentOutput.action`, `done` fires first,
        agent terminates at Step 1 with 0 chars.

    Raw mode calls `llm.invoke(messages)` directly (no `bind_tools`, no
    `with_structured_output`) so none of the above applies. The
    tolerant JSON extractor installed by `install_tolerant_extractor()`
    covers Qwen's free-form JSON quirks.

    Non-Qwen models return ``None`` so browser_use's existing auto
    detection (`function_calling` for ChatOpenAI) is preserved
    byte-for-byte (R6 fairness guardrail).
    """
    if wire_id and wire_id.startswith(_RAW_MODE_WIRE_PREFIXES):
        return "raw"
    return None

@TOOL.register_module(name="auto_browser_use_tool", force=True)
class AutoBrowserUseTool(AsyncTool):
    name = "auto_browser_use_tool"
    description = "A powerful browser automation tool that allows interaction with web pages through various actions. Automatically browse the web and extract information based on a given task."
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task to perform"
            },
        },
        "required": ["task"],
    }
    output_type = "any"

    def __init__(self,
                 model_id: str = "gpt-4.1",
                 max_steps: int = 50,
                 ):
        """
        Args:
            model_id: LangChain-compatible model alias the browser-use Agent
                should drive. Anything without a ``langchain-`` prefix is
                auto-prefixed.
            max_steps: Hard ceiling on the browser-use library's inner
                planning loop per single tool invocation. The default of 50
                matches browser-use's own default and suits production
                research tasks. For smoke / handoff validation runs, drop
                this to 8–15 via ``--cfg-options
                auto_browser_use_tool_config.max_steps=10`` so the agent
                exits the inner loop quickly, producing more outer-loop
                delegations (and therefore more REVIEW / diagnose / skill
                evidence) within the same wall-clock budget.
        """

        super(AutoBrowserUseTool, self).__init__()

        self.model_id = model_id
        self.max_steps = int(max_steps)
        self.http_server_path = assemble_project_path("src/tools/browser/http_server")
        self.http_save_path = assemble_project_path("src/tools/browser/http_server/local")
        os.makedirs(self.http_save_path, exist_ok=True)

        self._init_pdf_server()

    def _init_pdf_server(self):

        server_proc = subprocess.Popen(
            ["python3", "-m", "http.server", "8080"],
            cwd=self.http_server_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=None
        )

        @atexit.register
        def shutdown_server():
            print("Shutting down http.server...")
            try:
                server_proc.send_signal(signal.SIGTERM)
                server_proc.wait(timeout=5)
            except Exception as e:
                print("Force killing server...")
                server_proc.kill()

    async def _browser_task(self, task):
        controller = Controller(http_save_path=self.http_save_path)

        if "langchain" not in self.model_id:
            model_id = f"langchain-{self.model_id}"
        else:
            model_id = self.model_id

        if model_id not in model_manager.registed_models:
            raise ValueError(
                f"Model '{model_id}' is not registered. "
                f"Ensure a LangChain-compatible wrapper is registered under this key."
            )
        model = model_manager.registed_models[model_id]

        # Qwen-specific remediation. Two layers applied when the wrapped
        # LangChain model resolves to a wire id under `_RAW_MODE_WIRE_PREFIXES`:
        #   (L1) `tool_calling_method='raw'` — sidesteps browser_use's
        #        `with_structured_output` path which, post-P5 `tool_choice="auto"`
        #        downgrade, causes Qwen to emit every optional field of
        #        `AgentOutput.action` simultaneously (`done` fires first).
        #   (L2) Install the tolerant JSON extractor (idempotent, process-
        #        wide) — raw-mode output is structurally correct but
        #        wrapped inconsistently ("json\n{...}\n```" missing the
        #        opening fence), which the upstream parser can't handle.
        # Non-Qwen models take the default path (tool_calling_method=None
        # → browser_use auto-detects `'function_calling'`) and skip the
        # extractor patch install — Mistral/Kimi/future browsers are
        # byte-identical with pre-fix behaviour. See
        # docs/handoffs/HANDOFF_QWEN_BROWSER_RAW_MODE.md for root-cause
        # analysis + live-probe evidence.
        wire_id = _resolve_wire_id(model)
        tool_calling_method = _pick_browser_tool_calling_method(wire_id)
        if tool_calling_method == "raw":
            install_tolerant_extractor()

        # Unwrap KeyRotatingChatOpenAI → instance[0] for browser_use
        # compatibility (see `_unwrap_for_browser_use` docstring). This
        # is a structural requirement of browser_use's Pydantic
        # validation, unrelated to the Qwen fix.
        bu_model = _unwrap_for_browser_use(model)

        agent_kwargs = dict(
            task=task,
            llm=bu_model,
            enable_memory=False,
            controller=controller,
            page_extraction_llm=bu_model,
        )
        if tool_calling_method is not None:
            agent_kwargs["tool_calling_method"] = tool_calling_method
        browser_agent = Agent(**agent_kwargs)

        # Cap the browser_agent run so a silent LangChain-wrapper LLM hang
        # can't wedge the whole planner step beyond `max_steps * _PER_STEP_TIMEOUT_S`.
        # Uniform across all models; Mistral/Qwen/Gemma iterations normally
        # finish in seconds so this is a no-op for them.
        #
        # CRITICAL: the try/finally around browser_agent.close() is what
        # prevents the cleanup-deadlock observed on Kimi during E0 2026-04-19.
        # When the OUTER per-question 1200s timeout in run_gaia.py cancels
        # agent.run() while this coroutine is inside browser_agent.run(),
        # asyncio raises CancelledError — Playwright driver subprocess and
        # chromium children DO NOT auto-close on cancel. The parent process
        # then deadlocks at ~0% CPU waiting for the Playwright event loop to
        # drain (observed: 58-min wedge on Kimi PID 28788 after a per-Q
        # timeout on a browser question). Force-closing via
        # `browser_agent.close()` in a finally block drops the browser
        # resources synchronously so the async loop can resume and proceed
        # to the next question.
        overall_budget = max(1, self.max_steps) * _PER_STEP_TIMEOUT_S
        history = None
        try:
            try:
                history = await asyncio.wait_for(
                    browser_agent.run(max_steps=self.max_steps),
                    timeout=overall_budget,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"auto_browser_use_tool timed out after {overall_budget}s "
                    f"(max_steps={self.max_steps}, per-step cap {_PER_STEP_TIMEOUT_S}s). "
                    "Likely a hung LLM call inside the browser-use agent — the "
                    "LangChain wrappers (`langchain-or-*`) do not inherit native "
                    "provider-pin `extra_body`, which can stall on some OR sub-providers. "
                    "Recommend switching to deep_researcher_agent or refining the task with specific URLs."
                )
        finally:
            # Always close. `browser_agent.close()` is idempotent-safe: it
            # silently skips injected contexts and no-ops if browser is
            # already closed. Bound the close itself in case Playwright
            # itself hangs during teardown (extra defense).
            try:
                await asyncio.wait_for(browser_agent.close(), timeout=15)
            except Exception:
                pass
        contents = history.extracted_content()
        joined = "\n".join(c for c in contents if c)

        if not joined.strip():
            visited = getattr(history, "urls", lambda: [])() or []
            internal_steps = len(history.history) if hasattr(history, "history") else "N"
            raise RuntimeError(
                "auto_browser_use_tool returned no extracted content "
                f"after {internal_steps} internal steps. "
                f"Visited URLs: {visited[:5]}. "
                "Likely causes: page required login/CAPTCHA, JS-heavy content didn't render, "
                "or the target information isn't on the pages visited. "
                "Recommend switching to deep_researcher_agent or refining the task with specific URLs."
            )
        return joined

    async def forward(self, task: str) -> ToolResult:
        """
        Automatically browse the web and extract information based on a given task.

        Args:
            task: The task to perform
        Returns:
            ToolResult with the task result
        """
        try:
            result = await self._browser_task(task)
            return ToolResult(output=result, error=None)
        except Exception as e:
            return ToolResult(output=None, error=str(e))