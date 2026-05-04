"""Generator for the 16 GAIA-eval config files (4 models × 4 conditions).

Run this script once to (re)write all 16 config files under `configs/`. The
generated files inherit from the matching C-condition base (`config_gaia_c0.py`,
`config_gaia_c1.py`, `config_gaia_c2.py`, `config_gaia_c3.py`) and only
override:
  - `tag` (drives `workdir/gaia_<tag>/` output dir)
  - `model_id` on every agent + tool config
  - LangChain alias on `auto_browser_use_tool` (browser-use needs ChatModel)
  - `max_samples` (cap; commented sentinel — uncomment / override for smoke runs)
  - `concurrency` (per-config question parallelism)

Why generated rather than hand-written: the override surface is wide (4 agents
× 1-2 fields, plus 3 tools × 1-2 fields), each model needs all of them, and
mmengine config inheritance replaces dicts wholesale rather than deep-merging.
A single template here is the maintainable shape.

Usage:
    python scripts/gen_eval_configs.py            # write all 16 (4 models × 4 conditions)
    python scripts/gen_eval_configs.py --dry-run  # print to stdout instead
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

# (model_label, model_id_for_agents, langchain_alias, comment_about_thinking,
#  concurrency_default)
#
# `concurrency_default` is the per-question parallelism baked into each
# generated config. Historically 4 across the board. 2026-04-22: bumped Qwen
# to 8 after E0 v3 evidence showed 0 × 429 / Retry-After / RateLimitError
# in 3 006 calls at c=4 — headroom was left on the table (§P3 of
# docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md). Others remain at 4.
MODELS = [
    (
        "mistral",
        "mistral-small",
        "langchain-mistral-small",
        "Mistral Small 4 (mistral-small-2603) — native multimodal, no thinking mode.",
        4,
    ),
    (
        "kimi",
        "or-kimi-k2.5",
        "langchain-or-kimi-k2.5",
        "Kimi K2.5 via OpenRouter (operator preference — OPENROUTER_API_KEY). "
        "Native MOONSHOT_API_KEY path is intentionally left as a placeholder. "
        "OpenRouter does not enforce Moonshot's thinking-on default, so this "
        "also satisfies the C2/C3 JSON-output requirement without an extra_body "
        "override.",
        4,
    ),
    (
        "qwen",
        "or-qwen3.6-plus",
        "langchain-or-qwen3.6-plus",
        "Qwen3.6-Plus via OpenRouter (operator preference — D1, 2026-04-18). "
        "Chosen over Qwen3-Next for its vision modality (text+image+video per "
        "OR metadata) and 1M context. OR providers for the whole Qwen family "
        "reject `tool_choice=\"required\"` (404/400 depending on backend); "
        "this is handled transparently by the hybrid tool_choice dispatch in "
        "`src/models/tool_choice.py` (D3) — the `qwen/` wire-id prefix "
        "downgrades to `\"auto\"` and the retry guard in GeneralAgent / "
        "ToolCallingAgent re-prompts plain-text responses back into tool "
        "calls. $0.325 in / $1.95 out per M tokens. concurrency=8 per §P3: "
        "E0 v3 showed 0 × 429 in 3 006 calls at c=4; doubling consumes "
        "measured OR→Alibaba headroom.",
        8,
    ),
    (
        "gemma",
        "or-gemma-4-31b-it",
        "langchain-or-gemma-4-31b-it",
        "Gemma 4 31B Instruct via OpenRouter (D4, 2026-04-18). Dense Google "
        "frontier slot alongside the MoE variants (Mistral, Kimi, Qwen). "
        "Text+image+video modalities (no audio). Apache 2.0. Provider pin to "
        "DeepInfra+Together (both vLLM-backed, latest gemma4 parser) is set "
        "at registration in `src/models/models.py`; reasoning mode disabled "
        "there to prevent thinking-channel contamination of tool output. "
        "Live smoke probe 2026-04-18 confirmed `tool_choice=\"required\"` "
        "works directly with this provider pin (finish_reason=\"tool_calls\", "
        "no special-token leaks), so Gemma is NOT in "
        "`MODELS_REJECTING_REQUIRED` — the dispatcher passes `\"required\"` "
        "through unchanged. Concurrency is capped at 4 in "
        "`scripts/run_eval_matrix.sh` for this stream only (vLLM #39392 "
        "pad-bug under parallel load). $0.13 in / $0.38 out per M tokens.",
        4,
    ),
]

# (condition_label, base_config_relative, tag_suffix, max_steps_for_this_condition)
CONDITIONS = [
    ("c0", "./config_gaia_c0.py", "c0", None),
    ("c1", "./config_gaia_c1.py", "c1", None),
    ("c2", "./config_gaia_c2.py", "c2", None),
    ("c3", "./config_gaia_c3.py", "c3", None),
]

# C0 uses planning_agent (non-adaptive); C1/C2/C3 use adaptive_planning_agent.
# We re-declare the planning_agent_config in each file to swap model_id without
# losing the condition's enable_review / enable_skills / max_steps settings.
PLANNING_TEMPLATES: dict[str, str] = {
    "c0": dedent(
        '''\
        planning_agent_config = dict(
            type="planning_agent",
            name="planning_agent",
            model_id={model_id!r},
            description="A planning agent that can plan the steps to complete the task.",
            max_steps=20,
            template_path="src/agent/planning_agent/prompts/planning_agent.yaml",
            provide_run_summary=True,
            tools=["planning_tool"],
            managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"],
        )
        '''
    ),
    "c1": dedent(
        '''\
        planning_agent_config = dict(
            type="adaptive_planning_agent",
            name="adaptive_planning_agent",
            model_id={model_id!r},
            description="An adaptive planning agent that can diagnose and modify sub-agents at runtime.",
            max_steps=25,
            template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml",
            provide_run_summary=True,
            tools=["planning_tool"],
            managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"],
        )
        '''
    ),
    "c2": dedent(
        '''\
        planning_agent_config = dict(
            type="adaptive_planning_agent",
            name="adaptive_planning_agent",
            model_id={model_id!r},
            description=(
                "An adaptive planning agent with reactive diagnose/modify tools "
                "plus a structural REVIEW step (condition C2)."
            ),
            max_steps=25,
            template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c2.yaml",
            provide_run_summary=True,
            tools=["planning_tool"],
            managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"],
            enable_review=True,
        )
        '''
    ),
    "c3": dedent(
        '''\
        planning_agent_config = dict(
            type="adaptive_planning_agent",
            name="adaptive_planning_agent",
            model_id={model_id!r},
            description=(
                "An adaptive planning agent with reactive diagnose/modify tools, a "
                "structural REVIEW step, and a cross-task skill library (C3)."
            ),
            max_steps=25,
            template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c3.yaml",
            provide_run_summary=True,
            tools=["planning_tool"],
            managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"],
            enable_review=True,
            enable_skills=True,
            enable_skill_extraction=True,
            skills_dir=f"workdir/{{tag}}/skills",
        )
        '''
    ),
}


CONFIG_TEMPLATE = '''\
"""GAIA evaluation — {model_label_upper} × {condition_upper}.

Generated by scripts/gen_eval_configs.py. Do not edit by hand — regenerate
instead. Inherits from {base_config} and overrides every model_id to point
at the chosen model.

Model: {model_id}
{model_comment}

Output dir: workdir/{tag_dir_comment}/

Initial-run knobs (override via --cfg-options):
  - max_samples=N        # cap to first N questions; leave None for full set
  - concurrency=K        # parallel questions per config (default 4)
  - dataset.split="validation"  # smoke-test against the labeled validation
                                # split before submitting on test
"""

_base_ = {base_config!r}
{run_id_prelude}
tag = {tag_expr}

# ---- model overrides --------------------------------------------------------

# All four agents use the SAME model_id under the single-model evaluation
# constraint. C2/C3 reviewer + skill extractor inherit this id via the planner.

deep_researcher_agent_config = dict(
    type="deep_researcher_agent",
    name="deep_researcher_agent",
    model_id={model_id!r},
    description="A deep researcher agent that can conduct extensive web searches.",
    max_steps=3,
    template_path="src/agent/deep_researcher_agent/prompts/deep_researcher_agent.yaml",
    provide_run_summary=True,
    tools=["deep_researcher_tool", "python_interpreter_tool"],
)

deep_analyzer_agent_config = dict(
    type="deep_analyzer_agent",
    name="deep_analyzer_agent",
    model_id={model_id!r},
    description="A deep analyzer agent that can perform systematic, step-by-step analysis.",
    max_steps=3,
    template_path="src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml",
    provide_run_summary=True,
    tools=["deep_analyzer_tool", "python_interpreter_tool"],
)

browser_use_agent_config = dict(
    type="browser_use_agent",
    name="browser_use_agent",
    model_id={model_id!r},
    description="A browser use agent that can search relevant web pages and interact with them.",
    max_steps=5,
    template_path="src/agent/browser_use_agent/prompts/browser_use_agent.yaml",
    provide_run_summary=True,
    tools=["auto_browser_use_tool", "python_interpreter_tool"],
)

{planning_block}
agent_config = planning_agent_config

# ---- tool overrides ---------------------------------------------------------

deep_researcher_tool_config = dict(
    type="deep_researcher_tool",
    model_id={model_id!r},
    max_depth=2,
    max_insights=20,
    time_limit_seconds=60,
    max_follow_ups=3,
)

deep_analyzer_tool_config = dict(
    type="deep_analyzer_tool",
    analyzer_model_ids=[{model_id!r}],
    summarizer_model_id={model_id!r},
)

# auto_browser_use_tool requires a LangChain ChatModel wrapper, registered
# alongside each native model in src/models/models.py.
#
# max_steps=15 is the S4-time browser step cap (down from the
# AutoBrowserUseTool class default of 50). Rationale: typical GAIA browser
# flows need 2-12 internal steps; beyond ~15 the marginal accuracy gain is
# dominated by stuck-loop waste (CAPTCHA retries, cookie-modal fights,
# infinite scrolls). A single stuck 50-step invocation burns ~$0.10-1.00
# and 8-12 min wall. Applied uniformly across all 16 cells so
# C0/C1/C2/C3 deltas aren't contaminated by per-condition browser budget
# differences. For local smoke tests, override via --cfg-options
# (auto_browser_use_tool_config.max_steps=8). See
# docs/handoffs/HANDOFF_TEST_EVAL.md "Browser step cap policy".
auto_browser_use_tool_config = dict(
    type="auto_browser_use_tool",
    model_id={langchain_alias!r},
    max_steps=15,
)

# ---- run knobs --------------------------------------------------------------

# Default: full GAIA test split. Override on the CLI for smoke tests:
#   python examples/run_gaia.py --config {self_path} \\
#     --cfg-options max_samples=10 dataset.split=validation
max_samples = None
concurrency = {concurrency}

# Per-question wall clock timeout (secs). Pinned 2026-04-20 after the
# E0 v3 resume surfaced an asymmetry: training had been running at 1800s
# but test-time configs silently fell back to the run_gaia.py default of
# 1200s, biasing the C0-C2 vs C3 ablation at test time. Now uniformly
# 1800s across training (E0) and test (E3) for every (model, condition).
per_question_timeout_secs = 1800
'''


#: Prelude block emitted for ALL matrix configs. Reads DRA_RUN_ID from the
#: environment with a fresh-timestamp fallback, so every invocation lands
#: in its own isolated output directory unless the matrix runner (or the
#: operator) explicitly reuses a run id. For C3, the same run id also
#: drives the per-run `skills_dir` so extracted skills stay co-located
#: with the dra.jsonl they produced.
_RUN_ID_PRELUDE = (
    "\n"
    "import os as _os\n"
    "from datetime import datetime as _datetime\n"
    "_RUN_ID = _os.environ.get(\"DRA_RUN_ID\") or "
    "_datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n"
    # mmengine's Config.pretty_text walks every module-level name and emits it\n"
    # as a Python assignment via yapf. Class/module values (e.g.\n"
    # `_datetime=<class 'datetime.datetime'>`) are not valid Python and make\n"
    # yapf crash. Drop the imports now that _RUN_ID is captured.\n"
    "del _os, _datetime\n"
)


def render_config(model_label: str, model_id: str, langchain_alias: str,
                  model_comment: str, condition: str, base_config: str,
                  concurrency: int) -> str:
    tag_prefix = f"{condition}_{model_label}"
    self_path = f"configs/config_gaia_{condition}_{model_label}.py"
    planning_block = PLANNING_TEMPLATES[condition].format(model_id=model_id)

    # Every condition now carries a DRA_RUN_ID so repeat runs of the same
    # (model, condition) never collide — each run's dra.jsonl (and, for C3,
    # skills/) is inspectable forever after.
    run_id_prelude = _RUN_ID_PRELUDE
    tag_expr = f'f"gaia_{tag_prefix}_{{_RUN_ID}}"'
    tag_dir_comment = f"gaia_{tag_prefix}_<run_id>"

    return CONFIG_TEMPLATE.format(
        model_label_upper=model_label.upper(),
        condition_upper=condition.upper(),
        base_config=base_config,
        model_id=model_id,
        model_comment=model_comment,
        run_id_prelude=run_id_prelude,
        tag_expr=tag_expr,
        tag_dir_comment=tag_dir_comment,
        planning_block=planning_block.rstrip("\n"),
        langchain_alias=langchain_alias,
        self_path=self_path,
        concurrency=concurrency,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout instead of writing files")
    args = parser.parse_args()

    written = []
    for model_label, model_id, langchain_alias, model_comment, concurrency in MODELS:
        for condition, base_config, _tag_suffix, _max_steps in CONDITIONS:
            text = render_config(
                model_label=model_label,
                model_id=model_id,
                langchain_alias=langchain_alias,
                model_comment=model_comment,
                condition=condition,
                base_config=base_config,
                concurrency=concurrency,
            )
            out_path = CONFIGS_DIR / f"config_gaia_{condition}_{model_label}.py"
            if args.dry_run:
                print(f"# ===== {out_path.name} =====")
                print(text)
            else:
                out_path.write_text(text)
                written.append(out_path.name)

    if not args.dry_run:
        print(f"Wrote {len(written)} configs:")
        for name in written:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
