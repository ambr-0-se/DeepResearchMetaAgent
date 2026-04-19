"""
Condition C4 — GAIA with AdaptivePlanningAgent + REVIEW step + skill library.

Extends C3 (configs/config_gaia_c3.py) with:
- A cross-task skill library (`src/skills/`) exposed via `ActivateSkillTool`
  on the planner and every managed sub-agent.
- The planner's C4 prompt template references `{{skill_registry_block}}` and
  documents how to use `activate_skill`.
- Sub-agent YAMLs (deep_analyzer, browser_use, deep_researcher) already
  support the conditional skill block via Jinja, so they receive
  agent-specific skills without any further change.
- `enable_skill_extraction=True` is set in THIS config to turn on the
  training-mode extractor. For frozen-library EVAL runs, create a second
  config that inherits from this one and overrides
  `enable_skill_extraction=False`.

Sealing guarantees (unchanged from C3):
- ReviewAgent is not in any registry and not modifiable.
- ActivateSkillTool reads from the SkillRegistry but cannot mutate it.
- SkillExtractor is held on the planner (not in managed_agents), so
  `modify_subagent` cannot disable it.

Usage:
    # Training mode (extracts new skills at task end)
    python examples/run_gaia.py --config configs/config_gaia_c4.py

    # To run a frozen-library evaluation, copy this file and set
    # `enable_skill_extraction=False`.
"""

_base_ = './config_gaia_c3.py'

# Per-run isolation: every C4 invocation writes its dra.jsonl AND its
# skill library to `workdir/gaia_c4_<RUN_ID>/`. This means re-running
# the same model never clobbers the library a previous run evolved, and
# parallel runs (e.g. via the matrix runner) can never race on the same
# skills directory. Set `DRA_RUN_ID` explicitly to resume a prior run.
import os as _os
from datetime import datetime as _datetime
_RUN_ID = _os.environ.get("DRA_RUN_ID") or _datetime.now().strftime("%Y%m%d_%H%M%S")
# mmengine's Config.pretty_text would otherwise emit the module/class values of
# these imports as invalid Python (e.g. `_datetime=<class ...>`), crashing yapf.
del _os, _datetime

tag = f"gaia_c4_{_RUN_ID}"

# Override the adaptive planning agent config for C4.
planning_agent_config = dict(
    type="adaptive_planning_agent",
    name="adaptive_planning_agent",
    model_id="claude-3.7-sonnet-thinking",
    description=(
        "An adaptive planning agent with reactive diagnose/modify tools, a "
        "structural REVIEW step, and a cross-task skill library "
        "(pre-seeded + learned). Condition C4."
    ),
    max_steps=25,
    template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c4.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],  # diagnose_subagent, modify_subagent, activate_skill added automatically
    managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"],
    # Flags read by AdaptivePlanningAgent.__init__
    enable_review=True,
    enable_skills=True,
    enable_skill_extraction=True,  # TRUE = C4 training; flip to False for frozen-library eval
    # Per-run skill library. Seeded from `src/skills/` on first construction
    # (see `src/skills/_seed.py`). A `.seeded` marker prevents re-seed on
    # resume so learned skills survive.
    skills_dir=f"workdir/{tag}/skills",
)

# Use the C4 planning agent config as the main agent
agent_config = planning_agent_config

# Per-question wall clock timeout (secs). Pinned 2026-04-20 for fairness
# with E0 v3 C4 training (which used 1800s). Previously inherited
# run_gaia.py default 1200, creating a training/test asymmetry that
# could bias the C0-C3 vs C4 ablation at test time.
per_question_timeout_secs = 1800
