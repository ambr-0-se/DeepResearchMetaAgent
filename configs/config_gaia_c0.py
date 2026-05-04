"""
Condition C0 — GAIA baseline (vanilla PlanningAgent, no meta-agent tools).

Thin alias over `config_gaia.py` that retags the output directory so C0
results are written to `workdir/gaia_c0/` alongside C1/C2/C3 for direct
side-by-side comparison.

No functional change from `config_gaia.py` — the underlying planner, tools,
and managed agents are identical.

Usage:
    python examples/run_gaia.py --config configs/config_gaia_c0.py

Compared against:
    - C1: configs/config_gaia_c1.py (+ reactive diagnose/modify tools)
    - C2: configs/config_gaia_c2.py (+ structural REVIEW step)
    - C3: configs/config_gaia_c3.py (+ skill library)
"""

_base_ = './config_gaia.py'

# Per-run output isolation: every invocation writes its dra.jsonl to
# `workdir/gaia_c0_<RUN_ID>/` so re-runs never clobber prior results.
# Set `DRA_RUN_ID` explicitly to resume a prior run.
import os as _os
from datetime import datetime as _datetime
_RUN_ID = _os.environ.get("DRA_RUN_ID") or _datetime.now().strftime("%Y%m%d_%H%M%S")
# mmengine's Config.pretty_text would otherwise emit the module/class values of
# these imports as invalid Python (e.g. `_datetime=<class ...>`), crashing yapf.
del _os, _datetime

tag = f"gaia_c0_{_RUN_ID}"

# Per-question wall clock timeout (secs). Pinned 2026-04-20 for fairness
# with E0 v3 C3 / skill-library training (which used 1800s; legacy workdir
# `gaia_c4_*`). Previously inherited run_gaia.py default 1200, creating a
# training/test asymmetry that could bias the C0–C3 ablation at test time.
per_question_timeout_secs = 1800
