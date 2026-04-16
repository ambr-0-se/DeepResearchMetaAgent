"""
Condition C0 — GAIA baseline (vanilla PlanningAgent, no meta-agent tools).

Thin alias over `config_gaia.py` that retags the output directory so C0
results are written to `workdir/gaia_c0/` alongside C2/C3/C4 for direct
side-by-side comparison.

No functional change from `config_gaia.py` — the underlying planner, tools,
and managed agents are identical.

Usage:
    python examples/run_gaia.py --config configs/config_gaia_c0.py

Compared against:
    - C2: configs/config_gaia_adaptive.py (+ reactive diagnose/modify tools)
    - C3: configs/config_gaia_c3.py       (+ structural REVIEW step)
    - C4: configs/config_gaia_c4.py       (+ skill library)
"""

_base_ = './config_gaia.py'

# Separate results directory so C0/C2/C3/C4 don't overwrite each other.
tag = "gaia_c0"
