#!/usr/bin/env bash
# One-file-at-a-time pytest sweep for handoff #9 / farm CI (avoids pytest
# multi-file collection issues with src.logger stub — see CLAUDE.md).
# Usage: bash scripts/run_handoff_pytest_sweep.sh
# Expect: 140 passed total across the listed modules (requires conda env `dra`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${_DRA_SWEEP_REEXEC:-}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  if ! python -c "import mmengine" 2>/dev/null; then
    if conda run -n dra python -c "import mmengine" 2>/dev/null; then
      export _DRA_SWEEP_REEXEC=1
      exec conda run -n dra --no-capture-output bash "$0" "$@"
    fi
  fi
fi

FILES=(
  tests/test_failover_model.py
  tests/test_reasoning_preservation.py
  tests/test_tier_b_tool_messages.py
  tests/test_process_tool_calls_guard.py
  tests/test_max_steps_yield_order.py
  tests/test_rc2_diagnostic_hook.py
  tests/test_tool_generator.py
  tests/test_review_schema.py
  tests/test_skill_registry.py
  tests/test_skill_seed.py
  tests/test_tool_choice_dispatch.py
)

for f in "${FILES[@]}"; do
  echo "=== $f ==="
  pytest "$f" -q --tb=line
done

echo "=== Handoff pytest sweep: all modules passed (140 tests expected) ==="
