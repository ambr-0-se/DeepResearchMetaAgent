#!/usr/bin/env bash
# Handoff evidence collector. Walks workdir/gaia_c*_<model>_<run_id>/ for
# the most recent run id (or one passed as $1) and emits a PASS/FAIL
# summary against each handoff's documented validation criteria.
#
# Usage:
#   bash scripts/validate_handoffs.sh              # use latest DRA_RUN_ID
#   bash scripts/validate_handoffs.sh phc_2026...  # use a specific one

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" != "" ]]; then
  RUN="$1"
else
  RUN="$(cat workdir/run_logs/phase_c_runid.txt 2>/dev/null | cut -d= -f2)"
fi

if [[ -z "$RUN" ]]; then
  echo "No run id found. Provide one as first arg."
  exit 1
fi

echo "=== Handoff Validation Report ==="
echo "Run id: $RUN"
echo

# Helper: count matches across all cell logs for a given model(s).
# grep -c returns 1 when count == 0, so use `wc -l` against matches to
# guarantee a clean integer without tripping $(( )) arithmetic.
count_in() {
  local pattern="$1"; shift
  local total=0
  for cell in "$@"; do
    local logf="workdir/${cell}/log.txt"
    [[ -f "$logf" ]] || continue
    local n
    n=$(grep -E -- "$pattern" "$logf" 2>/dev/null | wc -l | tr -d ' ')
    [[ -z "$n" ]] && n=0
    total=$((total + n))
  done
  echo "$total"
}

CELLS_C0="gaia_c0_mistral_${RUN} gaia_c0_kimi_${RUN} gaia_c0_qwen_${RUN}"
CELLS_C2="gaia_c2_mistral_${RUN} gaia_c2_kimi_${RUN} gaia_c2_qwen_${RUN}"
CELLS_C3="gaia_c3_mistral_${RUN} gaia_c3_kimi_${RUN} gaia_c3_qwen_${RUN}"
CELLS_C4="gaia_c4_mistral_${RUN} gaia_c4_kimi_${RUN} gaia_c4_qwen_${RUN}"
CELLS_ALL="$CELLS_C0 $CELLS_C2 $CELLS_C3 $CELLS_C4"

echo "## Handoff #1 — Silent-failure fixes (browser + analyzer)"
echo "   browser 'no extracted content' count (expect 0 or low):"
echo "      $(count_in "auto_browser_use_tool returned no extracted content" $CELLS_ALL)"
echo "   analyzer 'code.txt' FileNotFoundError (expect 0):"
echo "      $(count_in "No such file or directory: 'code.txt'" $CELLS_ALL)"
echo

echo "## Handoff #2 — Provider matrix"
echo "   Kimi caller-sampling 400 errors (expect 0):"
echo "      $(count_in "400.*temperature|400.*top_p|400.*Invalid sampling" $CELLS_ALL)"
echo "   Qwen thinking-mode 400 errors (expect 0 after fix):"
echo "      $(count_in "tool_choice.*thinking mode|<400>.*InvalidParameter.*thinking" $CELLS_ALL)"
echo "   Qwen failover trigger count (stream log; >=1 if DashScope free tier hit):"
# The WARNING is emitted to stderr by FailoverModel — goes to run_logs/phc_<model>.log, not per-cell log.txt.
qwen_stream_log="workdir/run_logs/phc_qwen.log"
if [[ -f "$qwen_stream_log" ]]; then
  n=$(grep -E -- "\[FailoverModel:qwen3.6-plus-failover\] primary.*quota exhausted" "$qwen_stream_log" 2>/dev/null | wc -l | tr -d ' ')
  [[ -z "$n" ]] && n=0
  echo "      $n"
else
  echo "      (no stream log)"
fi
echo "   Qwen free tier exhaustion detection count:"
echo "      $(count_in "AllocationQuota.FreeTierOnly|The free tier of the model has been exhausted" $CELLS_ALL)"
echo

echo "## Handoff #3 — modify_subagent guidance"
echo "   Adaptive-planner tool exposure (C2+, should be >0):"
echo "      $(count_in "modify_subagent|diagnose_subagent" $CELLS_C2 $CELLS_C3 $CELLS_C4)"
echo

echo "## Handoff #4 — C3 REVIEW + C4 skills"
echo "   C3: enable_review=True wiring fired (expect 3 — one per model):"
echo "      $(count_in "\[AdaptivePlanningAgent\] enable_review=True; building ReviewStep" $CELLS_C3)"
echo "   C4: enable_skills=True + SkillRegistry built (expect 3):"
echo "      $(count_in "\[AdaptivePlanningAgent\] enable_skills=True; building SkillRegistry" $CELLS_C4)"
echo "   C4: seed_skills_dir seeded at least one skill per cell (expect >=3):"
echo "      $(count_in "\[seed_skills_dir\] seeded" $CELLS_C4)"
echo "   [REVIEW] markers in C3/C4 (non-zero if agent delegated at all):"
echo "      C3: $(count_in "\[REVIEW\]" $CELLS_C3)"
echo "      C4: $(count_in "\[REVIEW\]" $CELLS_C4)"
echo

echo "## Handoff #5 — RC1/RC2 guards"
echo "   RC1 premature-final-answer guard fires (informational):"
echo "      $(count_in "premature-final-answer guard|duplicate-final-answer guard" $CELLS_ALL)"
echo "   RC2 scope-error diagnostic fires (informational):"
echo "      $(count_in "\[RC2 diagnostic\] Scope error" $CELLS_ALL)"
echo

echo "## Handoff #7 — ToolGenerator (unit tests already passed)"
echo "   Dynamic tool generation attempts (informational):"
echo "      $(count_in "ToolGenerator.*generate|add_new_tool_to_agent" $CELLS_ALL)"
echo

echo "## Per-cell answer summary"
for cell in $CELLS_ALL; do
  f="workdir/${cell}/dra.jsonl"
  if [[ -f "$f" ]]; then
    python3 -c "
import json
try:
  with open('$f') as fh:
    for line in fh:
      d=json.loads(line)
      pred=(d.get('prediction') or d.get('output') or '')[:40]
      err=(d.get('agent_error') or '')[:60]
      truth=(d.get('true_answer') or '')[:30]
      print(f'  {\"$cell\":<50} pred={pred!r} truth={truth!r}  err={err!r}')
except Exception as e:
  print(f'  $cell: (parse err: {e})')
" 2>/dev/null
  else
    echo "  $cell: (no dra.jsonl)"
  fi
done

echo
echo "=== End Report ==="
