#!/usr/bin/env bash
# Integration step I3 — C4 end-to-end smoke (ONE model, default mistral):
#   I3a subset train on validation (extraction ON) → I3b snapshot → I3c short
#   validation eval with frozen library (extraction OFF via agent_config.*).
#
# Does NOT replace I2 (16-cell matrix smoke). Does NOT write E-track dirs under
# workdir/c4_trained_libraries/ — artifacts stay under workdir/c4_i3_<I3_RUN_ID>/.
#
# Env:
#   I3_MODEL          — mistral | kimi | qwen | gemma (default: mistral). No multi-model loop.
#   I3_RUN_ID         — optional stable id prefix (default: i3_<timestamp>_$$)
#   I3_TRAIN_SAMPLES  — max_samples for I3a (default: 5)
#   I3_EVAL_SAMPLES   — max_samples for I3c (default: 2)
#   SMOKE_CFG_OPTIONS — if unset, same step caps as scripts/run_eval_matrix.sh smoke.
#                       Set to empty string to omit caps.
#   PYTHON            — interpreter (default: python)
#
# Pass criteria (manual grep on I3c log under workdir/gaia_c4_<model>_<FREEZE_DRA_ID>/):
#   - grep -c "SkillExtractor active" log.txt  → expect 0
#   - SkillRegistry banner should reference the staging skills_dir path
#
# Not covered by scripts/validate_handoffs.sh (16-cell matrix ids only).
set -euo pipefail

if [[ "${_DRA_I3_REEXEC:-}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # Probe both mmengine AND crawl4ai (a bare base env may lack only one).
  # crawl4ai is the dra-only canary per CLAUDE.md; mmengine alone is
  # insufficient on machines where base conda has it pre-installed.
  if ! python -c "import mmengine, crawl4ai" 2>/dev/null; then
    if conda run -n dra python -c "import mmengine, crawl4ai" 2>/dev/null; then
      export _DRA_I3_REEXEC=1
      exec conda run -n dra --no-capture-output bash "$0" "$@"
    fi
  fi
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
I3_MODEL="${I3_MODEL:-mistral}"
I3_RUN_ID="${I3_RUN_ID:-i3_$(date +%Y%m%d_%H%M%S)_$$}"
I3_TRAIN_SAMPLES="${I3_TRAIN_SAMPLES:-5}"
I3_EVAL_SAMPLES="${I3_EVAL_SAMPLES:-2}"

if [ -z "${SMOKE_CFG_OPTIONS+x}" ]; then
  # Match scripts/run_eval_matrix.sh smoke defaults — aggressive I-track caps
  # that verify plumbing without paying for browser CAPTCHA loops.
  SMOKE_CFG_OPTIONS="agent_config.max_steps=6 auto_browser_use_tool_config.max_steps=4 deep_analyzer_agent_config.max_steps=2 deep_researcher_agent_config.max_steps=2 browser_use_agent_config.max_steps=2 deep_researcher_tool_config.time_limit_seconds=20"
fi

case "$I3_MODEL" in
  mistral|kimi|qwen|gemma) ;;
  *)
    echo "I3_MODEL must be one of: mistral, kimi, qwen, gemma (got: $I3_MODEL)" >&2
    exit 1
    ;;
esac

CFG="configs/config_gaia_c4_${I3_MODEL}.py"
if [[ ! -f "$CFG" ]]; then
  echo "Missing $CFG" >&2
  exit 1
fi

mkdir -p "workdir/c4_i3_${I3_RUN_ID}"
LOG="workdir/c4_i3_${I3_RUN_ID}/i3_pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "I3 C4 pipeline integration — model=$I3_MODEL  I3_RUN_ID=$I3_RUN_ID"
echo "ROOT=$ROOT"
echo "========================================"

# --- I3a: subset train (validation, extraction ON in config) ---
TRAIN_DRA_ID="i3train_${I3_MODEL}_${I3_RUN_ID}"
export DRA_RUN_ID="$TRAIN_DRA_ID"
echo ""
echo "=== I3a — subset train  DRA_RUN_ID=$DRA_RUN_ID  max_samples=$I3_TRAIN_SAMPLES ==="

TRAIN_OPTS=(dataset.split=validation "max_samples=$I3_TRAIN_SAMPLES")
if [[ -n "${SMOKE_CFG_OPTIONS}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_OPTS+=(${SMOKE_CFG_OPTIONS})
fi
if [[ "$I3_MODEL" == "gemma" ]]; then
  TRAIN_OPTS+=("concurrency=${GEMMA_CONCURRENCY:-4}")
fi

"$PYTHON" examples/run_gaia.py --config "$CFG" --cfg-options "${TRAIN_OPTS[@]}"

TRAIN_SKILLS="workdir/gaia_c4_${I3_MODEL}_${TRAIN_DRA_ID}/skills"
if [[ ! -d "$TRAIN_SKILLS" ]]; then
  echo "I3a failed: missing skills dir $TRAIN_SKILLS" >&2
  exit 1
fi

# Skill-count summary — distinguishes "extractor ran and emitted N learned
# skills" from "seed-only library, plumbing smoke only". Learned skills are
# the ones without `source: seeded` in their frontmatter. With a small
# I3_TRAIN_SAMPLES (default 5), zero learned is plausible and not an error —
# I3c still exercises the frozen-library reload path — but the operator
# should know before reading I3c results.
SKILL_TOTAL=$(find "$TRAIN_SKILLS" -mindepth 2 -name SKILL.md -type f 2>/dev/null | wc -l | tr -d ' ')
# `source:` lives under the `metadata:` block (indented). Match leading
# whitespace so indented YAML is counted correctly.
SKILL_SEEDED=$(grep -rlE --include=SKILL.md '^[[:space:]]+source:[[:space:]]+seeded[[:space:]]*$' "$TRAIN_SKILLS" 2>/dev/null | wc -l | tr -d ' ')
SKILL_LEARNED=$(( SKILL_TOTAL - SKILL_SEEDED ))
echo ""
echo "I3a skill count: ${SKILL_TOTAL} total (${SKILL_SEEDED} seeded, ${SKILL_LEARNED} learned)"
if [[ "$SKILL_LEARNED" -eq 0 ]]; then
  echo "NOTE: extractor emitted 0 learned skills on the ${I3_TRAIN_SAMPLES}-sample subset."
  echo "      I3c will freeze-eval against a seed-only library — valid smoke of the"
  echo "      frozen-reload path, but does not exercise learned-skill activation."
  echo "      Raise I3_TRAIN_SAMPLES if I3 needs to cover the learned path too."
fi

# --- I3b: snapshot into isolated staging (never touches c4_trained_libraries/) ---
STAGING="workdir/c4_i3_${I3_RUN_ID}/${I3_MODEL}_skills"
echo ""
echo "=== I3b — snapshot  $TRAIN_SKILLS  →  $STAGING ==="
rm -rf "$STAGING"
cp -a "$TRAIN_SKILLS" "$STAGING"

# --- I3c: frozen short eval on validation ---
FREEZE_DRA_ID="i3frz_${I3_MODEL}_${I3_RUN_ID}"
export DRA_RUN_ID="$FREEZE_DRA_ID"
echo ""
echo "=== I3c — frozen eval  DRA_RUN_ID=$DRA_RUN_ID  max_samples=$I3_EVAL_SAMPLES ==="
echo "    skills_dir=$STAGING  enable_skill_extraction=False"

FREEZE_OPTS=(
  dataset.split=validation
  "max_samples=$I3_EVAL_SAMPLES"
  "agent_config.skills_dir=${STAGING}"
  agent_config.enable_skill_extraction=False
)
if [[ -n "${SMOKE_CFG_OPTIONS}" ]]; then
  # shellcheck disable=SC2206
  FREEZE_OPTS+=(${SMOKE_CFG_OPTIONS})
fi
if [[ "$I3_MODEL" == "gemma" ]]; then
  FREEZE_OPTS+=("concurrency=${GEMMA_CONCURRENCY:-4}")
fi

"$PYTHON" examples/run_gaia.py --config "$CFG" --cfg-options "${FREEZE_OPTS[@]}"

FREEZE_LOG="workdir/gaia_c4_${I3_MODEL}_${FREEZE_DRA_ID}/log.txt"
echo ""
echo "=== I3 complete ==="
echo "Artifact root:    workdir/c4_i3_${I3_RUN_ID}/"
echo "I3a train skills: $TRAIN_SKILLS"
echo "I3b staging copy: $STAGING"
echo "I3c run log:      $FREEZE_LOG"
echo "Pipeline log:     $LOG"
echo ""
echo "Checks: grep -E 'SkillExtractor active|building SkillRegistry' \"$FREEZE_LOG\""
