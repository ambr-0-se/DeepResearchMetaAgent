#!/usr/bin/env bash
# Idempotent: downloads Chromium into ~/.cache/ms-playwright if missing.
# Run after `conda activate <env>` so the same Python/env is used as the agent.
#
# Optional:
#   PLAYWRIGHT_WITH_DEPS=1   — add --with-deps (Linux: system libs; may need apt on bare images)
#
# Usage (from repo root):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate dra
#   bash scripts/ensure_playwright_browsers.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"

if ! "$PYTHON" -c "import playwright" 2>/dev/null; then
    echo "[ensure_playwright] ERROR: Python package 'playwright' not found."
    echo "  Install: pip install playwright"
    exit 1
fi

INSTALL_ARGS=(install chromium)
if [[ "${PLAYWRIGHT_WITH_DEPS:-}" == "1" ]]; then
    INSTALL_ARGS+=(--with-deps)
fi

echo "[ensure_playwright] playwright ${INSTALL_ARGS[*]} (safe to repeat)..."
"$PYTHON" -m playwright "${INSTALL_ARGS[@]}"
echo "[ensure_playwright] OK"
