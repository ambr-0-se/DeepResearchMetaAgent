#!/usr/bin/env bash
# Smoke checks for HANDOFF rows #2 (provider matrix), #3 (modify_subagent prompts), #4 (C2/C3 review + skills).
# Intended to finish quickly; does not replace full GAIA/matrix validation.
set -euo pipefail
# Re-exec inside conda env `dra` when mmengine is not on the default PATH.
if [[ "${_DRA_SMOKE_REEXEC:-}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  if ! python -c "import mmengine" 2>/dev/null; then
    if conda run -n dra python -c "import mmengine" 2>/dev/null; then
      export _DRA_SMOKE_REEXEC=1
      exec conda run -n dra --no-capture-output bash "$0" "$@"
    fi
  fi
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export REPO_ROOT="$ROOT"

echo "== Tier 0: mmengine load for 16 matrix configs (4 models × C0/C1/C2/C3) =="
for cfg in configs/config_gaia_c{0,1,2,3}_{mistral,kimi,qwen,gemma}.py; do
  python -c "from mmengine.config import Config; Config.fromfile('${cfg}'); print('OK', '${cfg}')"
done

echo "== HANDOFF #3: rendered planner prompts (C1/C2/C3; uses Mistral matrix configs so create_agent does not require qwen3.6-plus-failover) =="
python - <<'PY'
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

repo = Path(os.environ["REPO_ROOT"])
os.chdir(repo)
load_dotenv(repo / ".env", override=True)

from mmengine.config import Config
from src.agent.agent import create_agent
from src.models import model_manager

model_manager.init_models(use_local_proxy=False)

async def main():
    for cfg_path in [
        "configs/config_gaia_c1_mistral.py",
        "configs/config_gaia_c2_mistral.py",
        "configs/config_gaia_c3_mistral.py",
    ]:
        cfg = Config.fromfile(cfg_path)
        agent = await create_agent(cfg)
        ti = agent.prompt_templates["task_instruction"]
        assert "Preference order" in ti, f"{cfg_path}: preference-order missing"
        assert "| Failure mode" in ti, f"{cfg_path}: failure-mode table missing"
        assert "Do NOT" in ti, f"{cfg_path}: anti-pattern block missing"
        for a in [
            "add_new_tool_to_agent",
            "remove_tool_from_agent",
            "set_agent_max_steps",
            "remove_agent",
        ]:
            assert a in ti, f"{cfg_path}: missing action example {a}"
        if "_c2" in cfg_path or "_c3" in cfg_path:
            assert "Override REVIEW" in ti, f"{cfg_path}: REVIEW anti-pattern missing"
        else:
            assert "before at least one delegation has been attempted" in ti
        print(f"OK {cfg_path}")

asyncio.run(main())
PY

echo "== HANDOFF #4: unit tests for review schema + skill registry =="
pytest tests/test_review_schema.py tests/test_skill_registry.py -q --tb=short

echo "== HANDOFF #4: seed skill validator =="
python -m src.skills.validate src/skills

echo "== HANDOFF #2: model registration (needs keys in .env) =="
python - <<'PY'
import os
from pathlib import Path

from dotenv import load_dotenv

repo = Path(os.environ["REPO_ROOT"])
os.chdir(repo)
load_dotenv(repo / ".env", override=True)

from src.models.models import ModelManager

m = ModelManager()
m.init_models(use_local_proxy=False)
# Wire ids must match scripts/gen_eval_configs.py matrix defaults.
core = [
    "mistral-small",
    "or-kimi-k2.5",
    "or-qwen3.6-plus",
    "or-gemma-4-31b-it",
]
missing_core = [n for n in core if n not in m.registed_models]
if missing_core:
    print("MISSING matrix models:", missing_core)
    raise SystemExit(1)
for n in (
    "langchain-mistral-small",
    "langchain-or-kimi-k2.5",
    "langchain-or-qwen3.6-plus",
    "langchain-or-gemma-4-31b-it",
):
    if n not in m.registed_models:
        print("MISSING langchain wrapper:", n)
        raise SystemExit(1)
print("OK: 4 matrix models + OpenRouter langchain aliases")
if "qwen3.6-plus-failover" in m.registed_models:
    print("INFO: qwen3.6-plus-failover also registered (optional)")
else:
    print(
        "INFO: qwen3.6-plus-failover not registered — matrix Qwen uses "
        "or-qwen3.6-plus directly; set STRICT_QWEN_FAILOVER=1 only if you "
        "require failover registration for a different config."
    )
    import os

    if os.environ.get("STRICT_QWEN_FAILOVER") == "1":
        raise SystemExit(1)
PY

echo "== Optional (run manually if time): pytest tests/test_failover_model.py tests/test_reasoning_preservation.py tests/test_tier_b_tool_messages.py -q =="
echo "All smoke_validate_handoffs_234 steps completed."
