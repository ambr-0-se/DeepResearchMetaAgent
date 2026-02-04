# Adaptive Planning Agent - Setup & Troubleshooting Guide

**Created:** 2025-02-03
**Purpose:** Quick reference for running adaptive agent on HKU CS GPU farm

---

## What Was Implemented

Added **AdaptivePlanningAgent** - a self-modifying planning agent with THINK-ACT-OBSERVE-REFLECT loop:

### New Files Created:
```
src/meta/                           # Meta-agent module
├── __init__.py
├── adaptive_mixin.py              # State management & modification methods
├── diagnose_tool.py               # Diagnose sub-agent failures
├── modify_tool.py                 # Modify sub-agents at runtime
├── tool_generator.py              # Generate tools from descriptions
└── agent_generator.py             # Create agents dynamically

src/agent/adaptive_planning_agent/  # Adaptive planning agent
├── __init__.py
├── adaptive_planning_agent.py
└── prompts/
    └── adaptive_planning_agent.yaml

configs/config_gaia_adaptive.py     # Config for GAIA evaluation
scripts/compare_results.py          # Compare baseline vs adaptive results
```

### Key Features:
1. **diagnose_subagent** tool - Investigates why sub-agents fail
2. **modify_subagent** tool - Adds/removes tools, modifies instructions, creates new agents
3. **Task-scoped modifications** - All changes reset after each task
4. **In-memory agent generation** - Creates specialized agents without file I/O

---

## Quick Start Commands

### 1. Environment Setup
```bash
conda create -n dra python=3.11
conda activate dra
cd /path/to/DeepResearchAgent
make install
```

### 2. Configure API Keys
```bash
cp .env.template .env
# Edit .env and add your API keys:
# - ANTHROPIC_API_KEY (for claude-3.7-sonnet-thinking)
# - FIRECRAWL_API_KEY (for web search)
```

### 3. Run Evaluation
```bash
# Baseline (existing system)
python examples/run_gaia.py --config configs/config_gaia.py

# Adaptive agent (new system)
python examples/run_gaia.py --config configs/config_gaia_adaptive.py

# Compare results
python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_adaptive/dra.jsonl
```

---

## Configuration Details

### Default Model
- **config_gaia_adaptive.py** uses `claude-3.7-sonnet-thinking`
- Change in config: `model_id="claude-3.7-sonnet-thinking"`

### Max Steps
- Baseline: 20 steps
- Adaptive: 25 steps (extra 5 for reflection/adaptation)

### Managed Agents
- deep_analyzer_agent
- browser_use_agent
- deep_researcher_agent

---

## Known Issues & Fixes

### Issue 1: Import Errors
**Symptom:** `ModuleNotFoundError: No module named 'huggingface_hub'`
**Fix:** Run `make install` (not `make install-requirements`)

### Issue 2: Registry Check Error
**Status:** ✅ Fixed (line 210 in modify_tool.py)
**What was fixed:** Changed `if tool_name not in TOOL` to `if tool_name not in TOOL.module_dict`

### Issue 3: GAIA Dataset Not Found
**Symptom:** Dataset download fails
**Fix:** Ensure network access and HuggingFace authentication
```bash
huggingface-cli login
```

### Issue 4: Out of Memory
**Symptom:** CUDA OOM or Python memory error
**Fix:** Reduce batch size or use smaller model
- Edit config: `max_steps=15` instead of 25
- Or use fewer managed agents

### Issue 5: Concurrent Task Execution
**Status:** ⚠️ Not supported (by design)
**Note:** AdaptivePlanningAgent.run() is not thread-safe. GAIA evaluation is sequential, so this is fine.

---

## Important Notes for GPU Farm

### 1. File Permissions
```bash
# Make sure scripts are executable
chmod +x scripts/compare_results.py
```

### 2. Output Directory
Results are saved to `workdir/{tag}/`:
- Baseline: `workdir/gaia/`
- Adaptive: `workdir/gaia_adaptive/`

### 3. API Rate Limits
- Anthropic Claude API has rate limits
- For GAIA validation set (165 questions), expect ~2-3 hours runtime
- Consider adding delays if hitting rate limits

### 4. Logging
Logs are saved to `workdir/{tag}/logs/`
Check these if agent behavior is unexpected.

---

## Testing Before Full Run

Test with a single question first:
```python
# Create test script: test_single.py
from src.agent.agent import create_agent
from mmengine import Config
import asyncio

async def test():
    config = Config.fromfile('configs/config_gaia_adaptive.py')
    agent = await create_agent(config, config.agent_config)
    
    test_task = "What is 2+2?"
    result = await agent.run(test_task)
    print(f"Result: {result}")

asyncio.run(test())
```

Run: `python test_single.py`

---

## Troubleshooting Checklist

Before asking for help:
- [ ] Ran `make install` (not `pip install -r requirements.txt`)
- [ ] Activated conda environment: `conda activate dra`
- [ ] Set API keys in `.env`
- [ ] Checked `workdir/*/logs/` for errors
- [ ] Tested with single question first
- [ ] Verified GPU availability: `nvidia-smi`

---

## Expected Results

### Baseline Performance (config_gaia.py)
Reference accuracy from previous runs (if available)

### Adaptive Agent Hypothesis
The adaptive agent should:
- ✅ Perform better on tasks requiring specialized tools
- ✅ Diagnose and fix sub-agent failures
- ⚠️ May have higher API costs due to extra steps
- ⚠️ May have longer runtime due to reflection

### Metrics to Track
- Accuracy by task level (Level 1, 2, 3)
- Number of tool modifications per task
- Number of agent creations per task
- Average steps per task

---

## Quick Debug Commands

```bash
# Check if adaptive agent is registered
python -c "from src.registry import AGENT; print('adaptive_planning_agent' in AGENT.module_dict)"

# Test imports
python -c "from src.meta import AdaptiveMixin, DiagnoseSubAgentTool, ModifySubAgentTool; print('OK')"

# Check config loading
python -c "from mmengine import Config; c = Config.fromfile('configs/config_gaia_adaptive.py'); print(c.agent_config)"
```

---

## Contact & Next Steps

After running evaluation:
1. Check results in `workdir/gaia_adaptive/dra.jsonl`
2. Run comparison script
3. Analyze which tasks improved/regressed
4. Check logs for adaptive tool usage patterns

## Files to Monitor
- `workdir/gaia_adaptive/dra.jsonl` - Results
- `workdir/gaia_adaptive/logs/` - Execution logs
- Memory usage during long runs
