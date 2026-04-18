# DeepResearchAgent

[![Website](https://img.shields.io/badge/🌐-Website-blue?style=for-the-badge&logo=github)](https://skyworkai.github.io/DeepResearchAgent/)
[![Paper](https://img.shields.io/badge/📄-arXiv%20Paper-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2506.12508)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

English | [简体中文](README_CN.md) | [🌐 **Website**](https://skyworkai.github.io/DeepResearchAgent/)

> ⚡️🚀 **Important**: We have developed a new agent protocol called **Tool-Environment-Agent (TEA)**, which allows you to build agents as flexibly as brewing tea. It’s still in the testing phase — if you’re interested, please check it out:
>
> 👉 [https://github.com/DVampire/AgentWorld](https://github.com/DVampire/AgentWorld)  
> 📄 [https://arxiv.org/abs/2506.12508](https://arxiv.org/abs/2506.12508)
## Introduction

DeepResearchAgent is a hierarchical multi-agent system designed not only for deep research tasks but also for general-purpose task solving. The framework leverages a top-level planning agent to coordinate multiple specialized lower-level agents, enabling automated task decomposition and efficient execution across diverse and complex domains.

> 🌐 **Check out our interactive website**: [https://skyworkai.github.io/DeepResearchAgent/](https://skyworkai.github.io/DeepResearchAgent/) - Explore the architecture, view experiments, and learn more about our research!

## Architecture

<p align="center">

  <img src="./docs/assets/architecture.png" alt="Architecture" width="700"/>

</p>

The system adopts a two-layer structure:

### 1. Top-Level Planning Agent

* Responsible for understanding, decomposing, and planning the overall workflow for a given task.
* Breaks down tasks into manageable sub-tasks and assigns them to appropriate lower-level agents.
* Dynamically coordinates the collaboration among agents to ensure smooth task completion.

**Adaptive Planning Agent** — An extended variant that adds runtime self-modification capabilities, used in experimental conditions C2/C3/C4:
* Uses a THINK-ACT-OBSERVE loop (same as the base `PlanningAgent`) augmented with reactive self-modification tools.
* Can diagnose sub-agent failures at runtime via `diagnose_subagent` and dynamically modify sub-agent tools, instructions, and capabilities via `modify_subagent` (condition **C2**, and inherited by C3/C4).
* All architectural modifications are task-scoped and reset after each task.
* **C3**: adds a structural REVIEW step — an automatic post-delegation assessment that produces a structured verdict with a root-cause taxonomy and a recommended `next_action` (proceed / retry / modify_agent / escalate). The review apparatus is sealed from `modify_subagent` to prevent reward hacking. See `configs/config_gaia_c3.py`.
* **C4**: adds a cross-task skill library following the [agentskills.io](https://agentskills.io/specification) spec. Each agent (planner and sub-agents) gets a consumer-scoped `activate_skill` tool. Seven pre-seeded skills ship with the repo under `src/skills/`; every C4 invocation copies them into a per-run `workdir/gaia_c4_<model>_<DRA_RUN_ID>/skills/` (see `src/skills/_seed.py`) so parallel runs and repeated runs never collide. An end-of-task `SkillExtractor` proposes new skills during training runs (with an entity blocklist + LLM-judge dedup). Disable extraction via `enable_skill_extraction=False` for frozen-library evaluation. See `configs/config_gaia_c4.py` and `src/skills/`.

### 2. Specialized Lower-Level Agents

* **Deep Analyzer**

  * Performs in-depth analysis of input information, extracting key insights and potential requirements.
  * Supports analysis of various data types, including text and structured data.
* **Deep Researcher**

  * Conducts thorough research on specified topics or questions, retrieving and synthesizing high-quality information.
  * Capable of generating research reports or knowledge summaries automatically.
* **Browser Use**

  * Automates browser operations, supporting web search, information extraction, and data collection tasks.
  * Assists the Deep Researcher in acquiring up-to-date information from the internet.
  
* **MCP Manager Agent**
  * Manages and orchestrates Model Context Protocol (MCP) tools and services.
  * Enables dynamic tool discovery, registration, and execution through MCP standards.
  * Supports both local and remote MCP tool integration for enhanced agent capabilities.

* **General Tool Calling Agent**
  * Provides a general-purpose interface for invoking various tools and APIs.
  * Supports function calling, allowing the agent to execute specific tasks or retrieve information from external services.

## Features

- Hierarchical agent collaboration for complex and dynamic task scenarios
- Extensible agent system, allowing easy integration of additional specialized agents
- Automated information analysis, research, and web interaction capabilities
- Secure Python code execution environment for tools, featuring configurable import controls, restricted built-ins, attribute access limitations, and resource limits. (See [PythonInterpreterTool Sandboxing](./docs/python_interpreter_sandbox.md) for details).
- Support for asynchronous operations, enabling efficient handling of multiple tasks and agents
- Adaptive self-modification: the AdaptivePlanningAgent can diagnose failures and modify sub-agents at runtime
- Token-budget-aware context pruning to handle long conversations within model context limits
- OpenAI-native Tier B tool message protocol for parallel tool call tracking
- Support for local and remote model inference, including OpenAI, Anthropic, Google LLMs, and local Qwen models via vLLM
- Support for image and video generation tools based on the Imagen and Veo3 models, respectively
- GAIA evaluation infrastructure with vLLM health watchdog, auto-restart, and transient error retry

Image and Video Examples:
<p align="center">
  <img src="./docs/assets/cat_yarn_playful_reference.png" alt="Image Example" width="300"/>
    <img src="./docs/assets/cat_playing_with_yarn_video.gif" alt="Video Example" width="600"/>
</p>

## Updates
* **2026.04**: Ship conditions **C3** and **C4** for the ADAS GAIA experiments.
  * **C3 — structural REVIEW step**: automatic post-delegation assessment via a sealed internal `ReviewAgent`. Produces a Pydantic `ReviewResult` with a verdict, 8-category root-cause taxonomy, and a polymorphic `next_action` that maps directly onto `modify_subagent` arguments when remediation is needed. Review findings are injected into the next THINK's observations with a `[REVIEW]` marker. Apparatus is sealed from `modify_subagent` to prevent reward hacking. See `src/meta/review_*.py` and `configs/config_gaia_c3.py`.
  * **C4 — cross-task skill library**: filesystem-backed skill registry following the [agentskills.io](https://agentskills.io/specification) standard. Each agent (planner + sub-agents) gets a consumer-scoped `activate_skill` tool. Seven pre-seeded skills cover all four consumer scopes. A six-stage `SkillExtractor` proposes new skills at task end (worthiness heuristic → LLM propose → entity blocklist → dedup → persist); frozen-library mode available for evaluation. See `src/skills/` and `configs/config_gaia_c4.py`.
* **2026.04**: Reorganise GAIA eval into four experimental conditions (C0/C2/C3/C4). Correct misleading THINK-ACT-OBSERVE-REFLECT naming (the base loop is plain THINK-ACT-OBSERVE; structural REVIEW is introduced by C3). Extract shared `src/meta/_memory_format.py` helpers for reuse across diagnose/review components.
* **2026.02**: Codebase cleanup — remove obsolete scripts, improve eval reporting (retry tracking, per-tool results, broader error classification), and update documentation.
* **2026.02**: GAIA evaluation robustness — vLLM health watchdog with auto-restart, transient error retry in eval runner, token-budget-aware context pruning, and planning tool auto-ID generation.
* **2026.02**: Add OpenAI-native Tier B tool message protocol for parallel tool call tracking.
* **2026.01**: Add AdaptivePlanningAgent with runtime self-modification (diagnose/modify sub-agents), GAIA evaluation analysis script (`scripts/analyze_results.py`), and Qwen3-VL-4B-Instruct support.
* **2025.08.04**: Add the support for loading mcp tools from the local json file.
* **2025.07.08**: Add the video generator tool, which can generate a video based on the input text and/or image. The video generator tool is based on the Veo3 model.
* **2025.07.08**: Add the image generator tool, which can generate images based on the input text. The image generator tool is based on the Imagen model.
* **2025.07.07**: Due to the limited flexibility of TOML configuration files, we have switched to using the config format supported by mmengine.
* **2025.06.20**: Add the support for the mcp (Both the local mcp and remote mcp).
* **2025.06.17**: Update technical report https://arxiv.org/pdf/2506.12508.
* **2025.06.01**: Update the browser-use to 0.1.48.
* **2025.05.30**: Convert the sub agent to a function call. Planning agent can now be gpt-4.1 or gemini-2.5-pro.
* **2025.05.27**: Support OpenAI, Anthropic, Google LLMs, and local Qwen models (via vLLM, see details in [Usage](#usage)).

## TODO List

* [x] Asynchronous feature completed
* [x] Image Generator Tool completed
* [x] Video Generator Tool completed
* [x] MCP completed
* [x] Load local MCP tools from JSON file completed
* [ ] AI4Research Agent to be developed
* [ ] Novel Writing Agent to be developed

## Installation

### Prepare Environment

```bash
# poetry install environment
conda create -n dra python=3.11
conda activate dra
make install

# (Optional) You can also use requirements.txt
conda create -n dra python=3.11
conda activate dra
make install-requirements

# playwright install if needed
pip install playwright
playwright install chromium --with-deps --no-shell
```

### Set Up `.env`

Please refer to the `.env.template` file and create a `.env` file in the root directory of the project. This file is used to configure API keys and other environment variables.

Refer to the following instructions to obtain the necessary google gemini-2.5-pro API key and set it in the `.env` file:

* [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
* [https://cloud.google.com/docs/authentication/application-default-credentials?hl=zh-cn](https://cloud.google.com/docs/authentication/application-default-credentials?hl=zh-cn)

```bash
brew install --cask google-cloud-sdk
gcloud init
gcloud auth application-default login
```

## Usage

### Main Example
A simple example to demonstrate the usage of the DeepResearchAgent framework.
```bash
python main.py
```

### Run Single Agent Example
A simple example to demonstrate the usage of a single agent, such as a general tool calling agent.
```bash
python examples/run_general.py
```

### Run GAIA Evaluation

```bash
# Download GAIA
mkdir data && cd data
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA

# Run baseline evaluation
python examples/run_gaia.py --config configs/config_gaia.py

# Run adaptive agent evaluation
python examples/run_gaia.py --config configs/config_gaia_adaptive.py

# Override config options (e.g. model, sample count)
python examples/run_gaia.py --config configs/config_gaia.py --cfg-options model_id=gpt-4.1 max_samples=10

# Compare baseline vs adaptive results
python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_adaptive/dra.jsonl

# Analyze results and generate HTML report
python scripts/analyze_results.py workdir/gaia/dra.jsonl
```

For GPU cluster evaluation with vLLM (SLURM), see [INSTRUCTIONS_RUN_EVAL.md](./INSTRUCTIONS_RUN_EVAL.md).

### ADAS evaluation matrix (16 cells — 4 models × 4 conditions)

The APAI4799 meta-agent research project adds a 16-cell GAIA evaluation matrix
comparing **C0 / C2 / C3 / C4** (defined above) across **4 frontier models**.
All configs are generated by [`scripts/gen_eval_configs.py`](./scripts/gen_eval_configs.py)
and launched in parallel by [`scripts/run_eval_matrix.sh`](./scripts/run_eval_matrix.sh).

```bash
# Regenerate all 16 configs (one per model × condition) from the template
python scripts/gen_eval_configs.py

# Smoke run — 5 questions × 16 cells = 80 Q, validation split, ~$2-5
bash scripts/run_eval_matrix.sh smoke

# Full submission run — test split, all 16 cells
bash scripts/run_eval_matrix.sh full

# Single model only, e.g. just Gemma
bash scripts/run_eval_matrix.sh full gemma
```

Model slots (see [`scripts/gen_eval_configs.py`](./scripts/gen_eval_configs.py)
`MODELS` table for current registration):

| Slot | `model_id` | Route | `tool_choice` handling | Rationale |
|------|------------|-------|------------------------|-----------|
| Mistral | `mistral-small` | Native Mistral La Plateforme | `"required"` passes through | Dense ~24B; `MISTRAL_API_KEY` |
| Kimi | `or-kimi-k2.5` | OpenRouter → `moonshotai/kimi-k2.5` | `"required"` works after `extra_body` fix | Provider pinned to `Moonshot AI` (allow_fallbacks=false); `reasoning.enabled=false` disables Moonshot's default thinking mode — required for `tool_choice="required"` on this provider. `stop` is stripped (Moonshot halts before tool-call emission when `stop` is non-empty, regardless of the stop string). |
| Qwen | `or-qwen3.6-plus` | OpenRouter → `qwen/qwen3.6-plus` | **Hybrid dispatch** → `"auto"` | Whole Qwen family rejects `tool_choice="required"` at OR's routing layer. [`src/models/tool_choice.py`](./src/models/tool_choice.py) downgrades any wire-id starting with `qwen/` to `"auto"` and a retry guard in `GeneralAgent._step_stream` / `ToolCallingAgent._step_stream` re-prompts plain-text replies back into tool calls (max 2 retries). |
| Gemma | `or-gemma-4-31b-it` | OpenRouter → `google/gemma-4-31b-it` | `"required"` passes through | Dense 31B, Apache 2.0 (only non-MoE in the matrix). Provider pin `["DeepInfra","Together"]` + `reasoning.enabled=false` + concurrency cap 4 in [`scripts/run_eval_matrix.sh`](./scripts/run_eval_matrix.sh) avoids the vLLM gemma4 parser pad-bug under parallel load. `:free` variant intentionally excluded. |

Extended protocol (pre-flight, smoke, C4 train/freeze, submission, resume,
triage): see [`docs/handoffs/HANDOFF_TEST_EVAL.md`](./docs/handoffs/HANDOFF_TEST_EVAL.md).
Per-model provider routing is declared in
[`src/models/models.py`](./src/models/models.py) `_register_openrouter_models`.
Hybrid `tool_choice` dispatch and the retry guard are in
[`src/models/tool_choice.py`](./src/models/tool_choice.py) and
[`src/agent/general_agent/general_agent.py`](./src/agent/general_agent/general_agent.py).

> **Env replication.** The `dra` conda env from `make install` / `make
> install-requirements` is the supported runtime. See
> [`CLAUDE.md`](./CLAUDE.md) for details and a pytest-collection gotcha
> when running the full test sweep.

## Experiments

We evaluated our agent on both GAIA validation and test sets, achieving state-of-the-art performance. Our system demonstrates superior performance across all difficulty levels.

<p align="center">
  <img src="./docs/assets/gaia_test.png" alt="GAIA Test Results" width="300"/>
  <img src="./docs/assets/gaia_validation.png" alt="GAIA Validation Results" width="300"/>
</p>

With the integration of the Computer Use and MCP Manager Agent, which now enables pixel-level control of the browser, our system demonstrates remarkable evolutionary capabilities. The agents can dynamically acquire and enhance their abilities through learning and adaptation, leading to significantly improved performance. The latest results show:
- **Test Set**: 83.39 (average), with 93.55 on Level 1, 83.02 on Level 2, and 65.31 on Level 3
- **Validation Set**: 82.4 (average), with 92.5 on Level 1, 83.7 on Level 2, and 57.7 on Level 3

## Questions

### 1. About Qwen Models

Our framework supports the following Qwen models (via vLLM):

* qwen2.5-7b-instruct
* qwen2.5-14b-instruct
* qwen2.5-32b-instruct
* Qwen3-VL-4B-Instruct (vision-language)

Update your config (mmengine Python format):

```python
model_id = "qwen2.5-7b-instruct"
```

### 2. Browser Use

If problems occur, reinstall:

```bash
pip install "browser-use[memory]"==0.1.48
pip install playwright
playwright install chromium --with-deps --no-shell
```

### 3. Sub-Agent Calling

Function-calling is now supported natively by GPT-4.1 / Gemini 2.5 Pro. Claude-3.7-Sonnet is also recommended.

### 4. Use vllm for local models
We provide huggingface as a shortcut to the local model. Also provide vllm as a way to start services so that parallel acceleration can be provided.

#### Step 1: Launch the vLLM Inference Service

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /input0/Qwen3-32B \
  --served-model-name Qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 16 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor_parallel_size 2' > vllm_qwen.log 2>&1 &
```

Update `.env`:

```bash
QWEN_API_BASE=http://localhost:8000/v1
QWEN_API_KEY="abc"
```

#### Step 2: Launch the Agent Service

```bash
python main.py
```

Example command:

```bash
Use deep_researcher_agent to search the latest papers on the topic of 'AI Agent' and then summarize it.
```

## Acknowledgement

DeepResearchAgent is primarily inspired by the architecture of smolagents. The following improvements have been made:
- The codebase of smolagents has been modularized for better structure and organization.
- The original synchronous framework has been refactored into an asynchronous one.
- The multi-agent setup process has been optimized to make it more user-friendly and efficient.

We would like to express our gratitude to the following open source projects, which have greatly contributed to the development of this work:
- [smolagents](https://github.com/huggingface/smolagents) - A lightweight agent framework.
- [OpenManus](https://github.com/mannaandpoem/OpenManus) - An asynchronous agent framework.
- [browser-use](https://github.com/browser-use/browser-use) - An AI-powered browser automation tool.
- [crawl4ai](https://github.com/unclecode/crawl4ai) - A web crawling library for AI applications.
- [markitdown](https://github.com/microsoft/markitdown) - A tool for converting files to Markdown format.

We sincerely appreciate the efforts of all contributors and maintainers of these projects for their commitment to advancing AI technologies and making them available to the wider community.

## Contribution

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.

## Cite

```bibtex
@misc{zhang2025agentorchestrahierarchicalmultiagentframework,
      title={AgentOrchestra: A Hierarchical Multi-Agent Framework for General-Purpose Task Solving}, 
      author={Wentao Zhang, Liang Zeng, Yuzhen Xiao, Yongcong Li, Ce Cui, Yilei Zhao, Rui Hu, Yang Liu, Yahui Zhou, Bo An},
      year={2025},
      eprint={2506.12508},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.12508}, 
}
```

---

### 🇨🇳 中文版说明文档

如果你更习惯阅读中文说明文档，请查阅 [README_CN.md](./README_CN.md)。
