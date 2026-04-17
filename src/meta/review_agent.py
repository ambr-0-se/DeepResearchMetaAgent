"""
ReviewAgent — internal, sealed review component for condition C3.

Architectural properties that make this agent SEALED against runtime modification:

1. Not registered via `@AGENT.register_module()` — so `AGENT.build()` cannot
   instantiate it from config, and it is invisible to config-driven
   managed-agents resolution.
2. Not added to the planning agent's `managed_agents` dict — so
   `modify_subagent._find_managed_agent()` cannot reach it.
3. Not added to the planning agent's `tools` dict — same reason.
4. Constructed via `object.__new__` + manual attribute init (following the
   pattern in `AgentGenerator._create_agent_instance`) to avoid YAML file
   I/O and keep the prompt template as an in-source string.

This sealing is deliberate: if the planner could modify the reviewer, the
agent could learn to make the reviewer lenient rather than solve the task
(classic reward hacking). Review apparatus must be a stable evaluation
surface across all C3 runs.

All LLM calls use `parent_agent.model` — no separate reviewer model.
"""

from typing import TYPE_CHECKING, Any

from src.agent.general_agent import GeneralAgent
from src.base.async_multistep_agent import populate_template
from src.logger import AgentLogger, LogLevel as LL, Monitor
from src.memory import AgentMemory
from src.meta.diagnose_tool import DiagnoseSubAgentTool
from src.tools.final_answer import FinalAnswerTool

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent
    from src.models import Model


# --- Prompt templates (in-source; avoids YAML file I/O and fs paths) --------

REVIEW_AGENT_SYSTEM_PROMPT: str = """\
You are a REVIEW agent. Your job is to assess whether a sub-agent's response
satisfied the task it was delegated, and — when it did not — to diagnose the
failure and recommend a next action.

You receive a single delegation to review, containing:
  * agent_name       — which sub-agent was delegated to
  * task_given       — the task passed to the sub-agent
  * expected_outcome — what the manager expected
  * actual_response  — what the sub-agent returned

WORKFLOW

Step 1 (ALWAYS). Emit a `final_answer_tool` call whose argument is a JSON object
matching the ReviewResult schema (see below). Do this in one step whenever
the verdict is clear from `actual_response` alone.

Step 2 (ONLY IF the verdict is not obvious from the response). Call
`diagnose_subagent` to inspect the sub-agent's execution history (reasoning,
tool calls, observations). Then emit `final_answer_tool`.

You have at most 3 steps. If you cannot reach a confident verdict, return
verdict="unsatisfactory" with next_action={"action": "escalate", ...}.

REVIEW RESULT SCHEMA

{
  "verdict":            "satisfactory" | "partial" | "unsatisfactory",
  "confidence":         <float in [0.0, 1.0]>,
  "summary":            "<one-line human-readable assessment>",
  "root_cause_primary": one of:
    "missing_tool"     — agent lacked a needed capability
    "wrong_tool"       — had the tool, picked wrong one
    "bad_instruction"  — task/prompt was underspecified
    "misread_task"     — agent misinterpreted the objective
    "external"         — network, rate limit, paywall
    "model_limit"      — reasoning capacity exceeded
    "unclear_goal"     — manager's task was ambiguous
    "incomplete"       — correct direction, not finished
  "root_cause_secondary": same values or null (optional, for compound causes),
  "root_cause_detail":  "<free text explanation>" (optional),
  "next_action":        one of the four discriminated variants below
}

root_cause_primary is REQUIRED when verdict != "satisfactory", and MUST be null
when verdict == "satisfactory".

NEXT ACTION VARIANTS (exactly one; dispatched on `action`)

{"action": "proceed"}
  Satisfactory. No further action. Use this for every verdict=satisfactory.

{"action": "retry",
 "agent_name": "<same sub-agent>",
 "revised_task": "<reformulated task text>",
 "additional_guidance": "<what to do differently>",
 "avoid_patterns": ["<pattern 1>", "<pattern 2>"]}
  Retry the same sub-agent with a clearer task. Use when the cause is
  bad_instruction, misread_task, or unclear_goal.

{"action": "modify_agent",
 "modify_action": one of [
   "add_existing_tool_to_agent", "add_new_tool_to_agent",
   "remove_tool_from_agent", "modify_agent_instructions",
   "add_agent", "remove_agent", "set_agent_max_steps"],
 "agent_name": "<sub-agent to modify>",
 "specification": "<spec string passed to modify_subagent>",
 "followup_retry": true|false}
  Modify a sub-agent's capability. Use when cause is missing_tool or
  wrong_tool. `modify_action` and `specification` are passed through to
  `modify_subagent` verbatim.

{"action": "escalate",
 "from_agent": "<original sub-agent>",
 "to_agent":   "<existing managed-agent name>",
 "reason":     "<why this agent is better suited>",
 "task":       "<task reformulated for the new agent>"}
  Switch to a different sub-agent. `to_agent` MUST be an existing managed
  agent name.

CONSTRAINTS

- You MUST return exactly one `final_answer_tool` call with a valid JSON payload.
- Do not propose modify_actions or agent names that don't exist. If unsure,
  prefer "retry" with clearer guidance over a speculative "modify_agent".
- Be concise: `summary` is a single line, `root_cause_detail` is at most 2
  sentences. The planner reads this verbatim in the next THINK.
- Only reach for `diagnose_subagent` when the failure reason isn't obvious.
  Every extra step costs tokens.

AVAILABLE TOOLS
{%- for tool in tools.values() %}
* {{ tool.name }}: {{ tool.description }}
{%- endfor %}
"""


REVIEW_AGENT_TASK_INSTRUCTION: str = """\
Review the following sub-agent delegation. Emit exactly one `final_answer_tool`
call whose argument is a ReviewResult JSON object.

{{task}}
"""


REVIEW_AGENT_USER_PROMPT: str = ""


# Minimal managed_agent block in case the ReviewAgent ever gets invoked as a
# tool (defensive; should never fire because it is not registered as a managed
# agent or in any tools dict).
REVIEW_AGENT_MANAGED_AGENT_TASK: str = (
    "You are the REVIEW agent '{{name}}'. A parent delegation unexpectedly "
    "treated you as a managed agent. Emit final_answer_tool with verdict=unsatisfactory, "
    "next_action=proceed, summary=\"review-agent-invoked-as-subagent\", and exit.\n"
    "Task: {{task}}"
)

REVIEW_AGENT_MANAGED_AGENT_REPORT: str = (
    "Review from {{name}}:\n{{final_answer}}"
)

# Minimal final_answer template block (prompt_templates key; tool is final_answer_tool).
# Used by AsyncMultiStepAgent when max_steps hit without a final answer.
REVIEW_AGENT_FINAL_ANSWER_PRE: str = (
    "You reached max_steps without emitting final_answer_tool. "
    "Provide a ReviewResult JSON with verdict=unsatisfactory, "
    "next_action={\"action\": \"escalate\", ...} based on what you saw."
)

REVIEW_AGENT_FINAL_ANSWER_POST: str = ""


def _build_prompt_templates() -> dict:
    """Assemble the prompt_templates dict that GeneralAgent expects."""
    return {
        "system_prompt": REVIEW_AGENT_SYSTEM_PROMPT,
        "user_prompt": REVIEW_AGENT_USER_PROMPT,
        "task_instruction": REVIEW_AGENT_TASK_INSTRUCTION,
        "managed_agent": {
            "task": REVIEW_AGENT_MANAGED_AGENT_TASK,
            "report": REVIEW_AGENT_MANAGED_AGENT_REPORT,
        },
        "final_answer": {
            "pre_messages": REVIEW_AGENT_FINAL_ANSWER_PRE,
            "post_messages": REVIEW_AGENT_FINAL_ANSWER_POST,
        },
    }


# --- ReviewAgent ------------------------------------------------------------

# NOTE: intentionally NOT decorated with @AGENT.register_module. Keeps this
# class out of the AGENT registry so config-driven managed-agents resolution
# cannot instantiate it, and so modify_subagent cannot find it.
class ReviewAgent(GeneralAgent):
    """
    Sealed review agent used by ReviewStep (C3).

    Instantiate via `ReviewAgent.build(...)`, NOT the inherited `__init__`,
    which would try to load a YAML template file. `build()` uses the
    `object.__new__` + manual attribute init pattern from
    `AgentGenerator._create_agent_instance` to keep the prompt in-source.

    Not a managed agent. Not in any tools dict. Invisible to modify_subagent.
    """

    #: Name used for logging; not registered anywhere.
    AGENT_NAME: str = "review_agent"

    #: Bounded reasoning depth — review should be fast.
    DEFAULT_MAX_STEPS: int = 3

    @classmethod
    def build(
        cls,
        parent_agent: "AsyncMultiStepAgent",
        model: "Model",
        max_steps: int | None = None,
    ) -> "ReviewAgent":
        """
        Construct a ReviewAgent sealed from the planner.

        Args:
            parent_agent: The planning agent whose sub-agent memory the review
                agent is allowed to inspect via DiagnoseSubAgentTool.
            model: LLM to use — should be `parent_agent.model`.
            max_steps: Override reasoning depth (default 3).

        Returns:
            A fully initialised ReviewAgent instance. The returned agent is
            NOT in any registry; the caller holds the only reference.
        """
        # Tools: DiagnoseSubAgentTool (with parent_agent) + final_answer_tool.
        # Do NOT include modify_subagent or any other adaptive tool — review
        # does not mutate the world, it assesses.
        diagnose_tool = DiagnoseSubAgentTool(parent_agent=parent_agent)
        final_answer_tool = FinalAnswerTool()
        tools = {
            diagnose_tool.name: diagnose_tool,
            final_answer_tool.name: final_answer_tool,
        }

        prompt_templates = _build_prompt_templates()

        # Minimal config object — no template_path (we supply templates in
        # memory), no file I/O at any construction step.
        class _ReviewAgentConfig:
            """Minimal config; mirrors the DynamicConfig pattern in AgentGenerator."""

            def __init__(self) -> None:
                self.name = cls.AGENT_NAME
                self.description = "Internal sealed review agent (C3)."
                self.max_steps = max_steps or cls.DEFAULT_MAX_STEPS
                self.template_path = None  # signals: no file load
                self.provide_run_summary = False
                # Context-pruning knobs referenced by GeneralAgent are
                # read via getattr with defaults, so we omit them here.

            def get(self, key: str, default: Any = None) -> Any:
                return getattr(self, key, default)

        config = _ReviewAgentConfig()

        # object.__new__ bypasses __init__ (which would try to load a YAML
        # template from self.config.template_path).
        agent = object.__new__(cls)

        # Core attributes
        agent.config = config
        agent.model = model
        agent.name = config.name
        agent.description = config.description
        agent.max_steps = config.max_steps
        agent.provide_run_summary = config.provide_run_summary

        # Tools
        agent.tools = tools

        # Empty managed_agents — review agent does not delegate.
        agent.managed_agents = {}

        # Prompt templates
        agent.prompt_templates = prompt_templates

        # Rendered prompts. The system prompt uses the {{tools}} iteration
        # from populate_template, same as GeneralAgent.initialize_system_prompt.
        agent.system_prompt = populate_template(
            prompt_templates["system_prompt"],
            variables={"tools": agent.tools, "managed_agents": agent.managed_agents},
        )
        agent.user_prompt = populate_template(
            prompt_templates["user_prompt"],
            variables={},
        )

        # Memory
        agent.memory = AgentMemory(
            system_prompt=agent.system_prompt,
            user_prompt=agent.user_prompt,
        )

        # Runtime bookkeeping (same as _create_agent_instance)
        agent.state = {}
        agent.step_number = 0
        agent.stream_outputs = False
        agent.max_tool_threads = None
        agent.logger = AgentLogger(level=LL.INFO)
        agent.final_answer_checks = []
        agent.return_full_result = False
        agent.step_callbacks = []
        agent.planning_interval = None
        agent.task = None
        agent.interrupt_switch = False
        agent.grammar = None
        agent.instructions = None

        # Tool-like interface (defensive — should never be invoked this way
        # because ReviewAgent is not in any managed_agents or tools dict).
        agent.inputs = {
            "task": {"type": "string", "description": "Delegation context to review."},
        }
        agent.output_type = "string"

        # Monitor for token metrics (parallel to all other agents).
        agent.monitor = Monitor(agent.model, agent.logger)
        agent.step_callbacks.append(agent.monitor.update_metrics)

        return agent
