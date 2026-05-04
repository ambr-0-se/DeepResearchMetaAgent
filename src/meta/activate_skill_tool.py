"""
ActivateSkillTool — loads a skill's full workflow into an agent's context.

This is the tool the planner (and sub-agents under C3) calls to perform the
"activation" step of progressive disclosure:

    Metadata always loaded  -->  skill registry block in system prompt
    Body loaded on demand   -->  this tool

Pattern mirrors `DiagnoseSubAgentTool` (same shape, different payload):
- Holds a reference to the agent that owns it (`parent_agent`)
- Holds a reference to the shared `SkillRegistry`
- Has a fixed `consumer` identity so it only activates skills visible to
  the agent it is attached to (planner or a specific sub-agent)

Why per-agent consumer binding:
    A single ActivateSkillTool is bound to one consumer. The planner's
    instance has consumer="planner"; each sub-agent's instance has
    consumer=<its_name>. If an agent tries to activate a skill scoped to
    a different consumer, the tool returns an error rather than leaking
    cross-scope content.
"""

from typing import TYPE_CHECKING

from src.logger import LogLevel, logger
from src.tools import AsyncTool

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent
    from src.skills._registry import SkillRegistry


class ActivateSkillTool(AsyncTool):
    """
    Tool that loads a specific SKILL.md body into the calling agent's
    conversation as the tool's response.

    Parameters:
        skill_name — lowercase-hyphen skill identifier
        reason     — one sentence explaining why this skill applies

    `reason` is required for research telemetry: we log every activation
    with its stated reason so later analysis can correlate which skills
    were invoked on which task types. The cost of one extra string in
    the tool call is negligible compared to the research value.
    """

    name = "activate_skill"
    description = (
        "Activate a skill from the skill registry to load its full workflow "
        "into your context. The tool returns the skill's Markdown body, "
        "which you should then follow step-by-step. Use this when you "
        "recognise a task pattern matching a skill's description in the "
        "<skill-registry> section of your system prompt. Provide the exact "
        "skill_name (lowercase with hyphens) and a one-sentence reason "
        "explaining why the skill applies."
    )
    parameters = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": (
                    "Exact skill name as shown in the <skill-registry>, "
                    "e.g. 'handling-file-attachments'."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "One sentence explaining why this skill is relevant to "
                    "the current task. Used for telemetry."
                ),
            },
        },
        "required": ["skill_name", "reason"],
    }
    output_type = "string"

    def __init__(
        self,
        parent_agent: "AsyncMultiStepAgent",
        registry: "SkillRegistry",
        consumer: str,
    ) -> None:
        """
        Bind the tool to a specific agent and a specific consumer scope.

        Args:
            parent_agent: The agent that owns this tool (held for logging
                context; not used for registry lookup because the registry
                is shared across agents).
            registry: The shared SkillRegistry instance.
            consumer: Which consumer view this tool operates under.
                Usually "planner" for the planning agent, or the sub-agent's
                own `name` attribute when installed on a sub-agent.
        """
        super().__init__()
        self.parent_agent = parent_agent
        self.registry = registry
        self.consumer = consumer

    async def forward(self, skill_name: str, reason: str) -> str:
        """
        Look up and return the skill body, or an error string.

        Behaviour:
          - Skill missing → returns an error string listing available skill
            names for this consumer. Does NOT raise — the caller agent sees
            the error in its observation and can react.
          - Skill exists but not visible to this consumer → returns an error
            explaining the scope mismatch.
          - Skill exists and visible → returns the body. Telemetry is logged.
        """
        # Visibility check: derive visible set from the registry for this
        # consumer. Using metadata_for (not raw lookup) so "all"-scoped
        # skills are included.
        visible_names = {m.name for m in self.registry.metadata_for(self.consumer)}

        if skill_name not in visible_names:
            # Distinguish "missing" from "not visible" for a better error.
            exists = self.registry.get(skill_name) is not None
            if exists:
                msg = (
                    f"Error: skill '{skill_name}' exists but is not visible "
                    f"to consumer '{self.consumer}'. Its consumer scope "
                    f"does not match. Available for you: "
                    f"{sorted(visible_names) or '(none)'}"
                )
            else:
                msg = (
                    f"Error: no skill named '{skill_name}'. Available for "
                    f"consumer '{self.consumer}': {sorted(visible_names) or '(none)'}"
                )
            logger.log(
                f"[ActivateSkillTool consumer={self.consumer}] "
                f"activation failed: {msg}",
                level=LogLevel.WARNING,
            )
            return msg

        body = self.registry.load_body(skill_name)
        logger.log(
            f"[ActivateSkillTool consumer={self.consumer}] "
            f"activated skill={skill_name!r} reason={reason!r}",
            level=LogLevel.INFO,
        )
        return (
            f"## Activated skill: {skill_name}\n\n"
            f"(reason given: {reason})\n\n"
            f"{body}"
        )
