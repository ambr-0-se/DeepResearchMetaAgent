"""
AdaptivePlanningAgent: Planning agent with runtime self-modification capabilities.

This agent extends PlanningAgent with the ability to:
- Diagnose sub-agent failures by examining their execution history
- Modify sub-agents' tools, instructions, and capabilities at runtime
- Create new specialized sub-agents dynamically
- All modifications are task-scoped and reset after each task
"""

from pathlib import Path
from typing import Any, Optional
import yaml

from src.agent.planning_agent import PlanningAgent
from src.base.async_multistep_agent import PromptTemplates
from src.memory import ActionStep, AgentMemory
from src.models import Model
from src.registry import AGENT
from src.utils import assemble_project_path
from src.logger import logger, LogLevel

# Import meta-agent components
from src.meta.activate_skill_tool import ActivateSkillTool
from src.meta.adaptive_mixin import AdaptiveMixin
from src.meta.diagnose_tool import DiagnoseSubAgentTool
from src.meta.modify_tool import ModifySubAgentTool
from src.meta.review_step import ReviewStep
from src.skills import SkillRegistry
from src.skills._extractor import SkillExtractor
from src.skills._seed import seed_skills_dir


@AGENT.register_module(name="adaptive_planning_agent", force=True)
class AdaptivePlanningAgent(AdaptiveMixin, PlanningAgent):
    """
    Planning agent with THINK-ACT-OBSERVE loop and reactive self-modification capabilities.

    Extends PlanningAgent with:
    - AdaptiveMixin for runtime modifications to sub-agents
    - diagnose_subagent tool for investigating sub-agent failures (reactive; agent-invoked)
    - modify_subagent tool for modifying sub-agent capabilities (reactive; agent-invoked)
    - Enhanced prompt template with reactive diagnose/modify guidance
    - State management for task-scoped changes

    Note: a structural REVIEW step (automatic post-delegation assessment) is added by the
    C2 variant via the optional `review_step` component; see `src/meta/review_step.py`.

    All modifications made during a task are automatically reset after task completion,
    ensuring isolation between tasks.
    """
    
    def __init__(
        self,
        config,
        tools: list[Any],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        review_step: Optional["ReviewStep"] = None,
        skill_registry: Optional["SkillRegistry"] = None,
        **kwargs,
    ):
        """
        Initialize the AdaptivePlanningAgent.

        Args:
            config: Agent configuration. Two optional flags selected by
                the experimental condition:
                - `enable_review` (bool, default False): if True, a
                  `ReviewStep` is built and attached; this is the C2 flag.
                - `enable_skills` (bool, default False): reserved for C3.
            tools: List of tools for the agent
            model: LLM model to use
            prompt_templates: Optional custom prompt templates
            planning_interval: Steps between planning phases
            stream_outputs: Whether to stream outputs
            max_tool_threads: Max threads for parallel tool execution
            review_step: Explicit `ReviewStep` instance (bypasses
                `config.enable_review`). Useful for programmatic
                construction in tests. If omitted, the config flag is
                consulted and a ReviewStep is built internally.
            **kwargs: Additional arguments passed to parent
        """
        # Call parent constructor
        super().__init__(
            config=config,
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            stream_outputs=stream_outputs,
            max_tool_threads=max_tool_threads,
            **kwargs,
        )

        # Add adaptive tools (diagnose and modify)
        self._add_adaptive_tools()

        # Optional compositional components, selected by config:
        # C1 -> both None (reactive tools only)
        # C2 -> review_step is built; skill_registry is None
        # C3 -> both are built
        self.review_step: Optional[ReviewStep] = review_step or self._build_review_step_from_config()
        self.skill_registry: Optional[SkillRegistry] = (
            skill_registry or self._build_skill_registry_from_config()
        )

        # Install skill tools on the planner AND each managed sub-agent
        # (one-time). Per-run skill_registry_block refresh happens in run().
        if self.skill_registry is not None:
            self._install_skill_tools()

        # Skill extractor (optional — C3 training mode). Frozen-library
        # eval mode leaves this as None; extraction runs only when the
        # `enable_skill_extraction` flag is set.
        self.skill_extractor: Optional[SkillExtractor] = None
        if self.skill_registry is not None and getattr(
            self.config, "enable_skill_extraction", False
        ):
            self.skill_extractor = SkillExtractor(
                parent_agent=self, registry=self.skill_registry
            )
            logger.log(
                "[AdaptivePlanningAgent] enable_skill_extraction=True; "
                "SkillExtractor active (C3 training mode)",
                level=LogLevel.INFO,
            )

        # Refresh system prompt to include the new tools
        self._refresh_system_prompt()

        logger.log(
            f"[AdaptivePlanningAgent] Initialized with {len(self.tools)} tools, "
            f"{len(self.managed_agents)} managed agents, "
            f"review_step={'on' if self.review_step is not None else 'off'}, "
            f"skill_registry={'on' if self.skill_registry is not None else 'off'}",
            level=LogLevel.INFO
        )

    def _build_review_step_from_config(self) -> Optional[ReviewStep]:
        """
        Consult `self.config.enable_review` and build a ReviewStep if set.

        Returns None when the flag is missing or False, so C0/C1 behaviour
        is unchanged. The `config` object may be an mmengine Config
        (attribute access) or a plain dict-like; we use `getattr` with a
        default so both work.
        """
        enable_review = getattr(self.config, "enable_review", False)
        if not enable_review:
            return None
        logger.log(
            "[AdaptivePlanningAgent] enable_review=True; building ReviewStep (C2)",
            level=LogLevel.INFO,
        )
        return ReviewStep(parent_agent=self)

    def _build_skill_registry_from_config(self) -> Optional[SkillRegistry]:
        """
        Consult `self.config.enable_skills` and build a SkillRegistry if set.

        Returns None when the flag is missing or False, so C0/C1/C2 behaviour
        is unchanged. Reads `config.skills_dir` for the on-disk location;
        defaults to `src/skills`.
        """
        enable_skills = getattr(self.config, "enable_skills", False)
        if not enable_skills:
            return None
        skills_dir = getattr(self.config, "skills_dir", "src/skills")
        skills_path = Path(skills_dir)
        logger.log(
            f"[AdaptivePlanningAgent] enable_skills=True; building SkillRegistry "
            f"at {skills_path} (C3)",
            level=LogLevel.INFO,
        )
        # Per-run seeding: when `skills_dir` is a fresh run directory
        # (e.g. `workdir/gaia_c3_<model>_<run_id>/skills`), the canonical
        # seed set from `src/skills/` is copied in on first construction.
        # A marker inside `skills_dir` makes subsequent constructions
        # (resume, same run id) a no-op so learned skills survive.
        # When `skills_dir` itself IS `src/skills/`, the marker-guard still
        # protects against accidental re-seed loops; the copy never fires
        # because every seed's destination already exists.
        seed_skills_dir(skills_path)
        return SkillRegistry(skills_path)

    def _install_skill_tools(self) -> None:
        """
        Install an `ActivateSkillTool` on the planner and each managed
        sub-agent. Each instance is bound to a specific `consumer` scope
        so the registry only exposes skills visible to that agent.

        Called ONCE at `__init__` time. Because the AdaptiveMixin's
        state-management captures sub-agent tools at the start of each
        `run()`, installing here (before the first `run`) means the
        tools persist across task resets — they are part of the "original
        state" the mixin restores to.

        The per-run `skill_registry_block` injection into task instructions
        is handled separately by `_refresh_skill_registry_blocks()`, which
        runs before every task so newly-extracted skills are visible.
        """
        assert self.skill_registry is not None  # caller guards this

        # Planner's own ActivateSkillTool (consumer = "planner")
        planner_tool = ActivateSkillTool(
            parent_agent=self,
            registry=self.skill_registry,
            consumer="planner",
        )
        self.tools[planner_tool.name] = planner_tool

        # Sub-agent tools, one per managed agent, bound to that agent's scope
        for sub_name, sub_agent in self.managed_agents.items():
            sub_tool = ActivateSkillTool(
                parent_agent=sub_agent,
                registry=self.skill_registry,
                consumer=sub_name,
            )
            # Respect whatever tools dict the sub-agent already has.
            if not hasattr(sub_agent, "tools") or sub_agent.tools is None:
                sub_agent.tools = {}
            sub_agent.tools[sub_tool.name] = sub_tool

        logger.log(
            f"[AdaptivePlanningAgent] installed activate_skill on planner + "
            f"{len(self.managed_agents)} sub-agents",
            level=LogLevel.INFO,
        )

    def _refresh_skill_registry_blocks(self) -> None:
        """
        Recompute the `skill_registry_block` for the planner and each
        sub-agent, and stash it in their `_extra_task_variables` dict so
        `GeneralAgent.initialize_task_instruction` picks it up at render
        time.

        Called before every `run()` (see the override below) because:
        - C3 training extracts new skills at task end, so the registry
          contents change between tasks.
        - The block is not refreshed mid-task — within a single task the
          consumer sees a fixed snapshot, avoiding distracting registry
          churn.
        """
        assert self.skill_registry is not None

        # Planner
        planner_extras = getattr(self, "_extra_task_variables", None) or {}
        planner_extras["skill_registry_block"] = (
            self.skill_registry.render_registry_block("planner")
        )
        self._extra_task_variables = planner_extras

        # Sub-agents
        for sub_name, sub_agent in self.managed_agents.items():
            sub_extras = getattr(sub_agent, "_extra_task_variables", None) or {}
            sub_extras["skill_registry_block"] = (
                self.skill_registry.render_registry_block(sub_name)
            )
            sub_agent._extra_task_variables = sub_extras
    
    def _add_adaptive_tools(self) -> None:
        """Add the diagnostic and modification tools."""
        # Add diagnose_subagent tool
        diagnose_tool = DiagnoseSubAgentTool(self)
        self.tools[diagnose_tool.name] = diagnose_tool
        
        # Add modify_subagent tool
        modify_tool = ModifySubAgentTool(self)
        self.tools[modify_tool.name] = modify_tool
        
        logger.log(
            "[AdaptivePlanningAgent] Added adaptive tools: diagnose_subagent, modify_subagent",
            level=LogLevel.DEBUG
        )
    
    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Run the agent on a task with state management.
        
        Stores original state before execution and restores it after,
        ensuring all modifications are task-scoped.
        
        Args:
            task: The task to execute
            stream: Whether to stream outputs
            reset: Whether to reset memory before running
            images: Optional images for the task
            additional_args: Additional arguments
            max_steps: Override for max steps
            
        Returns:
            Task result
        """
        # Store original state before task execution
        self._store_original_state()

        # Phase 1 (2026-04-26): per-task counter for consecutive same-agent
        # delegations. Used by _post_action_hook to inject a [PLANNER NOTE]
        # when the planner hammers the same sub-agent repeatedly. Resets on
        # every run() so the budget is task-scoped (matches the rest of
        # AdaptivePlanningAgent's state-reset semantics).
        self._consecutive_subagent_calls: tuple[str, int] = ("", 0)
        self._planner_note_emitted_for: set[str] = set()

        # C2/C3: initialise the ReviewStep per-task ledger. Idempotent;
        # clears any stale chain/blocklist/metrics state and captures the
        # original user task for inclusion in the reviewer's context.
        # No-op under C0/C1 (review_step is None). Metrics extracted by
        # run_gaia.py at the caller level after run() returns (see plan
        # §Layer 3 — finally-branch is unreliable under P1 semantics, so
        # we do NOT add a reset here in run()'s finally).
        if self.review_step is not None:
            self.review_step.on_task_start(task)

        # C3: refresh skill registry blocks so newly-extracted skills from
        # prior tasks are visible to the planner and sub-agents this run.
        # Must happen AFTER _store_original_state (which captures tools)
        # and BEFORE super().run() (which renders task_instruction).
        if self.skill_registry is not None:
            self._refresh_skill_registry_blocks()

        logger.log(
            f"[AdaptivePlanningAgent] Starting task with adaptive capabilities",
            level=LogLevel.INFO
        )

        try:
            # Execute the task using parent's run method
            result = await super().run(
                task=task,
                stream=stream,
                reset=reset,
                images=images,
                additional_args=additional_args,
                max_steps=max_steps,
            )

            # C3 training: extract a new skill from this trajectory if the
            # extractor is enabled and the pipeline decides it's worthwhile.
            # Runs BEFORE _reset_to_original_state so self.memory still has
            # the full trajectory. Never raises — returns None on failure.
            if self.skill_extractor is not None:
                await self._maybe_extract_skill(task, result)

            return result

        finally:
            # Always reset state after task, even on error
            self._reset_to_original_state()
            logger.log(
                "[AdaptivePlanningAgent] Task complete, state reset",
                level=LogLevel.DEBUG
            )

    async def _maybe_extract_skill(self, task: str, final_answer: Any) -> None:
        """
        Gather the inputs the SkillExtractor needs and invoke it.

        Pulls the last REVIEW verdict/root-cause from the most recent
        ActionStep (if REVIEW is active — C2+). Passes `task_success=None`
        because the planning agent has no access to a ground-truth scorer
        at this point; the extractor will fall back to the REVIEW verdict
        as its success signal.
        """
        assert self.skill_extractor is not None  # caller guards

        last_review_verdict: Optional[str] = None
        last_review_root_cause: Optional[str] = None

        # Walk memory.steps backwards for the most recent action step
        # with an attached review_result (set by _post_action_hook under C2).
        for step in reversed(getattr(self.memory, "steps", [])):
            review_result = getattr(step, "review_result", None)
            if review_result is not None:
                last_review_verdict = review_result.verdict
                if review_result.root_cause_primary is not None:
                    last_review_root_cause = review_result.root_cause_primary.value
                break

        try:
            await self.skill_extractor.extract_and_maybe_persist(
                task=task,
                final_answer=final_answer,
                task_success=None,
                final_review_verdict=last_review_verdict,
                final_review_root_cause=last_review_root_cause,
            )
        except Exception as e:
            # Defensive: SkillExtractor is contracted not to raise, but an
            # unexpected bug here must not break the planner's run.
            logger.log(
                f"[AdaptivePlanningAgent._maybe_extract_skill] extractor raised: "
                f"{type(e).__name__}: {e}",
                level=LogLevel.ERROR,
            )
    
    def initialize_system_prompt(self) -> str:
        """
        Initialize the system prompt for the agent.
        
        Overrides parent to ensure adaptive tools are properly included.
        
        Returns:
            The initialized system prompt string
        """
        from src.base.async_multistep_agent import populate_template
        
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents
            },
        )
        return system_prompt
    
    async def _post_action_hook(self, memory_step: ActionStep) -> None:
        """
        Fire the structural REVIEW step (C2) after each action step.

        Overrides the no-op hook in `AsyncMultiStepAgent`. If
        `self.review_step` is None (conditions C0/C1), this is effectively
        still a no-op — matching baseline behaviour.

        ReviewStep may:
          * Return None → the action step wasn't a sub-agent delegation;
            nothing to inject.
          * Return a ReviewResult → its rendered form is appended to
            `memory_step.observations` with a `[REVIEW]` marker, so the
            planner's next THINK sees the review in its input messages
            (via `ActionStep.to_messages()`).

        Errors in the review path are swallowed: we log and continue so
        that a broken reviewer cannot break the planner's run. This
        matches the contract documented on `AsyncMultiStepAgent._post_action_hook`.
        """
        # Phase 1 (2026-04-26): repeated-delegation guard. Independent of
        # ReviewStep — fires under C2/C3 alike (when review_step is non-None),
        # not under C1 (review disabled). The delegation counter below runs for
        # all adaptive conditions C1+ (anywhere this hook runs).
        # Counts consecutive delegations to the same sub-agent in this task;
        # once the threshold is reached, injects a one-shot [PLANNER NOTE]
        # into observations so the next THINK sees the warning.
        self._update_subagent_call_counter_and_maybe_inject_note(memory_step)

        if self.review_step is None:
            return

        try:
            review_result = await self.review_step.run_if_applicable(memory_step)
        except Exception as e:
            logger.log(
                f"[AdaptivePlanningAgent._post_action_hook] review failed "
                f"({type(e).__name__}: {e}); continuing without review.",
                level=LogLevel.ERROR,
            )
            return

        if review_result is None:
            return

        rendered = review_result.render()
        existing = memory_step.observations or ""
        memory_step.observations = (
            existing + ("\n\n" if existing else "") + f"[REVIEW]\n{rendered}"
        )
        # Attach the raw ReviewResult for downstream analysis (e.g. skill
        # extraction in Phase 2 / C3). Uses setattr because ActionStep is
        # a dataclass without this field defined.
        setattr(memory_step, "review_result", review_result)

    def _update_subagent_call_counter_and_maybe_inject_note(
        self, memory_step: ActionStep
    ) -> None:
        """Track consecutive same-sub-agent delegations and inject a
        [PLANNER NOTE] when the count reaches the threshold.

        Phase 1 mitigation for the failure mode where the planner re-delegates
        the same sub-agent indefinitely after errors. Threshold is 3 (≥3
        consecutive calls to the same sub-agent triggers the note). The note
        is injected once per (task, agent) pair to avoid spamming the
        context window.
        """
        THRESHOLD = 3

        tool_calls = getattr(memory_step, "tool_calls", None) or []
        managed = getattr(self, "managed_agents", None) or {}
        if not managed:
            return

        # Find the first sub-agent delegation in this step (planner may
        # parallel-call multiple agents, but we count the canonical "this
        # step delegated to X" event by the first match).
        sub_name: Optional[str] = None
        for tc in tool_calls:
            name = getattr(tc, "name", None)
            if name in managed:
                sub_name = name
                break

        if sub_name is None:
            # Non-delegation step → reset the streak.
            self._consecutive_subagent_calls = ("", 0)
            return

        prev_name, prev_count = self._consecutive_subagent_calls
        new_count = prev_count + 1 if prev_name == sub_name else 1
        self._consecutive_subagent_calls = (sub_name, new_count)

        if new_count < THRESHOLD:
            return
        if sub_name in self._planner_note_emitted_for:
            return

        note = (
            f"[PLANNER NOTE] You have delegated to '{sub_name}' "
            f"{new_count} times in a row. If the latest result did not "
            "give you new factual evidence, STOP delegating to this agent. "
            "Either (a) try a different sub-agent or a different "
            "decomposition of the question, or (b) commit to a concrete "
            "best-guess answer now via final_answer_tool. Do NOT reply "
            "'Unable to determine' — a wrong concrete guess is preferred "
            "over a refusal."
        )
        existing = memory_step.observations or ""
        memory_step.observations = (
            existing + ("\n\n" if existing else "") + note
        )
        self._planner_note_emitted_for.add(sub_name)
        logger.log(
            f"[AdaptivePlanningAgent] Injected [PLANNER NOTE] for "
            f"'{sub_name}' after {new_count} consecutive delegations.",
            level=LogLevel.INFO,
        )

    def get_adaptive_status(self) -> dict:
        """
        Get the current adaptive state for debugging/monitoring.
        
        Returns:
            Dictionary with current adaptive state information
        """
        return {
            "tools": list(self.tools.keys()),
            "managed_agents": list(self.managed_agents.keys()),
            "has_original_state": hasattr(self, '_original_state'),
            "managed_agent_tools": {
                name: list(agent.tools.keys()) if hasattr(agent, 'tools') else []
                for name, agent in self.managed_agents.items()
            }
        }
