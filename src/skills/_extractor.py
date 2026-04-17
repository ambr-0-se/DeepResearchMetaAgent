"""
SkillExtractor — task-end pipeline that proposes new SKILL.md files from
successful/failed trajectories (condition C4 training loop).

Pipeline (all steps are conditional — any one may veto):

    worthiness_heuristic   — cheap pre-filter; skips trivial tasks.
           ↓
    LLM propose skill      — one call; returns YAML+body or {"skip": "..."}
           ↓
    entity blocklist       — regex; rejects skills that leaked proper nouns,
                              specific numbers, dates, URLs.
           ↓
    de-duplication check   — LLM-as-judge; rejects if near-duplicate of an
                              existing skill description.
           ↓
    registry.add           — atomic write + in-memory update.

Contract:
- Runs at task end, inside `AdaptivePlanningAgent.run()`'s finally block.
- Must not raise — a broken extractor must not break the task. Any failure
  is logged and the method returns None.
- Uses `parent_agent.model` for all LLM calls (same model as the planner).
- Gated by `config.enable_skill_extraction` — when False, the registry is
  frozen (test-time / eval behaviour).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.logger import LogLevel, logger
from src.models import ChatMessage, MessageRole
from src.skills._model import (
    CANONICAL_SKILL_TYPES,
    Skill,
    SkillMetadata,
    SKILL_DESCRIPTION_MAX_LEN,
    SKILL_NAME_MAX_LEN,
    SKILL_NAME_PATTERN,
)

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent
    from src.skills._registry import SkillRegistry


# --- Blocklist: specific entities that should NOT appear in a generalisable
#     skill. The point of a skill is to describe a PATTERN; patterns that
#     reference "Apple Inc." or "42" are just memorised answers and do not
#     transfer to held-out tasks.

_BLOCKLIST_PATTERNS: list[re.Pattern[str]] = [
    # Four-digit years (specific dates)
    re.compile(r"\b(19|20)\d{2}\b"),
    # Monetary amounts with specific currency symbols
    re.compile(r"[\$€£¥][\d,]+(\.\d+)?"),
    # Specific URLs (not generic site: patterns)
    re.compile(r"https?://\S+\.[a-z]{2,}/\S*"),
    # Long numeric sequences (e.g., specific identifiers, phone numbers)
    re.compile(r"\b\d{5,}\b"),
]

#: Minimum trajectory size for a task to be considered worth extraction.
#: Single-delegation tasks rarely teach a generalisable pattern.
_MIN_DELEGATIONS_FOR_SUCCESS_SKILL: int = 3

# Which root-cause categories are actionable enough to extract a
# failure-avoidance skill from. See review_schema.RootCauseCategory.
_ACTIONABLE_FAILURE_CAUSES: frozenset[str] = frozenset({
    "missing_tool",
    "wrong_tool",
    "bad_instruction",
    "misread_task",
})


# --- Prompts ----------------------------------------------------------------

_PROPOSE_SKILL_SYSTEM_PROMPT: str = """\
You are a SKILL EXTRACTOR. Your job is to review a completed task trajectory
and decide whether it reveals a generalisable PATTERN worth saving as a
reusable skill — or whether the trajectory is too task-specific to teach
anything transferable.

A good skill describes WHAT to do and WHEN, in terms that apply to a whole
CLASS of tasks. A bad "skill" just memorises specific facts, entities, or
numbers from THIS task.

OUTPUT FORMAT (strict JSON, no markdown fences):

To SKIP extraction, return:
{
  "skip": true,
  "reason": "<one sentence: why this trajectory is too specific / trivial / already covered>"
}

To PROPOSE a skill, return:
{
  "skip": false,
  "name": "<kebab-case, matches ^[a-z0-9]+(-[a-z0-9]+)*$, ≤64 chars>",
  "description": "<40-1024 chars; what the skill does AND when to use it. Written so another agent can decide whether to activate it. Third person, specific keywords.>",
  "consumer": "planner" | "deep_analyzer_agent" | "browser_use_agent" | "deep_researcher_agent" | "all",
  "skill_type": one of [
    "delegation_pattern", "task_decomposition", "failure_avoidance",
    "modification_pattern", "verification_pattern", "tool_usage",
    "domain_workflow"
  ],
  "body": "<Markdown body for SKILL.md. Include these sections:
    ## When to activate
    ## Workflow (numbered steps, each referencing concrete tools / agents)
    ## Avoid (list common failure modes this skill addresses)>"
}

STRICT RULES — skip extraction if ANY apply:

1. The trajectory is only 1-2 steps long. Trivial tasks teach trivial
   "skills" that just say "use the one tool you have".
2. The proposed skill would contain specific entity names (companies,
   people, places), specific numbers, specific dates, or specific URLs.
   Generalise or skip.
3. The proposed skill duplicates something that an experienced human
   would consider common sense (e.g. "use python to compute things").
4. The trajectory succeeded by accident, without following any identifiable
   pattern. You cannot extract a pattern from randomness.
5. The pattern works only on this one task. If you cannot describe a
   DIFFERENT task where the same pattern would apply, skip.

NEVER include in the body:
- Specific entity names, proper nouns, numeric answers, dates, URLs from
  this trajectory. Use placeholders like <file_path>, <entity>, <number> if
  illustrative examples are needed.
"""


_DEDUP_SYSTEM_PROMPT: str = """\
You are a DE-DUPLICATION judge. Given a proposed new skill and a list of
existing skill descriptions, decide whether the proposal is substantially
the same as an existing skill. If so, the new skill should NOT be added.

Return strict JSON (no markdown fences):
{
  "is_duplicate": true | false,
  "near_match": "<name of the closest existing skill, if any, else null>",
  "reason": "<one sentence>"
}

A proposal is a duplicate when it describes the same WORKFLOW + same WHEN
as an existing skill. Superficial wording differences do not count; same
intent + same trigger means duplicate.

A proposal is NOT a duplicate when it addresses a genuinely different
trigger or workflow, even if the topic overlaps slightly with an existing
skill.
"""


# --- Extractor --------------------------------------------------------------

class SkillExtractor:
    """
    End-of-task skill extraction pipeline.

    Wire one instance per AdaptivePlanningAgent (not registered anywhere;
    just held as an attribute). The extractor shares the planner's model
    for all LLM calls.
    """

    def __init__(
        self,
        parent_agent: "AsyncMultiStepAgent",
        registry: "SkillRegistry",
    ) -> None:
        self.parent = parent_agent
        self.registry = registry

    async def extract_and_maybe_persist(
        self,
        task: str,
        final_answer: Any,
        task_success: Optional[bool] = None,
        final_review_verdict: Optional[str] = None,
        final_review_root_cause: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        Run the extraction pipeline at the end of a task.

        Returns the newly-persisted Skill on success; None when the pipeline
        decided to skip (any pipeline stage may veto).

        Arguments:
            task — the original user task text.
            final_answer — whatever the planner produced as final_answer.
            task_success — external success signal (e.g. from the GAIA
                scorer) if available. When None, we infer from whether
                `final_answer` is non-empty and the last REVIEW was
                satisfactory.
            final_review_verdict — verdict of the LAST REVIEW step in the
                trajectory ("satisfactory" / "partial" / "unsatisfactory"),
                or None if REVIEW was not active.
            final_review_root_cause — root cause reported by the LAST failing
                REVIEW, as the string value of RootCauseCategory. Used by
                the worthiness filter to decide if a failure-avoidance
                skill is extractable.

        Never raises. Any failure is caught, logged, and returns None.
        """
        try:
            return await self._run_pipeline(
                task=task,
                final_answer=final_answer,
                task_success=task_success,
                final_review_verdict=final_review_verdict,
                final_review_root_cause=final_review_root_cause,
            )
        except Exception as e:
            logger.log(
                f"[SkillExtractor] pipeline failed ({type(e).__name__}: {e}); "
                f"no skill extracted",
                level=LogLevel.ERROR,
            )
            return None

    # -- pipeline stages -----------------------------------------------------

    async def _run_pipeline(
        self,
        *,
        task: str,
        final_answer: Any,
        task_success: Optional[bool],
        final_review_verdict: Optional[str],
        final_review_root_cause: Optional[str],
    ) -> Optional[Skill]:
        trajectory_summary = self._summarise_trajectory()

        # Stage 1: cheap worthiness heuristic.
        if not self._is_worthy_of_extraction(
            trajectory_summary=trajectory_summary,
            task_success=task_success,
            final_review_verdict=final_review_verdict,
            final_review_root_cause=final_review_root_cause,
        ):
            return None

        # Stage 2: LLM proposal.
        proposal = await self._llm_propose_skill(
            task=task,
            trajectory_summary=trajectory_summary,
            final_answer=final_answer,
            task_success=task_success,
            final_review_verdict=final_review_verdict,
            final_review_root_cause=final_review_root_cause,
        )
        if proposal is None or proposal.get("skip", True):
            reason = (proposal or {}).get("reason", "unknown")
            logger.log(
                f"[SkillExtractor] LLM proposer returned skip: {reason}",
                level=LogLevel.DEBUG,
            )
            return None

        # Stage 3: structural validation (name, description, body).
        validated = self._validate_proposal(proposal)
        if validated is None:
            return None
        name, description, consumer, skill_type, body = validated

        # Stage 4: entity blocklist.
        if self._contains_blocked_entities(description) or self._contains_blocked_entities(body):
            logger.log(
                f"[SkillExtractor] proposed skill '{name}' contains blocked entities; rejecting",
                level=LogLevel.WARNING,
            )
            return None

        # Stage 5: de-duplication against existing skills.
        if await self._is_duplicate_of_existing(name, description):
            return None

        # Stage 6: persist.
        metadata = SkillMetadata(
            name=name,
            description=description,
            consumer=consumer,
            skill_type=skill_type,
            source=self._infer_source(task_success, final_review_verdict),
            verified_uses=0,
            confidence=0.5,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            learned_from_task_type=None,  # could be populated from task heuristics later
        )
        skill = Skill(
            path=self.registry.skills_dir / name,
            metadata=metadata,
            body=body,
        )
        try:
            self.registry.add(skill)
        except ValueError as e:
            # Name collision (two tasks proposed the same name in the same run)
            logger.log(
                f"[SkillExtractor] could not add skill '{name}': {e}",
                level=LogLevel.WARNING,
            )
            return None
        logger.log(
            f"[SkillExtractor] extracted new skill: {name} (consumer={consumer})",
            level=LogLevel.INFO,
        )
        return skill

    def _summarise_trajectory(self) -> str:
        """
        Compact textual summary of the parent agent's memory.steps used in
        the extractor prompt. We reuse the same formatter as DiagnoseSubAgentTool
        / ReviewAgent for consistency.
        """
        from src.meta._memory_format import format_execution_history
        return format_execution_history(self.parent)

    def _is_worthy_of_extraction(
        self,
        *,
        trajectory_summary: str,
        task_success: Optional[bool],
        final_review_verdict: Optional[str],
        final_review_root_cause: Optional[str],
    ) -> bool:
        """
        Cheap pre-filter before any LLM call.

        Success-path extraction requires a non-trivial trajectory (≥ N
        delegations) to avoid extracting "use the one tool you have" as
        a skill.

        Failure-path extraction requires an ACTIONABLE root cause — external
        failures (network, paywall) and model limits don't yield useful
        generalisable patterns.
        """
        # Count sub-agent delegations by counting "Tool Called: <managed_agent_name>"
        # occurrences in the trajectory summary. Cheap and format-stable.
        managed_names = getattr(self.parent, "managed_agents", {})
        delegation_count = sum(
            trajectory_summary.count(f"Tool Called: {name}(")
            for name in managed_names
        )

        # Success branch
        is_success = (
            task_success is True
            or (task_success is None and final_review_verdict == "satisfactory")
        )
        if is_success:
            return delegation_count >= _MIN_DELEGATIONS_FOR_SUCCESS_SKILL

        # Failure branch — extract only if the cause is actionable
        if final_review_root_cause in _ACTIONABLE_FAILURE_CAUSES:
            return True

        return False

    async def _llm_propose_skill(
        self,
        *,
        task: str,
        trajectory_summary: str,
        final_answer: Any,
        task_success: Optional[bool],
        final_review_verdict: Optional[str],
        final_review_root_cause: Optional[str],
    ) -> Optional[dict]:
        """Single LLM call to propose a skill or skip."""
        user_content = (
            f"TASK:\n{task}\n\n"
            f"TRAJECTORY (compact memory of the solve):\n{trajectory_summary[:6000]}\n\n"
            f"FINAL ANSWER:\n{str(final_answer)[:1000]}\n\n"
            f"TASK SUCCESS SIGNAL: {task_success}\n"
            f"LAST REVIEW VERDICT: {final_review_verdict}\n"
            f"LAST REVIEW ROOT CAUSE: {final_review_root_cause}\n\n"
            "Propose a generalisable skill OR skip."
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_PROPOSE_SKILL_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]
        try:
            response = await self.parent.model(messages, stop_sequences=None)
        except Exception as e:
            logger.log(
                f"[SkillExtractor] LLM propose call failed: {type(e).__name__}: {e}",
                level=LogLevel.ERROR,
            )
            return None
        return _parse_json_response(response.content)

    def _validate_proposal(
        self, proposal: dict
    ) -> Optional[tuple[str, str, str, str, str]]:
        """
        Validate the LLM's proposal against the schema. Returns the
        five-tuple (name, description, consumer, skill_type, body) on
        success, None on any validation failure.
        """
        try:
            name = str(proposal["name"]).strip()
            description = str(proposal["description"]).strip()
            consumer = str(proposal.get("consumer", "planner")).strip()
            skill_type = str(proposal.get("skill_type", "delegation_pattern")).strip()
            body = str(proposal["body"]).strip()
        except KeyError as e:
            logger.log(
                f"[SkillExtractor] proposal missing required field: {e}",
                level=LogLevel.WARNING,
            )
            return None

        if len(name) == 0 or len(name) > SKILL_NAME_MAX_LEN:
            logger.log(f"[SkillExtractor] bad name length: {name!r}", level=LogLevel.WARNING)
            return None
        if not SKILL_NAME_PATTERN.match(name):
            logger.log(f"[SkillExtractor] name does not match regex: {name!r}", level=LogLevel.WARNING)
            return None
        if len(description) < 40 or len(description) > SKILL_DESCRIPTION_MAX_LEN:
            logger.log(
                f"[SkillExtractor] description length {len(description)} out of range",
                level=LogLevel.WARNING,
            )
            return None
        if skill_type not in CANONICAL_SKILL_TYPES:
            logger.log(
                f"[SkillExtractor] non-canonical skill_type {skill_type!r}; accepting anyway",
                level=LogLevel.DEBUG,
            )
            # Non-canonical types are accepted (warned at validate-time), per the model module.
        # Consumer must be "planner", "all", or an existing managed-agent name.
        valid_consumers = {"planner", "all"} | set(getattr(self.parent, "managed_agents", {}).keys())
        if consumer not in valid_consumers:
            logger.log(
                f"[SkillExtractor] unknown consumer {consumer!r}; valid: {sorted(valid_consumers)}",
                level=LogLevel.WARNING,
            )
            return None
        if len(body) < 50:
            logger.log(
                f"[SkillExtractor] body too short ({len(body)} chars)",
                level=LogLevel.WARNING,
            )
            return None

        return name, description, consumer, skill_type, body

    @staticmethod
    def _contains_blocked_entities(text: str) -> bool:
        """Return True if the text matches any entity blocklist pattern."""
        for pat in _BLOCKLIST_PATTERNS:
            if pat.search(text):
                return True
        return False

    async def _is_duplicate_of_existing(self, name: str, description: str) -> bool:
        """
        Ask the LLM whether this proposal duplicates an existing skill.

        Cheap pre-check: if a skill with the same name already exists, it's
        obviously a duplicate (don't even bother with the LLM call).
        """
        if self.registry.get(name) is not None:
            logger.log(
                f"[SkillExtractor] name collision with existing skill '{name}'",
                level=LogLevel.WARNING,
            )
            return True

        existing = self.registry.metadata_for("all") + [
            m for m in self.registry.metadata_for("planner")
        ]
        # Flatten and dedupe (metadata_for is consumer-filtered; we want ALL
        # existing skills for the duplicate check regardless of scope).
        seen_names = set()
        all_existing = []
        for m in (
            [m for m in (self.registry.get(n).metadata for n in self.registry.names())]
        ):
            if m.name not in seen_names:
                seen_names.add(m.name)
                all_existing.append(m)

        if not all_existing:
            return False

        existing_listing = "\n".join(f"- {m.name}: {m.description}" for m in all_existing)
        user_content = (
            f"PROPOSED:\n- {name}: {description}\n\n"
            f"EXISTING:\n{existing_listing}\n\n"
            "Is the proposed skill a duplicate of any existing one?"
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_DEDUP_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]
        try:
            response = await self.parent.model(messages, stop_sequences=None)
        except Exception as e:
            logger.log(
                f"[SkillExtractor] dedup LLM call failed: {type(e).__name__}: {e}; "
                f"treating as non-duplicate",
                level=LogLevel.WARNING,
            )
            return False
        parsed = _parse_json_response(response.content)
        if parsed is None:
            return False
        is_dup = bool(parsed.get("is_duplicate", False))
        if is_dup:
            logger.log(
                f"[SkillExtractor] duplicate detected: {name!r} near-match "
                f"{parsed.get('near_match')}; reason: {parsed.get('reason')}",
                level=LogLevel.INFO,
            )
        return is_dup

    @staticmethod
    def _infer_source(
        task_success: Optional[bool], final_review_verdict: Optional[str]
    ) -> str:
        """Return 'success' or 'failure' for the Skill's source metadata."""
        if task_success is True:
            return "success"
        if final_review_verdict == "satisfactory":
            return "success"
        return "failure"


# --- Helpers ----------------------------------------------------------------

def _parse_json_response(raw: Any) -> Optional[dict]:
    """
    Parse an LLM response (string or content list) into a dict.

    Handles the common case where the model wraps JSON in markdown fences.
    Returns None on any parse error rather than raising — extractor pipeline
    stages should treat unparsable responses as "skip".
    """
    if raw is None:
        return None
    # Anthropic-style content blocks: list of {"type": "text", "text": "..."}
    if isinstance(raw, list):
        text_parts = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        raw = "".join(text_parts)
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.log(
            f"[SkillExtractor] failed to parse LLM JSON: {e}; raw prefix: {text[:200]!r}",
            level=LogLevel.WARNING,
        )
        return None
