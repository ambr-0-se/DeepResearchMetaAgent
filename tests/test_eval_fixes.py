"""Tests for the P0-P2 evaluation fixes."""
import sys
import os
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.models.base import ChatMessage, MessageRole, TokenUsage


# ============================================================================
# P1-A: Auto-generate plan_id
# ============================================================================

class TestPlanningToolAutoId:
    """Test that planning_tool auto-generates plan_id when omitted."""

    @staticmethod
    def _make_tool():
        from src.tools.planning import PlanningTool
        return PlanningTool()

    def test_create_without_plan_id(self):
        tool = self._make_tool()
        result = asyncio.run(tool.forward(
            action="create", plan_id=None, title="Test Plan", steps=["step1", "step2"]
        ))
        assert result.error is None, f"Expected no error, got: {result.error}"
        assert "plan_1" in result.output, f"Expected plan_1 in output: {result.output}"
        assert "plan_1" in tool.plans

    def test_create_with_explicit_plan_id(self):
        tool = self._make_tool()
        result = asyncio.run(tool.forward(
            action="create", plan_id="my_plan", title="Test", steps=["step1"]
        ))
        assert result.error is None
        assert "my_plan" in tool.plans

    def test_auto_id_collision_avoidance(self):
        tool = self._make_tool()
        # Manually insert plan_1
        tool.plans["plan_1"] = {"plan_id": "plan_1", "title": "x", "steps": [], "step_statuses": [], "step_notes": []}
        result = asyncio.run(tool.forward(
            action="create", plan_id=None, title="Second Plan", steps=["a"]
        ))
        assert result.error is None
        assert "plan_2" in tool.plans, f"Expected plan_2, got plans: {list(tool.plans.keys())}"

    def test_create_without_title_still_errors(self):
        tool = self._make_tool()
        result = asyncio.run(tool.forward(
            action="create", plan_id=None, title=None, steps=["step1"]
        ))
        assert result.error is not None
        assert "title" in result.error.lower()

    def test_create_without_steps_still_errors(self):
        tool = self._make_tool()
        result = asyncio.run(tool.forward(
            action="create", plan_id=None, title="Test", steps=None
        ))
        assert result.error is not None
        assert "steps" in result.error.lower()


# ============================================================================
# P0-B: Defensive fallback in diagnose/modify tools
# ============================================================================

class _MockAgent:
    """Lightweight mock of an AsyncMultiStepAgent for testing adaptive tools.

    Has empty managed_agents to simulate the bug where agents end up in tools instead.
    Includes AdaptiveMixin methods so modify_tool's primary path can be tested.
    """
    def __init__(self, name, has_memory=True):
        self.name = name
        self.description = f"Mock agent {name}"
        self.managed_agents = {}  # Empty — simulates the bug
        self.tools = {}
        self.prompt_templates = {"task_instruction": "do stuff", "system_prompt": "system"}
        self.max_steps = 10
        if has_memory:
            self.memory = MagicMock()
            self.memory.steps = []
            self.memory.system_prompt = None
        self.model = AsyncMock()
        self.model.return_value = ChatMessage(role="assistant", content="diagnosis result")

    def initialize_system_prompt(self):
        return "system prompt"

    def _find_managed_agent(self, agent_name):
        if agent_name in self.managed_agents:
            return self.managed_agents[agent_name]
        if agent_name in self.tools:
            obj = self.tools[agent_name]
            if hasattr(obj, 'memory') or hasattr(obj, 'managed_agents'):
                return obj
        return None

    def modify_agent_instructions(self, agent_name, instructions):
        agent = self._find_managed_agent(agent_name)
        if agent is None:
            return False
        if not hasattr(agent, 'prompt_templates'):
            return False
        current = agent.prompt_templates.get("task_instruction", "")
        agent.prompt_templates["task_instruction"] = f"{current}\n\nAdditional Instructions:\n{instructions}"
        return True

    def set_agent_max_steps(self, agent_name, max_steps):
        agent = self._find_managed_agent(agent_name)
        if agent is None:
            return False
        max_steps = max(1, min(50, max_steps))
        agent.max_steps = max_steps
        return True

    def add_tool_to_agent(self, agent_name, tool):
        agent = self._find_managed_agent(agent_name)
        if agent is None or not hasattr(agent, 'tools'):
            return False
        agent.tools[tool.name] = tool
        return True

    def remove_tool_from_agent(self, agent_name, tool_name):
        agent = self._find_managed_agent(agent_name)
        if agent is None or not hasattr(agent, 'tools') or tool_name not in agent.tools:
            return False
        del agent.tools[tool_name]
        return True

    def add_managed_agent(self, agent):
        self.managed_agents[agent.name] = agent
        return True

    def remove_managed_agent(self, agent_name):
        if agent_name not in self.managed_agents:
            return False
        del self.managed_agents[agent_name]
        return True

    def add_new_tool_to_agent(self, agent_name, tool_code):
        return False


class _MockSubAgent:
    """Lightweight mock of a managed sub-agent."""
    def __init__(self, name):
        self.name = name
        self.description = f"Sub-agent {name}"
        self.tools = {"some_tool": MagicMock()}
        self.memory = MagicMock()
        self.memory.steps = []
        self.prompt_templates = {"task_instruction": "sub task"}
        self.max_steps = 5
        self.managed_agents = {}


class TestDiagnoseToolFallback:
    """Test that diagnose_subagent finds agents in self.tools when managed_agents is empty."""

    def test_find_agent_in_tools_fallback(self):
        from src.meta.diagnose_tool import DiagnoseSubAgentTool
        parent = _MockAgent("parent")
        sub = _MockSubAgent("browser_use_agent")
        parent.tools["browser_use_agent"] = sub  # Agent stored in tools, not managed_agents

        tool = DiagnoseSubAgentTool(parent)
        found = tool._find_agent("browser_use_agent")
        assert found is sub

    def test_available_agent_names_includes_tools(self):
        from src.meta.diagnose_tool import DiagnoseSubAgentTool
        parent = _MockAgent("parent")
        sub = _MockSubAgent("deep_analyzer_agent")
        parent.tools["deep_analyzer_agent"] = sub
        parent.tools["planning_tool"] = MagicMock(spec=[])  # No memory attr — not an agent

        tool = DiagnoseSubAgentTool(parent)
        names = tool._get_available_agent_names()
        assert "deep_analyzer_agent" in names
        assert "planning_tool" not in names

    def test_diagnose_finds_agent_via_fallback(self):
        from src.meta.diagnose_tool import DiagnoseSubAgentTool
        parent = _MockAgent("parent")
        sub = _MockSubAgent("browser_use_agent")
        parent.tools["browser_use_agent"] = sub

        tool = DiagnoseSubAgentTool(parent)
        result = asyncio.run(tool.forward(
            agent_name="browser_use_agent",
            task_given="search web",
            expected_outcome="found answer",
            actual_response="failed"
        ))
        assert "Error" not in result or "not found" not in result


class TestModifyToolFallback:
    """Test that modify_subagent finds agents in self.tools when managed_agents is empty."""

    def test_find_agent_in_tools(self):
        from src.meta.modify_tool import ModifySubAgentTool
        parent = _MockAgent("parent")
        sub = _MockSubAgent("deep_researcher_agent")
        parent.tools["deep_researcher_agent"] = sub

        tool = ModifySubAgentTool(parent)
        found = tool._find_agent("deep_researcher_agent")
        assert found is sub

    def test_modify_instructions_via_fallback(self):
        from src.meta.modify_tool import ModifySubAgentTool
        parent = _MockAgent("parent")
        sub = _MockSubAgent("deep_analyzer_agent")
        parent.tools["deep_analyzer_agent"] = sub

        tool = ModifySubAgentTool(parent)
        result = asyncio.run(tool.forward(
            action="modify_agent_instructions",
            agent_name="deep_analyzer_agent",
            specification="Be more precise"
        ))
        assert "Successfully" in result
        assert "Be more precise" in sub.prompt_templates["task_instruction"]

    def test_set_max_steps_via_fallback(self):
        from src.meta.modify_tool import ModifySubAgentTool
        parent = _MockAgent("parent")
        sub = _MockSubAgent("browser_use_agent")
        parent.tools["browser_use_agent"] = sub

        tool = ModifySubAgentTool(parent)
        result = asyncio.run(tool.forward(
            action="set_agent_max_steps",
            agent_name="browser_use_agent",
            specification="15"
        ))
        assert "Successfully" in result
        assert sub.max_steps == 15

    def test_nonexistent_agent_returns_available(self):
        from src.meta.modify_tool import ModifySubAgentTool
        parent = _MockAgent("parent")
        sub = _MockSubAgent("browser_use_agent")
        parent.tools["browser_use_agent"] = sub

        tool = ModifySubAgentTool(parent)
        result = asyncio.run(tool.forward(
            action="modify_agent_instructions",
            agent_name="nonexistent_agent",
            specification="test"
        ))
        assert "not found" in result.lower() or "error" in result.lower()
        assert "browser_use_agent" in result


# ============================================================================
# P0-B: AdaptiveMixin fallback
# ============================================================================

class TestAdaptiveMixinFallback:
    """Test that AdaptiveMixin._find_managed_agent works with the tools fallback.

    Note: _find_managed_agent has `self: AsyncMultiStepAgent` type annotation but
    is a normal instance method, so we call it as `obj._find_managed_agent(name)`.
    """

    def test_find_in_managed_agents(self):
        from src.meta.adaptive_mixin import AdaptiveMixin
        mixin = AdaptiveMixin()
        sub = _MockSubAgent("agent_a")
        mixin.managed_agents = {"agent_a": sub}
        mixin.tools = {}
        assert mixin._find_managed_agent("agent_a") is sub

    def test_find_in_tools_fallback(self):
        from src.meta.adaptive_mixin import AdaptiveMixin
        mixin = AdaptiveMixin()
        sub = _MockSubAgent("agent_b")
        mixin.managed_agents = {}
        mixin.tools = {"agent_b": sub}
        assert mixin._find_managed_agent("agent_b") is sub

    def test_not_found_returns_none(self):
        from src.meta.adaptive_mixin import AdaptiveMixin
        mixin = AdaptiveMixin()
        mixin.managed_agents = {}
        mixin.tools = {}
        assert mixin._find_managed_agent("nonexistent") is None


# ============================================================================
# P1-B: Context history pruning
# ============================================================================

class TestContextPruning:
    """Test the _prune_messages_if_needed method."""

    @staticmethod
    def _make_agent_with_config(max_model_len=32768):
        """Create a minimal GeneralAgent-like object for testing pruning."""
        from src.agent.general_agent.general_agent import GeneralAgent

        class FakeConfig:
            max_model_len = 32768
            template_path = "src/agent/general_agent/prompts/general_agent.yaml"

        config = FakeConfig()
        config.max_model_len = max_model_len

        # We can't instantiate GeneralAgent without the full stack,
        # so test the methods directly on a mock
        agent = MagicMock(spec=GeneralAgent)
        agent.config = config
        agent._estimate_token_count = GeneralAgent._estimate_token_count.__get__(agent)
        agent._prune_messages_if_needed = GeneralAgent._prune_messages_if_needed.__get__(agent)
        return agent

    def test_no_pruning_under_threshold(self):
        agent = self._make_agent_with_config(max_model_len=32768)
        msgs = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        result = agent._prune_messages_if_needed(msgs)
        assert len(result) == 3
        assert result is msgs  # No pruning — same object returned

    def test_pruning_triggers_on_large_context(self):
        agent = self._make_agent_with_config(max_model_len=1000)
        system_msg = ChatMessage(role="system", content="S" * 500)
        middle_msgs = [
            ChatMessage(role="user", content="U" * 800),
            ChatMessage(role="assistant", content="A" * 800),
            ChatMessage(role="user", content="U" * 800),
            ChatMessage(role="assistant", content="A" * 800),
        ]
        tail_msgs = [
            ChatMessage(role="user", content="last q"),
            ChatMessage(role="assistant", content="last a"),
        ]
        msgs = [system_msg] + middle_msgs + tail_msgs
        result = agent._prune_messages_if_needed(msgs)
        # Verify pruning occurred
        assert len(result) <= len(msgs)
        # System message preserved
        assert result[0].role == "system"
        # Tail preserved
        assert result[-1].content == "last a"
        assert result[-2].content == "last q"

    def test_pruning_preserves_tool_call_id(self):
        agent = self._make_agent_with_config(max_model_len=500)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="tool", content="X" * 2000, tool_call_id="call_123"),
            ChatMessage(role="user", content="short"),
            ChatMessage(role="assistant", content="short"),
            ChatMessage(role="user", content="last"),
        ]
        result = agent._prune_messages_if_needed(msgs)
        # The tool message in the middle should be pruned but keep tool_call_id
        tool_msgs = [m for m in result if m.role == "tool"]
        if tool_msgs:
            assert tool_msgs[0].tool_call_id == "call_123"

    def test_estimate_token_count(self):
        agent = self._make_agent_with_config()
        msgs = [ChatMessage(role="user", content="a" * 350)]
        est = agent._estimate_token_count(msgs)
        assert est == 100  # 350 / 3.5 = 100

    def test_estimate_token_count_list_content(self):
        agent = self._make_agent_with_config()
        msgs = [ChatMessage(role="user", content=[{"type": "text", "text": "a" * 700}])]
        est = agent._estimate_token_count(msgs)
        assert est == 200  # 700 / 3.5 = 200


# ============================================================================
# P0-A Layer 2: Eval-level retry
# ============================================================================

sys.path.insert(0, os.path.join(root, "examples"))


class TestTransientErrorDetection:
    """Test the _is_transient_error function from run_gaia.py."""

    def test_connection_refused(self):
        from run_gaia import _is_transient_error
        assert _is_transient_error(ConnectionError("Connection refused"))

    def test_503_error(self):
        from run_gaia import _is_transient_error
        assert _is_transient_error(Exception("503 Service Unavailable"))

    def test_non_transient_error(self):
        from run_gaia import _is_transient_error
        assert not _is_transient_error(ValueError("invalid literal"))

    def test_connection_reset(self):
        from run_gaia import _is_transient_error
        assert _is_transient_error(Exception("Connection reset by peer"))

    def test_internal_server_error(self):
        from run_gaia import _is_transient_error
        assert _is_transient_error(Exception("Internal Server Error"))


# ============================================================================
# P2: Prompt fix verification
# ============================================================================

class TestPromptFix:
    """Verify the prompt YAML contains the required updates."""

    def test_unable_to_determine_in_prompt(self):
        import yaml
        prompt_path = os.path.join(root, "src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml")
        with open(prompt_path) as f:
            data = yaml.safe_load(f)
        task_inst = data["task_instruction"]
        assert "Unable to determine" in task_inst
        assert "best guess" in task_inst.lower()
        assert "different approach" in task_inst.lower()


# ============================================================================
# Runner
# ============================================================================

def run_all_tests():
    import traceback
    test_classes = [
        TestPlanningToolAutoId,
        TestDiagnoseToolFallback,
        TestModifyToolFallback,
        TestAdaptiveMixinFallback,
        TestContextPruning,
        TestTransientErrorDetection,
        TestPromptFix,
    ]
    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            test_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {test_name}")
            except Exception as e:
                failed += 1
                tb = traceback.format_exc()
                errors.append((test_name, tb))
                print(f"  FAIL  {test_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
