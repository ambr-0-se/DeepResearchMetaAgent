"""
Snapshot-style tests for the REVIEW system prompt.

Extracts the template constant directly from source (avoids triggering
the full import chain of review_agent / general_agent / agent.__init__)
and renders it via Jinja2 with representative variables.

Includes a prompt-length assertion as a bloat guard.
"""
import re
import sys
from pathlib import Path

from jinja2 import Template

root = str(Path(__file__).resolve().parents[1])


def _extract_template() -> str:
    """Pull REVIEW_AGENT_SYSTEM_PROMPT from the .py source as a string.

    This avoids importing src.meta.review_agent (which eagerly imports the
    general_agent chain). A plain regex lift is enough because the constant
    is a single triple-quoted string.
    """
    src = (Path(root) / "src" / "meta" / "review_agent.py").read_text()
    m = re.search(
        r'REVIEW_AGENT_SYSTEM_PROMPT:\s*str\s*=\s*"""\\\n(.*?)"""',
        src,
        re.DOTALL,
    )
    assert m, "REVIEW_AGENT_SYSTEM_PROMPT constant not found in review_agent.py"
    return m.group(1)


class _FakeTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description


def _render(sub_agent_catalog: str = "") -> str:
    tmpl = _extract_template()
    tools = {
        "diagnose_subagent": _FakeTool("diagnose_subagent", "inspect sub-agent history"),
        "final_answer_tool": _FakeTool("final_answer_tool", "return the ReviewResult JSON"),
    }
    return Template(tmpl).render(
        tools=tools,
        managed_agents={},
        sub_agent_catalog=sub_agent_catalog,
    )


# --- section presence -------------------------------------------------------

class TestPromptContentWithCatalog:
    def setup_method(self):
        self.rendered = _render(
            "* deep_analyzer_agent — analyses structured data\n"
            "  tools: [python_interpreter_tool, deep_analyzer_tool]\n"
            "* browser_use_agent — web navigation\n"
            "  tools: [auto_browser_use_tool]"
        )

    def test_has_sub_agent_catalog_header(self):
        assert "AVAILABLE SUB-AGENTS" in self.rendered

    def test_has_root_cause_advisory_table(self):
        assert "ROOT CAUSE → RECOMMENDED NEXT ACTION" in self.rendered
        # All 8 root causes listed
        for rc in ["missing_tool", "wrong_tool", "bad_instruction",
                   "misread_task", "unclear_goal", "incomplete",
                   "external", "model_limit"]:
            assert rc in self.rendered, f"missing root cause {rc} in prompt"

    def test_retry_unavailable_appears_on_cap0_rows_only(self):
        """Exactly 4 rows get `retry unavailable` — plus the 'retry =
        disallowed' directive below the table. Counted substring
        conservatively >= 4."""
        n = self.rendered.count("retry unavailable")
        assert n >= 4, f"expected retry unavailable >=4 times, got {n}"

    def test_all_three_worked_examples_present(self):
        assert "EXAMPLE 1 — modify_agent with add_existing_tool_to_agent" in self.rendered
        assert "EXAMPLE 2 — modify_agent with add_new_tool_to_agent" in self.rendered
        assert "EXAMPLE 3 — retry" in self.rendered

    def test_no_prefer_retry_antibias_line(self):
        """The old 'If unsure, prefer retry with clearer guidance over a
        speculative modify_agent' line must be gone."""
        assert "prefer 'retry' with clearer guidance" not in self.rendered

    def test_sub_agent_names_inlined(self):
        """Catalog content appears in the rendered output."""
        assert "deep_analyzer_agent" in self.rendered
        assert "browser_use_agent" in self.rendered
        assert "python_interpreter_tool" in self.rendered

    def test_prompt_length_under_budget(self):
        """Bloat guard — the reviewer has only 3 steps. A too-long prompt
        eats its own thinking budget."""
        assert len(self.rendered) < 12_000, f"prompt too long: {len(self.rendered)}"


class TestPromptContentWithoutCatalog:
    def setup_method(self):
        self.rendered = _render(sub_agent_catalog="")

    def test_no_sub_agent_catalog_references(self):
        """Empty catalog → all guarded sections omitted cleanly."""
        assert "AVAILABLE SUB-AGENTS" not in self.rendered
        assert "listed under AVAILABLE SUB-AGENTS" not in self.rendered

    def test_advisory_table_still_present(self):
        assert "ROOT CAUSE → RECOMMENDED NEXT ACTION" in self.rendered

    def test_examples_still_present(self):
        assert "EXAMPLE 1" in self.rendered
        assert "EXAMPLE 2" in self.rendered
        assert "EXAMPLE 3" in self.rendered

    def test_prompt_length_under_budget(self):
        assert len(self.rendered) < 12_000

    def test_shorter_than_with_catalog(self):
        longer = _render("* deep_analyzer_agent — desc\n  tools: [t]")
        assert len(self.rendered) <= len(longer)
