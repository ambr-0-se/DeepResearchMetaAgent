---
name: delegation-failure-recovery
description: Recovery order for sub-agent delegations flagged `unsatisfactory` by REVIEW. Prefer retry with clearer instructions, then modify_subagent (tools or instructions), and only escalate to a different sub-agent as a last resort. Use when REVIEW returns verdict=unsatisfactory or when you see a sub-agent delegation fail twice in a row.
metadata:
  consumer: planner
  skill_type: failure_avoidance
  source: seeded
  verified_uses: 0
  confidence: 0.7
---

# Delegation Failure Recovery (planner workflow)

## When to activate
- REVIEW reported `verdict=unsatisfactory` on a just-completed delegation, OR
- A sub-agent response is empty / generic / "I couldn't" / contains only an error, OR
- You are about to call `diagnose_subagent` because a sub-agent failed in an unclear way.

## Recovery order (try in sequence, not all at once)

1. **Retry with clearer instructions**. Cheapest option — no structural change. Write a reformulated task that explicitly:
   - States the expected output format ("Return a number only", "Return JSON with keys X, Y").
   - Lists specific approaches to try first ("Use python_interpreter to compute X; if that fails, try method Y").
   - Names patterns to avoid ("Do not return 'I don't know' — make a best-effort guess based on available context").

2. **Modify the sub-agent's instructions** if retry fails. Use `modify_subagent` with `action=modify_agent_instructions` to append a persistent rule the sub-agent should always follow for this task type. Example: `"When analyzing financial data, always use python_interpreter to verify calculations and show intermediate steps."`

3. **Add a tool** if the failure was a capability gap (e.g. sub-agent couldn't do math because it didn't have `python_interpreter_tool`). Use `modify_subagent` with `action=add_existing_tool_to_agent`.

4. **Generate a new tool** as a last resort for novel capability gaps. Use `modify_subagent` with `action=add_new_tool_to_agent` + a natural-language spec. Only do this when no existing tool fits.

5. **Escalate to a different sub-agent** if the original sub-agent is fundamentally unsuited. Example: `browser_use_agent` is bad at reading scanned PDFs — switch to `deep_analyzer_agent`.

## Avoid
- Calling `modify_subagent` with `action=add_agent` for transient failures — creating new managed agents is expensive and they are reset after the task.
- Retrying with the SAME task text — that just repeats the failure.
- Escalating before trying retry + modify (wastes the planner's context accumulated so far).

## Budget note
- After 2 failed attempts on the same sub-question, stop retrying and accept a best-effort answer based on what you have. Do not let one sub-question consume all of max_steps.
