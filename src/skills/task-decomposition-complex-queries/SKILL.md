---
name: task-decomposition-complex-queries
description: Decomposition heuristics for tasks that require more than three distinct tool types or more than two sub-agent delegations. Use when a task spans research + analysis + verification, or mentions multiple entities that each need separate lookup. Produces a numbered plan with explicit per-step delegation targets before any execution.
metadata:
  consumer: planner
  skill_type: task_decomposition
  source: seeded
  verified_uses: 0
  confidence: 0.75
---

# Task Decomposition for Complex Queries (planner workflow)

## When to activate
- Task requires ≥ 3 distinct tool types (e.g. web search + calculation + image analysis), OR
- Task mentions ≥ 2 unrelated entities/topics that each need separate investigation, OR
- Task asks for comparison / ranking / aggregation across multiple sources, OR
- Task has a verification step ("compute X, then verify against Y").

## Workflow
1. **Enumerate the sub-questions** explicitly before any delegation. Use `planning_tool` (`create` with steps) to record them.
2. For EACH sub-question, name:
   - The sub-agent to delegate to (deep_analyzer / browser_use / deep_researcher)
   - The input (absolute path / URL / specific entity)
   - The expected output shape (number, list, boolean, structured data)
3. Check for dependencies between sub-questions. If sub-question B needs the answer from A, run them sequentially. If they are independent, they can be delegated in parallel (one per step).
4. After all sub-questions are resolved, aggregate results in a `deep_analyzer_agent` call with `python_interpreter_tool` (for calculations) or inline reasoning (for qualitative merging).
5. Run a final verification check against the original question text — confirm every part of the task is addressed before `final_answer`.

## Avoid
- Delegating the full complex task to a single sub-agent — they lack cross-domain orchestration.
- Skipping the plan step and delegating ad-hoc — this leads to missed sub-questions.
- Aggregating by concatenation when calculation is needed (use `python_interpreter_tool`).

## Example plan
Task: "Compare the Q3 revenue of Apple and Microsoft, expressed in the same currency."

Decomposition:
1. Find Apple Q3 revenue. → `deep_researcher_agent` (web search; likely in USD)
2. Find Microsoft Q3 revenue. → `deep_researcher_agent` (web search; likely in USD)
3. Aggregate + compare → `deep_analyzer_agent` with python (both already USD → simple diff; else currency conversion first)
