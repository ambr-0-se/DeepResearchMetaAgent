# Experiment Findings — APAI4799 MetaAgent

_Started: 2026-04-21. Living document — append new sections as experiments
complete; do not rewrite prior entries._

This is the canonical record of findings, takeaways, caveats, and
paper-relevant data points across all experiment phases (I-track smokes,
E0 training, E1 snapshot, E2 freeze, E3 test, plus any ad-hoc runs). It
feeds the paper's methodology, results, limitations, and discussion
sections — anyone writing `methodology.tex`, `experiments.tex`,
`discussion.tex`, or `limitations.tex` should start here.

**This document is NOT:**
- A status log. For "what is running right now" see `workdir/E0_STATE.md`
  or equivalent phase-state docs.
- A plan. For "what we intend to do next" see
  `docs/report_source/project_state_gaia.md` and `project_state_arc.md`.
- A handoff. For cross-session continuity see `docs/handoffs/`.

**How to use this file:**
- Read top-down for a snapshot of everything known to be paper-relevant.
- Each experiment phase gets its own top-level `##` section.
- Cross-cutting insights (bugs, methodology corrections, scorer changes,
  etc.) go under dedicated top-level sections labelled *"Cross-cutting"*.
- When an experiment phase completes, append — do not overwrite. Prior
  phases' findings are themselves citable artefacts.

**Top-level table of contents**

- [§ E0 v3 (GAIA C3 training, 80-Q validation subsample, 2026-04-20)](#e0-v3--gaia-c3-training-80-q-validation-subsample-2026-04-20)
- _[E1 snapshot — pending]_
- _[E2 freeze-smoke — pending]_
- _[E3 test-split submission — pending]_
- _[ARC track — pending merge from Liangrui]_
- [§ Cross-cutting — scorer parity](#cross-cutting--scorer-parity-with-hf-leaderboard)
- [§ Cross-cutting — data-logging bug on timeout](#cross-cutting--intermediate_steps-discarded-on-timeout-path)
- [§ Cross-cutting — known GAIA ground-truth errors](#cross-cutting--known-gaia-ground-truth-errors)

---

## E0 v3 — GAIA C3 training, 80-Q validation subsample, 2026-04-20

_Created: 2026-04-21_
_Authoritative source: `workdir/gaia_c4_{mistral,qwen}_20260420_E0v3/` — **paper condition C3** (skill library + extraction). The `gaia_c4_*` directory prefix predates the May 2026 rename to contiguous **C0–C3**; new runs use `workdir/gaia_c3_*`._

**Run identity**

| Field | Value |
|---|---|
| Condition | **C3** training (enable_review=True, enable_skills=True, enable_skill_extraction=True) |
| Split | GAIA validation |
| Sample | random 80 of 165 questions, seed=42 (`GAIADataset.shuffle` + `max_samples=80`) |
| Concurrency | 4 (asyncio.gather batches) |
| Per-Q timeout | 1800 s |
| Planner max steps | 15; sub-agents 3; auto_browser_use 10; deep_researcher time_limit 45 s |
| Models | Mistral Small 4 (via OpenRouter → Mistral), Qwen 3.6 Plus (via `or-qwen3.6-plus`) |

### E0.1 Accuracy and error decomposition

Scoring via `question_scorer` (byte-level port of the official HF leaderboard scorer).

| Model | Correct / Scorable | Correct / 80 | Wrong | Gave-up (`"Unable to determine"`) | Errors |
|---|---|---|---|---|---|
| Mistral | **11/33 (33.3 %)** | 13.75 % | 14 | 8 | 47 (46 timeout + 1 HTTP 400) |
| Qwen | **16/26 (61.5 %)** | 20.0 % | 10 | 0 | 54 (all timeout) |

**By level**

| Level | Mistral correct/scorable | Qwen correct/scorable |
|---|---|---|
| L1 | 7/16 (43.8 %) | 8/10 (80.0 %) |
| L2 | 3/11 (27.3 %) | 7/14 (50.0 %) |
| L3 | 1/6 (16.7 %) | 1/2 (50.0 %) |

**Caveat:** these numbers are **not a C3-vs-baseline comparison**. E0 is a
training phase; C0/C1/C2 on the same validation subsample are not run. The
C3-over-C2 delta is decided by E3 on the test split.

### E0.2 Skill library — loading, retrieval, learning

**Pipeline health (verified end-to-end):**
- `SkillRegistry._scan` runs at every task start; registry size grows
  monotonically with learned skills: Mistral 7 → 13, Qwen 7 → 9.
- `ActivateSkillTool` is installed on the planner + 3 sub-agents at startup
  (log: `installed activate_skill on planner + 3 sub-agents`).
- `render_registry_block` produces one-line-per-skill prompt injection,
  consumer-scoped (log grep of late-run blocks confirms learned entries
  are present by name + description).
- `SkillExtractor` runs end-of-task; persists via atomic temp+rename.

**Activations observed**

Planner scope (from `dra.jsonl` tool_calls):

| Scope | Mistral | Qwen |
|---|---|---|
| Planner total calls / tasks | 60 / 26 (32.5 % of tasks) | 10 / 10 (12.5 %) |
| Top activated | `handling-file-attachments` 40, `task-decomposition-complex-queries` 18, `delegation-failure-recovery` 2 | `handling-file-attachments` 7, `escalate-to-deep-research-on-browser-failure` 2, `research-fallback-sources` 1 |

Sub-agent scope (from `log.txt` — not in `dra.jsonl.intermediate_steps`):

| Scope | Mistral | Qwen |
|---|---|---|
| Sub-agent total | 57 calls | 18 calls |
| Top activated | `multi-hop-math-verification` 27, `research-fallback-sources` 15, `browser-paywall-recovery` 13, `pdf-table-extraction` 2 | `browser-paywall-recovery` 7, `pdf-table-extraction` 5, `multi-hop-math-verification` 5, `research-fallback-sources` 1 |

**Learned-skill retrieval rate is essentially zero:**

| Model | Learned skill | Activations |
|---|---|---:|
| Mistral | `verify-non-commutativity-of-binary-operation` | 0 |
| Mistral | `compare-architecture-layer-counts` | 0 |
| Mistral | `estimate-false-positives-from-p-value-threshold` | 0 |
| Mistral | `find-min-ratio-entity` | 0 |
| Mistral | `sum-attributes-from-structured-data` | 0 |
| Mistral | `count-keyword-occurrences-in-document` | 0 |
| Mistral | `grid-based-region-traversal-check` | 0 |
| Qwen | `transit-hop-counting` | 0 |
| Qwen | `escalate-to-deep-research-on-browser-failure` | **3** |

**Takeaway for paper:** the library pipeline is mechanically correct
(persist → reload → inject → activate). What is weak is the model's
*selection* of learned entries against seeded ones. Plausible causes are
(a) narrower / more task-specific learned titles, (b) late availability
(skills extracted mid-run have fewer retrieval chances), (c) prompt-position
anchoring of seed skills. **Any C3-over-C2 delta on E3 test split must be
attributed predominantly to *seed skills*, not to the extracted library.**
Framing for the paper: E0 demonstrates end-to-end extractor functioning;
test-time generalisation of learned entries is a separate question that
E3 will not definitively answer for this library size.

### E0.3 Timeout attribution — systematic pass

Tool: [`scripts/timeout_analysis.py`](../../scripts/timeout_analysis.py).
Slices `log.txt` by timestamp into concurrency-4 batches (concurrent
tasks within a batch share interleaved log content, so attribution is
per-batch and per-task attribution is inherited from the batch).

**Provider routing (important for interpreting the signatures):**
- Mistral runs via native **Mistral La Plateforme** (`https://api.mistral.ai/v1`),
  registered as `mistral-small` → `mistral-small-2603`.
- Qwen runs via **OpenRouter** (`or-qwen3.6-plus`), not DashScope direct.
- _Earlier draft of this section reversed these and treated "429" bursts
  as OpenRouter rate-limiting on Mistral. That was wrong — see §E0.3.d
  for the false-positive root cause and the corrected numbers below._

#### E0.3.a Per-task attribution summary (corrected 2026-04-21)

**Mistral (46 timeouts, 22 batches)**

| Bucket | Timeouts | Share |
|---|---:|---:|
| review_retry_loop | 46 | **100.0 %** |

Every single Mistral timeout batch is dominated by REVIEW-agent retry
loops. Real HTTP 429 events against native Mistral: only 2 across the
entire run. Provider-reset events (upstream connect / reset reason:
overflow) contribute a secondary 35 events across 22 timeout batches
(≈ 1.6/batch).

**Qwen (54 timeouts, 20 batches)**

| Bucket | Timeouts | Share |
|---|---:|---:|
| review_retry_loop | 14 | 25.9 % |
| mixed(review_retry_loop + dr_query_timeout) | 13 | 24.1 % |
| mixed(dr_query_timeout + dr_summary_timeout) | 10 | 18.5 % |
| mixed(review_retry_loop + dr_summary_timeout) | 7 | 13.0 % |
| mixed(dr_summary_timeout + review_retry_loop) | 4 | 7.4 % |
| mixed(dr_query_timeout + review_retry_loop) | 2 | 3.7 % |
| mixed(dr_summary_timeout + dr_query_timeout) | 2 | 3.7 % |
| dr_query_timeout | 2 | 3.7 % |

Real HTTP 429 events against Qwen via OpenRouter: **zero**. Real 5xx
events: zero. The Qwen failure story is purely REVIEW retry loops +
DeepResearchTool cumulative 60 s sub-timeouts. Our earlier "Qwen is
rate-limited" narrative was a false positive (§E0.3.d).

#### E0.3.b Aggregated signature rates inside timeout-containing batches (corrected)

**Mistral (per-minute)**

| Signature | Count | per_min |
|---|---:|---:|
| review_retry_loop | 730 | 1.23 |
| provider_reset (upstream connect / reset reason: overflow) | 35 | 0.06 |
| rate_limit_429 (true HTTP 429, not skill-prompt text) | 2 | ≈ 0 |
| dr_query_timeout / dr_analyze_timeout | 2 | ≈ 0 |

**Qwen (per-minute)**

| Signature | Count | per_min |
|---|---:|---:|
| review_retry_loop | 210 | 0.34 |
| dr_query_timeout | 152 | 0.25 |
| dr_summary_timeout | 116 | 0.19 |
| dr_analyze_timeout | 44 | 0.07 |
| rate_limit_429 / gateway_5xx | 0 | 0 |

**Takeaway (corrected):** both models are bottlenecked by *internal agent
behaviour*, not by provider rate-limiting.
- **Mistral** (native La Plateforme): REVIEW-retry loops are the singular
  dominant failure. 1.23 retry verdicts/min sustained across 22 timeout
  batches. Real provider rate-limit pressure is negligible. A small
  secondary signal (≈ 0.06/min) is upstream gateway resets from
  La Plateforme — real but not primary.
- **Qwen** (OpenRouter): REVIEW-retry loops + cumulative
  DeepResearchTool 60 s sub-timeouts compound. Qwen uses DR much more
  than Mistral does, so the 60 s internal caps fire repeatedly. On task
  f46b4380 alone, ≥ 18 sub-timeouts ≈ 1,080 s of pure fallback wait
  inside the 1800 s per-Q budget.

#### E0.3.c Case studies (5 per model)

Numbers below are retries whose `revised_task` text contained keywords
from the task's question (so attribution to the specific task is robust
even under batch interleaving).

**Mistral**

| task_id | question gist | true | retry_loop matches | agents retried | distinct phrasings |
|---|---|---|---:|---|---:|
| 023e9d44 | Cross-country drive CA → ME (multi-hop geography) | 8 | 0 | — | — |
| 71345b0a | 2008 leap-day Wikipedia Dragon page joke | "Here be dragons" | 3 | browser×2, deep_researcher×1 | 3 |
| ed58682d | King of Pop, 5th single from 6th album, last word before 2nd chorus | stare | 3 | deep_researcher×1 | 2 |
| ad2b4d70 | Eva Draconis YouTube personal site symbol meaning | "War is not here…" | **7** | **browser×4** | **5** |
| 87c610df | April 1977 PM of first place mentioned in Esther | Morarji Desai | 5 | analyzer×2, browser×2 | 4 |

**Qwen**

| task_id | question gist | true | retry_loop matches | agents retried | distinct phrasings |
|---|---|---|---:|---|---:|
| f46b4380 | CA→ME drive, bottle refunds per state (same as Mistral 023e9d44 peer) | Harbinger, Tidal | 6 | — | 5 |
| d0633230 | scikit-learn July 2017 changelog bug fix | BaseLabelPropagation | 4 | — | 2 |
| 0383a3ee | BBC Earth Silliest Animal video bird species | Rockhopper penguin | **8** | browser×1, deep_researcher×1 | **7** |
| 872bfbb1 | 2008 painting "Embroidery from Uzbekistan" fruit × Oct 1949 breakfast | pears, bananas | 2 | — | 2 |
| b2c257e0 | butterfat above/below US standard | +4.6 | 2 | — | 2 |

**Observation:** the worst offenders are video/image/web-scrape tasks
(Eva Draconis symbol, BBC Earth video, Wikipedia Dragon history). The
REVIEW step re-phrases the same delegation 5-7 times against the same
sub-agent because the bottleneck is external content access, not prompt
clarity. Rephrasing doesn't fix paywalls or YouTube transcript extraction.

#### E0.3.d Signature false-positive post-mortem (correction log)

An earlier version of §E0.3.a reported "rate_limit_429 dominant" for
48 % of Mistral timeouts and "rate_limit_429 + review_retry_loop mixed"
for 46 % of Qwen timeouts. That conclusion was wrong and has been
replaced by the corrected tables above.

**Root cause:** the `rate_limit_429` signature used a bare `\b429\b`
regex. One of the seeded skills (`browser-paywall-recovery`) contains
prompt text like *"on a 429 error, try archive.org"*. Every rendering
of that skill into a prompt block matched the signature, even when no
HTTP 429 ever occurred. Similarly, bare `\b(502|503|504)\b` matched
occasional digit tokens in tool arguments and skill text.

**Corrected patterns (in `scripts/timeout_analysis.py`):**

```python
"rate_limit_429": re.compile(
    r"(RateLimitError|raw_status_code['\":\s]+429|Error code: 429|HTTP 429|429 Too Many)"
),
"gateway_5xx": re.compile(
    r"(Error code: (502|503|504)|HTTP 50[234]|raw_status_code['\":\s]+50[234]|"
    r"Bad Gateway|Service Unavailable|Gateway Timeout)"
),
```

**Corrected numbers (re-run 2026-04-21):**
- Mistral real HTTP 429: **2 events total** across 46 timeouts.
- Qwen real HTTP 429: **0 events total** across 54 timeouts.
- Mistral gateway 5xx: 0 events (the 31 earlier hits were all false
  positives).
- Provider-reset signal (`upstream connect error` / `reset reason:
  overflow`) remains trustworthy; those phrases only appear in actual
  Envoy-style error frames. 35 events for Mistral, 0 for Qwen.

**Takeaway for methodology in the paper:** when scraping agent logs for
failure signatures, match the exception-type / status-line tokens
produced by the HTTP client, not bare numeric substrings. Skill
templates, tool arguments, and URLs all contain incidental numeric
tokens that look like HTTP status codes.

### E0.4 Notable wrong-answer evidence

From sampling the 14 Mistral + 10 Qwen wrong answers, three categories are
distinguishable:

1. **Strict-scorer formatting** (~30 % of wrong answers).
   E.g. task `6f37996b`: answer `b,e` → Qwen returned `b,e` (correct) and
   got False because the scorer split `b, e` into `['b', ' e']` and `b,e`
   into `['b', 'e']`. Our scorer strips whitespace from list elements per
   the leaderboard spec, so this particular one scores correctly; the
   actual wrong entries in this bucket are mostly number-rounding
   mismatches (e.g. `1.46` vs `1.456` on task `7dd30055`).

2. **GAIA ground-truth data error** (see Cross-cutting section for
   catalogue; confirmed `ded28325` = `Ploybius`/`Polybius` typo).

3. **Genuine retrieval/reasoning error** (~60 %).
   E.g. `c3a79cfe` (National Air and Space Museum inventor count), both
   models answered `4` vs truth `8`. These are legitimate GAIA-hard cases.

### E0.5 Takeaways to surface in the paper

For `methodology.tex`:
- The scorer used is a byte-identical port of the HF leaderboard scorer
  (with a `None` guard added for parity — see Cross-cutting section).
  Cite the leaderboard space URL.
- C3 training-phase sample is 80 random questions, seed=42, drawn before
  filtering, so the distribution across L1/L2/L3 matches validation.
- Per-Q timeout is 1800 s, per-step caps: planner 15, sub-agents 3,
  auto_browser_use 10, deep_researcher time_limit 45 s.

For `experiments.tex`:
- E0 raw accuracy (Mistral 13.75 %, Qwen 20.0 % over 80) should NOT be
  used as a C3 evaluation number — it is a training-phase sample. The
  paper's C0/C1/C2/C3 ablation deltas come from E3 test split.
- Skill library grew 7 → 13 (Mistral) and 7 → 9 (Qwen) during E0.

For `discussion.tex` / `limitations.tex`:
- Timeouts dominate errors (58–68 % of rows). Provider routing: Mistral
  runs on native La Plateforme, Qwen on OpenRouter. Real HTTP 429
  pressure is negligible on both (2 events Mistral, 0 events Qwen across
  the entire run) — see §E0.3.d for a post-mortem of the earlier
  false-positive "rate_limit" narrative.
- Timeout causes are **agent-internal**, not provider-side:
  (i) REVIEW-driven retry loops dominate both models (100 % of Mistral
  timeout batches, 71 % of Qwen timeout batches), and (ii) for Qwen only,
  cumulative DeepResearchTool 60 s sub-timeouts compound inside the
  1800 s per-Q budget. Both are fairness-preserving across C0/C1/C2/C3
  when mitigations are applied uniformly.
- REVIEW with `retry` verdicts is currently unbounded; 8 distinct
  rephrasings of the same delegation on task `0383a3ee` is the most
  extreme case observed. A 2-rephrase cap per sub-agent per task is the
  highest-leverage mitigation before E3.
- Learned-skill activation rate is near zero across both models. The
  extractor works end-to-end; the models just don't select learned
  entries over seeded ones. This is a legitimate research finding, not a
  bug, and it constrains claims about self-improvement.

For `related_work.tex` / framing:
- Under the taxonomy axes, E0 is **embedded ADAS**, **LLM feedback loop**
  (ReAct + REVIEW), **static benchmark** (GAIA), optimising **skill
  library (memory)** and indirectly **agent policy** (via REVIEW-driven
  modifications). Matches the "embedded ADAS is underdeveloped" gap
  identified in the literature review.

### E0.6 Pre-E3 action items (recommended, not yet applied)

Re-ordered 2026-04-21 after the §E0.3.d correction (rate-limiting is NOT
a primary cause, so provider-level mitigations drop off the list).

1. ~~**Cap REVIEW retries per sub-agent per task at 2**~~ → **LANDED**
   2026-04-24 on branch `feat/review-step-hardening`. Scope widened in
   implementation to cover more than a flat cap — see "Status" below.
   Still pending: live 5-Q smoke validation (recipe in
   [`docs/handoffs/HANDOFF_REVIEW_HARDENING.md`](../handoffs/HANDOFF_REVIEW_HARDENING.md)).
2. **Cap DeepResearchTool consecutive 60 s fallbacks per task at 3** in
   [`src/tools/deep_researcher.py`](../../src/tools/deep_researcher.py).
   Qwen-specific bottleneck (Mistral barely uses DR). Returns a
   `research_exhausted` sentinel so the planner changes strategy instead
   of burning budget. Fairness-safe because it applies uniformly across
   conditions and both models. **Not yet done.**
3. **Timeout bump to 2,400 s** for E3. Trivial config change; marginal
   lift once (1) and (2) are in place, but still a cheap safety margin
   for L3 tasks. **Not yet done.**
4. **`intermediate_steps` serialisation on timeout** — see Cross-cutting
   data-logging section. Not performance-critical but unlocks post-hoc
   analysis on E3. **Not yet done.**
5. ~~**`None` guard in `question_scorer`**~~ → **LANDED** 2026-04-21
   (see Cross-cutting scorer parity section).

None of 1-4 affect ablation integrity; all apply uniformly to C0/C1/C2/C3
and both models.

### Status of item #1 (REVIEW retry hardening) — landed 2026-04-24

Branch: `feat/review-step-hardening` (7 commits, ~296 tests passing
offline). Scope in the plan evolved beyond a flat cap during
implementation. **Delivered in this PR:**

- **Asymmetric per-root-cause retry caps** (not flat=2):
  `bad_instruction` / `misread_task` / `unclear_goal` → cap 2;
  `incomplete` → cap 1; `external` / `model_limit` / `missing_tool` /
  `wrong_tool` → cap 0 (retry never permitted — rephrasing cannot unlock
  paywalls, fix reasoning gaps, or synthesise missing capabilities).
- **Chain ledger** keyed on `(agent_name, intent_anchor)` — UUID
  inherited across reviewer-driven retries, fresh-minted on planner
  pivots. Resolves the narrow A→B→A via `_capped_anchors`.
- **Task-wide blocklist** `(agent_name, root_cause)` — stronger guard
  above the chain ledger. Once an agent fails with a cap=0 cause, ALL
  future delegations (review-driven OR planner-initiated with fresh
  anchor) are coerced. Closes the planner re-entry bypass that the
  per-chain cap alone doesn't cover.
- **Reviewer prompt rewrite** — sub-agent catalog + root-cause → action
  advisory table + three worked examples (add_existing primary /
  add_new gated / retry symmetric) + removed anti-bias line. Prompt
  length 7.9-8.3 KB (under the 12 KB budget asserted in tests).
- **Per-task metrics** (`retry_chains_started`, `retry_chains_capped`,
  `retry_coercions_to_proceed`, `blocklist_coercions`,
  `modify_agent_emitted`, `escalate_emitted`, `proceed_emitted`,
  `max_chain_length`) emitted into the `dra.jsonl` row under the new
  `review_metrics` key, for post-hoc calibration on E3.
- **Deferred to follow-up PR**: parallel-delegation review schema
  widening (original Layer 4). P6 is rare-and-silent in E0; moving the
  schema change post-E3 isolates risk from the ablation-critical
  retry-cap fix.

**Expected impact on E3** (projected from the plan; to be validated by
the live smoke run):

- `review_retry_loop` signature count per batch: 40-70 → ≤ 5.
- Per-Q wall time on runaway tasks: 1800 s → 300-900 s (2 retries max +
  escalate/modify/proceed).
- Scorable-rate on 80-Q sample: Mistral 33 → ~55-65; Qwen 26 → ~45-55.
- `modify_agent` selection rate: 2.5 % → 8-15 %.
- `escalate` selection rate: ~0 % → 5-10 %.

---

## Cross-cutting — scorer parity with HF leaderboard

_Applies to every GAIA-scored experiment (E0 / E1 / E2 / E3)._

The local `question_scorer` in [`src/metric/gaia_scorer.py`](../../src/metric/gaia_scorer.py)
was a near-byte-identical port of the [HF leaderboard scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
with one behavioural gap: a `None` prediction would raise inside
`normalize_number_str` rather than be scored as the literal string `"None"`.

**Fix applied 2026-04-21:** guard added at the top of `question_scorer`:

```python
if model_answer is None:
    model_answer = "None"
```

**Impact on current results:** zero. Both `scripts/analyze_results.py:350`
and `scripts/compare_results.py:77` already short-circuit
`pred is None / "" / "Unable to determine"` as `no_answer` before calling
the scorer. Zero rows in E0 `dra.jsonl` had `output=None` with no
`agent_error`. The fix is for leaderboard parity and safety against
future callers (tests, `monitor_tick.py`, ad-hoc analysis).

**Regression guard:** [`tests/test_gaia_scorer.py`](../../tests/test_gaia_scorer.py)
asserts `None` behaviour + core scorer invariants (string / number /
list, length-guard, punct-strip). 7/7 pass.

**Paper framing:** cite the leaderboard URL in `methodology.tex` and note
the single-line parity patch. This makes any accuracy number we report
directly comparable to the official leaderboard.

---

## Cross-cutting — `intermediate_steps` discarded on timeout path

_Applies to every GAIA run until patched. Known issue as of 2026-04-21;
not yet fixed._

**Location:** [`examples/run_gaia.py:202`](../../examples/run_gaia.py) and
[`:219`](../../examples/run_gaia.py) — both exception paths do
`intermediate_steps = []` before writing the row, even though
`agent.memory.steps` is intact and serialisable.

**Effect:** every timeout row has `intermediate_steps: []` in
`dra.jsonl`. Observed on 100/100 E0 timeout rows. Scoring is unaffected
(`agent_error` field drives error classification) but post-hoc per-step
forensics has to go through `log.txt`, which is 234 MB for Mistral E0
and interleaved across 4 concurrent tasks, making per-task analysis
significantly harder than it should be.

**Proposed surgical fix:**

```python
# at top of answer_single_question, before `for attempt in …:`
agent = None
intermediate_steps = []

# in BOTH `except asyncio.TimeoutError:` AND the generic exception path,
# replace `intermediate_steps = []` with:
intermediate_steps = _safe_serialize_steps(agent)

# module-level helper:
def _safe_serialize_steps(agent) -> list:
    """Best-effort; never raises. Returns [] if agent / memory missing."""
    if agent is None or not getattr(agent, "memory", None):
        return []
    steps = getattr(agent.memory, "steps", None)
    if not steps:
        return []
    try:
        for step in steps:
            step.model_input_messages = None  # symmetric with happy path
        return _serialize_steps(steps)
    except Exception as e:
        logger.warning(
            f"intermediate_steps serialisation failed on error path: {e}"
        )
        return []
```

Should be applied before E3 so that if any test-split row times out, the
post-hoc analysis isn't crippled.

---

## Cross-cutting — known GAIA ground-truth errors

_Applies to every GAIA-scored experiment. Append confirmed cases here as
they surface; note which split they live in._

| task_id | split | ground truth | observation |
|---|---|---|---|
| `ded28325` | validation | `Picnic is in Ploybius Plaza.` | `Ploybius` is a typo of `Polybius`. Qwen's correct `Polybius Plaza` answer scored False. |

**Paper framing:** list confirmed data errors in a threats-to-validity
block. If any show up in the test split, flag per-task and avoid
reporting those rows as "wrong" without qualification.
