---
name: research-fallback-sources
description: Ordered fallback strategies for deep_researcher_agent when primary search is weak or sources conflict. Prefer refined queries and domain-restricted search; triangulate across editorially independent sources. Notes default GAIA tool surface (deep_researcher_tool + python_interpreter_tool only).
metadata:
  consumer: deep_researcher_agent
  skill_type: failure_avoidance
  source: seeded
  verified_uses: 0
  confidence: 0.7
---

# Research Fallback Sources (deep_researcher_agent workflow)

## When to activate
- Top search results are irrelevant, contradictory, or only low-credibility (SEO-spam, synthetic content farms), OR
- The primary source is paywalled or 404, OR
- You have one source but need corroboration (aim for ≥ 2 **editorially independent** sources when feasible).

## Tool surface (default GAIA)
- This agent is typically configured with **`deep_researcher_tool`** and **`python_interpreter_tool`** only. An **`archive_searcher_tool`** exists in the codebase but is **not** mounted in the default GAIA configs — do **not** call tools that are not in your live tool list (calls will fail with “not found”). If your run added Wayback tooling via config, use it; otherwise use the fallbacks below.

## Fallback order

1. **Domain-restricted search**. Re-run via `deep_researcher_tool` with `site:` operators for higher-signal domains:
   - Technical/scientific: `site:arxiv.org`, `site:nature.com`, `site:acm.org`, `site:*.edu`
   - Companies / financial: `site:sec.gov`, `site:investor.<company>.com`
   - Current events: `site:reuters.com`, `site:apnews.com`, `site:bbc.com`
   - Reference: `site:en.wikipedia.org` (starting point — follow citations for the primary claim)
2. **Paywall / dead URL without Wayback tool:** refine the query (article title + outlet, alternate phrasing), include `web.archive.org` **in the search query** so result snippets may surface snapshots, or rely on the **planner** to delegate **`browser_use_agent`** to open `https://web.archive.org/web/<original URL>` when navigation is required.
3. **Triangulate across independent sources.** True independence means different ownership **and** editorial process — e.g. the same newswire paragraph syndicated to two outlets is **one** chain of custody, not two proofs. Prefer a regulator filing + a reputable newsroom, or a paper + its official supplement, over two SEO mirrors.
4. **Specialised databases (via search, not hallucinated URLs):**
   - Biographical facts → Wikipedia for leads, then official bio / institution.
   - Historical events → encyclopedia entries, then cited primary sources.
   - Scientific claims → the paper on arXiv or the publisher version, not only a press blog.
5. **Report conflicting evidence** instead of forcing a single number or narrative when sources disagree after a serious search effort.

## Avoid
- Treating the first search hit as authoritative without checking domain and date.
- Using thin affiliate or auto-generated pages as primary evidence.
- Citing Wikipedia’s sentence as the final authority — trace to the cited source when the claim matters.
- Concluding “no information available” after a single generic query.

## Credibility heuristic
When in doubt, prefer sources in this rough order:
1. Primary sources (official filings, press releases, peer-reviewed papers)
2. Established reference works (Britannica, major dictionaries)
3. Wire services / established news (Reuters, AP, BBC, major national papers)
4. Established secondary reference (Wikipedia when it provides verifiable citations)
5. Industry-specific publications with clear editorial standards

Below this line you are usually doing more harm than good.
