---
name: research-fallback-sources
description: Ordered fallback strategies for `deep_researcher_agent` when the primary search returns no relevant results, conflicting results, or only low-credibility sources. Try domain-restricted search, archive lookups, and triangulation across independent sources before concluding "no information available".
metadata:
  consumer: deep_researcher_agent
  skill_type: failure_avoidance
  source: seeded
  verified_uses: 0
  confidence: 0.7
---

# Research Fallback Sources (deep_researcher_agent workflow)

## When to activate
- Top search results are irrelevant, contradictory, or only low-credibility (SEO-spam, AI-generated content farms), OR
- The primary source is paywalled and the archive-searcher cannot help, OR
- You have one source but need corroboration (every fact in a final answer should have ≥ 2 independent sources when feasible).

## Fallback order

1. **Domain-restricted search**. Re-run the query with `site:` operators for high-quality domains:
   - Technical/scientific: `site:arxiv.org`, `site:nature.com`, `site:acm.org`, `site:*.edu`
   - Companies / financial: `site:sec.gov`, `site:investor.<company>.com`
   - Current events: `site:reuters.com`, `site:apnews.com`, `site:bbc.com`
   - Reference: `site:en.wikipedia.org` (use as a starting point — then follow its citations)
2. **Use `archive_searcher_tool`** if the primary source is paywalled or 404'd. Wayback often has older, uncensored versions.
3. **Triangulate across independent sources**. Two sources owned by the same parent company (e.g., Fox News and WSJ) do not count as independent. Aim for at least two sources with different editorial leadership.
4. **Specialised databases**:
   - Biographical facts → Wikipedia, then the subject's official bio / institutional page.
   - Historical events → encyclopedia (Britannica, Wikipedia), then primary sources cited there.
   - Scientific claims → the original paper on arXiv / publisher site, not a press summary.
5. **Report "evidence is conflicting" rather than picking arbitrarily** if sources disagree. The manager wants truth, not confidence theater.

## Avoid
- Treating the first search hit as authoritative without checking the domain or source.
- Using content-farm sites (sites whose only purpose is SEO) as a primary source — they frequently contain AI hallucinations.
- Citing Wikipedia's claim as the answer — Wikipedia is a starting point; the citation it uses is the actual source.
- Concluding "no information available" after only one search attempt.

## Credibility heuristic
When in doubt, prefer sources in this rough order:
1. Primary sources (official filings, press releases, peer-reviewed papers)
2. Established reference works (Britannica, major dictionaries)
3. Wire services / established news (Reuters, AP, BBC, major national papers)
4. Established secondary reference (Wikipedia, when it has citations)
5. Industry-specific publications with clear editorial standards

Below this line you are usually doing more harm than good.
