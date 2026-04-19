---
name: browser-paywall-recovery
description: Recovery workflow when `auto_browser_use_tool` hits a paywall, cookie banner, login wall, CAPTCHA challenge, or 403/429 error. Try archive.org, Google cache, and site-specific alternatives before giving up. Use whenever the browser cannot access a page that looked reachable from the search results.
metadata:
  consumer: browser_use_agent
  skill_type: failure_avoidance
  source: seeded
  verified_uses: 0
  confidence: 0.75
---

# Browser Paywall / Access Recovery (browser_use_agent workflow)

## When to activate
- Page shows paywall / "subscribe to continue" / login wall, OR
- **Page shows a CAPTCHA challenge** (reCAPTCHA, hCaptcha, Cloudflare "verify you are human", Google "I'm not a robot"), OR
- HTTP 403 (forbidden) or 429 (rate-limited), OR
- Cookie-consent modal blocks interaction and you cannot dismiss it.

## CAPTCHA handling — do not retry the same URL

CAPTCHAs are bot-defense walls the browser agent cannot pass. Retrying the same
URL just burns `auto_browser_use_tool.max_steps` (capped; observed 25+ wasted
CAPTCHA-retry iterations under the old default of 50 steps). **On the FIRST
CAPTCHA sighting, switch source** — do not try a second CAPTCHA-gated page from
the same site, and do not try to solve the challenge.

## Recovery order

1. **Try the Wayback Machine (archive.org)**. Prepend `https://web.archive.org/web/` to the original URL; many articles have at least one snapshot that is easier to read than the live paywalled page (coverage is not guaranteed).
2. **Try Google Cache** by navigating to `https://webcache.googleusercontent.com/search?q=cache:<URL>` only as a low-probability fallback — public cache access is often blocked, rate-limited, or empty for major publishers.
3. **Try the site's AMP / print / reader version**:
   - Append `?outputType=amp` or check for `/amp/` routes.
   - Append `?print=1` or look for a "print this article" link.
4. **Try a different source** — if the user wants a specific fact, the same fact often appears on Wikipedia, the source's press-release, or a secondary news site. Go back to the search results and pick another hit.
5. **If all else fails**, report that the page is inaccessible and provide the best alternative snippet you did find (search result summary, cached excerpt). Do NOT return an empty result.

## Avoid
- Dismissing cookie modals with "Agree" as a default tactic — it often still leaves the article behind a paywall and burns steps.
- Repeatedly retrying the same URL after a 429 — rate limits don't clear instantly.
- Attempting to solve a CAPTCHA via clicks, scrolls, or text input — the browser agent cannot pass these, and trying wastes steps. Switch source immediately.
- Using `python_interpreter_tool` to send HTTP requests bypassing the browser (`requests` is not on the default interpreter allowlist anyway). Even where raw HTTP works, it lacks full browser cookie/JS behavior and can trigger bot defenses.

## Example
```
Original: https://www.nytimes.com/2025/01/01/article.html  → paywall
Try:     https://web.archive.org/web/https://www.nytimes.com/2025/01/01/article.html
```
