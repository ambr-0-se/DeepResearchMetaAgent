---
name: browser-paywall-recovery
description: Recovery workflow when `auto_browser_use_tool` hits a paywall, cookie banner, login wall, or 403/429 error. Try archive.org, Google cache, and site-specific alternatives before giving up. Use whenever the browser cannot access a page that looked reachable from the search results.
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
- HTTP 403 (forbidden) or 429 (rate-limited), OR
- Cookie-consent modal blocks interaction and you cannot dismiss it.

## Recovery order

1. **Try the Wayback Machine (archive.org)**. Prepend `https://web.archive.org/web/` to the original URL; the Wayback snapshot is usually paywall-free and does not require login.
2. **Try Google Cache** by navigating to `https://webcache.googleusercontent.com/search?q=cache:<URL>` (note: Google deprecated public cache in 2024; works intermittently).
3. **Try the site's AMP / print / reader version**:
   - Append `?outputType=amp` or check for `/amp/` routes.
   - Append `?print=1` or look for a "print this article" link.
4. **Try a different source** — if the user wants a specific fact, the same fact often appears on Wikipedia, the source's press-release, or a secondary news site. Go back to the search results and pick another hit.
5. **If all else fails**, report that the page is inaccessible and provide the best alternative snippet you did find (search result summary, cached excerpt). Do NOT return an empty result.

## Avoid
- Trying to dismiss cookie modals by clicking "Agree" — this often succeeds but is ethically questionable and frequently just reveals the paywall anyway.
- Repeatedly retrying the same URL after a 429 — rate limits don't clear instantly.
- Using `python_interpreter_tool` to send HTTP requests bypassing the browser (`requests.get`). It often works but lacks the cookie/JS handling needed for modern sites and can look like bot traffic.

## Example
```
Original: https://www.nytimes.com/2025/01/01/article.html  → paywall
Try:     https://web.archive.org/web/https://www.nytimes.com/2025/01/01/article.html
```
