#!/usr/bin/env python3
"""Print Firecrawl team credit usage (reads FIRECRAWL_API_KEY from environment / .env)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    if load_dotenv:
        load_dotenv(repo / ".env", verbose=False)

    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not key:
        print("FIRECRAWL_API_KEY is not set (add it to .env).", file=sys.stderr)
        return 2

    url = os.getenv("FIRECRAWL_CREDIT_URL", "https://api.firecrawl.dev/v1/team/credit-usage")
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:500]}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print(body[:2000])
        return 0

    if isinstance(data, dict) and data.get("success") and isinstance(data.get("data"), dict):
        data = data["data"]

    if os.getenv("FIRECRAWL_CREDITS_JSON"):
        print(json.dumps(data, indent=2))
        return 0

    # v1 snake_case vs v2 camelCase
    remaining = data.get("remainingCredits")
    if remaining is None:
        remaining = data.get("remaining_credits")
    plan = data.get("planCredits")
    if plan is None:
        plan = data.get("plan_credits")

    if remaining is not None:
        print(f"remaining_credits: {remaining}")
    if plan is not None:
        print(f"plan_credits: {plan}")
    if remaining is None and plan is None:
        print(body[:2000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
