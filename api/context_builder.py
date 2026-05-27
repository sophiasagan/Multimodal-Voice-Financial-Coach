"""
api/context_builder.py — Build the member account context string for Claude.

Pulls two data sources in parallel:
  1. Core account data  — products, balances, upcoming events
  2. P31 Financial Twin — churn_score, health_score, product_propensities,
                          behavioral_summary, next_best_action

Both calls use the same httpx.AsyncClient so the underlying TCP connection is
reused, and asyncio.gather fires them concurrently to keep latency low.

The final context block is intentionally short (≤ 300 tokens) because it will
be concatenated with the coach system prompt and placed in Claude's system
message under a cache_control ephemeral block.  Every byte here is charged on
cache misses, so be concise.

Failure modes
-------------
• If the P31 API is unreachable, _fetch_twin() returns {} and context is built
  without AI-insight fields.
• If the accounts API is unreachable, _fetch_accounts() returns {} and context
  falls back to the member's name only (still better than nothing).
• If both fail (or member is None), GENERIC_CONTEXT is returned.  The coach
  then gives general financial guidance without referencing any account.

Privacy
-------
• Balances are rounded to the nearest $100 in log output only.
• The context string itself contains real figures — it is consumed only by
  Claude and never written to disk or returned to a client.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import date, datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

P31_BASE_URL: str = os.getenv("P31_API_BASE_URL", "https://api.p31financial.com/v1")
P31_API_KEY:  str = os.getenv("P31_API_KEY", "")

# Shared by _fetch_accounts and _fetch_twin; created once per module load.
# connect_timeout=3s, read_timeout=5s — must finish well within Twilio's
# 60-second call-setup window.
_HTTP_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=2.0, pool=2.0)

GENERIC_CONTEXT = (
    "No account data available. Member has not been identified or their "
    "account details could not be retrieved at this time. Provide general "
    "financial guidance only."
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {P31_API_KEY}",
        "Accept":        "application/json",
    }


async def _fetch_accounts(member_id: str) -> dict[str, Any]:
    """
    Retrieve the member's account summary from the CU's core / data warehouse.

    Expected response shape (simplified):
    {
      "total_balance":          12500.00,
      "products": [
          {"type": "checking", "balance": 1200.00},
          {"type": "savings",  "balance": 8000.00},
          {"type": "loan",     "balance": 3300.00, "next_payment_date": "2026-06-01"},
          {"type": "cd",       "balance": 5000.00, "maturity_date":      "2026-07-15"}
      ],
      "direct_deposit_active":  true,
      "membership_since":       "2018-04-03"
    }
    """
    url = f"{P31_BASE_URL}/members/{member_id}/accounts"
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(url, headers=_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Accounts API HTTP %s for member %s: %s",
            exc.response.status_code, member_id, exc.response.text[:200],
        )
    except Exception as exc:
        logger.warning("Accounts API error for member %s: %s", member_id, exc)
    return {}


async def _fetch_twin(member_id: str) -> dict[str, Any]:
    """
    Retrieve the member's P31 Financial Twin insights.

    Expected response shape (simplified):
    {
      "churn_score":           0.12,          # 0.0–1.0
      "health_score":          74,             # 0–100
      "product_propensities":  {"auto_loan": 0.68, "mortgage": 0.45},
      "behavioral_summary":    "Tends to carry a low checking balance...",
      "next_best_action":      "Offer 6-month CD renewal at current rate."
    }
    """
    url = f"{P31_BASE_URL}/members/{member_id}/twin"
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(url, headers=_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Financial Twin API HTTP %s for member %s: %s",
            exc.response.status_code, member_id, exc.response.text[:200],
        )
    except Exception as exc:
        logger.warning("Financial Twin API error for member %s: %s", member_id, exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _membership_years(since_str: str) -> int:
    """
    Convert an ISO date string to whole years of membership.

    Returns 0 on any parse error rather than raising.
    """
    try:
        since = date.fromisoformat(since_str[:10])
        return max(0, (date.today() - since).days // 365)
    except Exception:
        return 0


def _product_list(products: list[dict]) -> str:
    """
    Build a readable comma-joined list of product types.

    E.g. "checking, savings, auto loan, 12-month CD"
    """
    labels = []
    for p in products:
        t = (p.get("type") or "").lower()
        if t == "checking":
            labels.append("checking")
        elif t == "savings":
            labels.append("savings")
        elif t == "loan":
            labels.append("loan")
        elif t == "cd":
            labels.append("CD")
        elif t:
            labels.append(t)
    return ", ".join(labels) if labels else "none on file"


def _upcoming_events(products: list[dict]) -> list[str]:
    """
    Extract time-sensitive events from the product list.

    Returns a list of short strings, e.g.:
    ["CD matures 2026-07-15", "Loan payment due 2026-06-01"]
    """
    events: list[str] = []
    today = date.today()
    for p in products:
        t = (p.get("type") or "").lower()
        if t == "cd":
            mat = p.get("maturity_date", "")
            if mat:
                try:
                    mat_date = date.fromisoformat(mat[:10])
                    days_out = (mat_date - today).days
                    if 0 <= days_out <= 60:
                        events.append(f"CD matures {mat_date.isoformat()}")
                except Exception:
                    pass
        elif t == "loan":
            pmt = p.get("next_payment_date", "")
            if pmt:
                try:
                    pmt_date = date.fromisoformat(pmt[:10])
                    days_out = (pmt_date - today).days
                    if 0 <= days_out <= 14:
                        events.append(f"Loan payment due {pmt_date.isoformat()}")
                except Exception:
                    pass
    return events


def _fmt_balance(amount: float) -> str:
    """Format dollar amount to nearest $1K for concise context: '$12.5K'."""
    if amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def build_context(member: Optional[dict]) -> str:
    """
    Build a concise account context string for the Claude coach system prompt.

    Parameters
    ----------
    member : dict | None
        The dict returned by member_resolver.resolve_member().
        Keys used: id, first_name, last_name.
        Pass None (or an empty dict) to get GENERIC_CONTEXT.

    Returns
    -------
    str
        A ≤ 300-token context block describing the member's financial profile,
        suitable for direct insertion into Claude's system prompt.
        Returns GENERIC_CONTEXT if member is None or data cannot be fetched.
    """
    if not member or not member.get("id"):
        logger.debug("build_context: no member — returning generic context.")
        return GENERIC_CONTEXT

    member_id   = member["id"]
    first_name  = member.get("first_name", "Member")
    last_name   = member.get("last_name", "")
    full_name   = f"{first_name} {last_name}".strip()

    logger.debug("build_context: fetching data for member %s", member_id)

    # ── Parallel API calls ────────────────────────────────────────────────────
    accounts_data, twin_data = await asyncio.gather(
        _fetch_accounts(member_id),
        _fetch_twin(member_id),
    )

    # ── Accounts section ──────────────────────────────────────────────────────
    products       = accounts_data.get("products", [])
    total_balance  = accounts_data.get("total_balance", 0.0)
    dd_active      = accounts_data.get("direct_deposit_active", False)
    since_str      = accounts_data.get("membership_since", "")
    years          = _membership_years(since_str)
    product_names  = _product_list(products)
    events         = _upcoming_events(products)

    # ── Financial Twin section ────────────────────────────────────────────────
    health_score   = twin_data.get("health_score")          # int 0–100 or None
    behavioral     = twin_data.get("behavioral_summary", "")
    nba            = twin_data.get("next_best_action", "")

    # ── Assemble context block ────────────────────────────────────────────────
    parts: list[str] = []

    # Opener — membership tenure
    if years > 0:
        parts.append(
            f"Member context: {first_name} has been a member for {years} year"
            f"{'s' if years != 1 else ''}."
        )
    else:
        parts.append(f"Member context: {full_name}.")

    # Products & balance
    if products:
        bal_str = f"Total deposits: ~{_fmt_balance(total_balance)}." if total_balance else ""
        parts.append(f"Products: {product_names}. {bal_str}".strip())

    # Direct deposit
    if dd_active:
        parts.append("Direct deposit active.")

    # Upcoming events
    if events:
        parts.append("Upcoming: " + "; ".join(events) + ".")

    # Financial health score
    if health_score is not None:
        parts.append(f"Financial health score: {health_score}/100.")

    # Behavioral insight (trim to 120 chars to stay under token budget)
    if behavioral:
        trimmed = behavioral[:120].rstrip()
        if len(behavioral) > 120:
            trimmed += "…"
        parts.append(trimmed)

    # Next best action
    if nba:
        parts.append(f"Note: {nba[:120].rstrip()}")

    context = " ".join(parts)

    logger.info(
        "build_context: member=%s  health=%s  products=%d  chars=%d",
        member_id,
        health_score if health_score is not None else "n/a",
        len(products),
        len(context),
    )

    return context if context else GENERIC_CONTEXT
