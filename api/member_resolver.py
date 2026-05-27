"""
api/member_resolver.py — Phone-number-based member identity resolution.

Security model
--------------
For standard calls, matching the inbound Twilio phone number to an active
member record in the database is the sole identity check.  No account
balances or sensitive financial data are returned here — only the profile
fields that allow context_builder to fetch full account context from the
P31 Financial Twin API.

The contract for callers:
  • resolve_member() returns a dict  → member is identified (phone matched)
  • resolve_member() returns None    → caller is unknown; coach continues in
                                       generic (unauthenticated) guidance mode

For high-security CUs, set SSN_VERIFICATION=true in the environment.
This flags the returned member dict with ssn_verified=False.  coach.py /
call_handler.py must check member["ssn_verified"] before letting
context_builder expose real account data.  Use verify_ssn_last4() to
complete that second factor via speech input.

Privacy notes
-------------
• Full phone numbers are NEVER logged — only the last-4 digits.
• SSN digits are NEVER logged — only the boolean match outcome.
• The member's name is logged at INFO for operational support tracing;
  redact or hash in production if your CU's policy requires it.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://localhost/cu_coach",
)

# Enable for high-security CUs that require a second verbal factor.
# When True: the returned member dict has ssn_verified=False until
# verify_ssn_last4() is called and passes.
REQUIRE_SSN_VERIFICATION: bool = (
    os.getenv("SSN_VERIFICATION", "false").lower() == "true"
)

# ─────────────────────────────────────────────────────────────────────────────
# Async database engine  (lazy singleton; shared across all WebSocket sessions)
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[AsyncEngine] = None


def _get_engine() -> AsyncEngine:
    """
    Return the shared async SQLAlchemy engine, creating it on first call.

    pool_pre_ping=True detects stale connections that survived a DB restart
    without adding meaningful latency on healthy connections.
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


# ── Minimal table reflection (mirrors the ORM model / Alembic migration) ────
# Only columns needed for identity resolution are declared here.
# context_builder calls the P31 API with member["id"] to get account data.

_members = sa.Table(
    "members",
    sa.MetaData(),
    sa.Column("id",             sa.String, primary_key=True),
    sa.Column("member_number",  sa.String),   # human-readable CU member #
    sa.Column("first_name",     sa.String),
    sa.Column("last_name",      sa.String),
    sa.Column("phone_e164",     sa.String),   # canonical: +1XXXXXXXXXX
    sa.Column("email",          sa.String),   # for follow-up notifications
    sa.Column("status",         sa.String),   # active | suspended | closed
    sa.Column("ssn_last4_hash", sa.String),   # bcrypt; only read when SSN verify on
)

# ─────────────────────────────────────────────────────────────────────────────
# Phone normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_phone(raw: str) -> str:
    """
    Reduce any phone string to E.164 format (+COUNTRYCODESUBSCRIBER).

    Handles common North American formats and passes international numbers
    through unchanged (preserving the country code already present).

    Examples
    --------
    "+1 (555) 867-5309"  →  "+15558675309"
    "(555) 867-5309"     →  "+15558675309"   # 10-digit US assumed
    "5558675309"         →  "+15558675309"
    "15558675309"        →  "+15558675309"
    "+44 20 7946 0958"   →  "+442079460958"  # non-US preserved
    """
    digits = re.sub(r"\D", "", raw)

    # Bare 10-digit North American number — prepend US/CA country code
    if len(digits) == 10:
        digits = "1" + digits

    return "+" + digits


def _phone_tail(phone: str) -> str:
    """
    Return the last-4 digits of a phone number for safe log output.

    Never logs the full number; this is the only form that appears in logs.
    """
    digits = re.sub(r"\D", "", phone)
    return f"…{digits[-4:]}" if len(digits) >= 4 else "…????"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def resolve_member(phone_number: Optional[str]) -> Optional[dict]:
    """
    Look up an active CU member by their inbound caller phone number.

    Parameters
    ----------
    phone_number : str | None
        Raw phone string as received from Twilio (typically E.164, e.g.
        "+15558675309").  None-safe: returns None immediately if falsy.

    Returns
    -------
    dict
        Keys: id, member_number, first_name, last_name,
              phone_e164, email, status
        Also sets ssn_verified=False when REQUIRE_SSN_VERIFICATION is on.
    None
        When the phone number is unknown, not in the database, or the
        matching account is not active (suspended / closed).

    Caller behaviour on None
    ------------------------
    The voice pipeline continues without account context.  The coach
    delivers generic financial guidance only — no balances, no history,
    no PII.  This is the correct safe-fallback posture.

    Database errors
    ---------------
    Caught and logged; the function returns None so the call proceeds
    rather than crashing.  A DB outage degrades to generic-guidance mode,
    not a dead line.
    """
    if not phone_number:
        logger.debug("resolve_member: no phone number provided — guest mode.")
        return None

    canonical = normalize_phone(phone_number)
    logger.debug("Resolving member  phone=%s", _phone_tail(canonical))

    try:
        async with _get_engine().connect() as conn:
            stmt = (
                sa.select(_members)
                .where(
                    _members.c.phone_e164 == canonical,
                    _members.c.status     == "active",
                )
                .limit(1)
            )
            row = (await conn.execute(stmt)).mappings().first()

    except Exception as exc:
        # Treat any DB error as "member not found" — the call keeps going.
        logger.error(
            "DB error during member lookup  phone=%s: %s",
            _phone_tail(canonical), exc,
        )
        return None

    if row is None:
        logger.info(
            "Member not found (or not active)  phone=%s — generic guidance mode.",
            _phone_tail(canonical),
        )
        return None

    member: dict = dict(row)

    logger.info(
        "Member resolved  id=%s  name=%s %s  phone=%s  status=%s",
        member["id"],
        member["first_name"],
        member["last_name"],
        _phone_tail(canonical),
        member["status"],
    )

    # Flag that SSN 2nd-factor is required before full account access.
    # context_builder checks member.get("ssn_verified") when this is on.
    if REQUIRE_SSN_VERIFICATION:
        member["ssn_verified"] = False

    return member


# ─────────────────────────────────────────────────────────────────────────────
# High-security option: last-4 SSN verbal verification
# ─────────────────────────────────────────────────────────────────────────────

async def verify_ssn_last4(member: dict, spoken_digits: str) -> bool:
    """
    Validate the member's verbally spoken last-4 SSN against the stored hash.

    This is the second verification factor for high-security CUs.  It is
    called from coach.py after collecting the member's response to a prompt
    like "To access your account details, could you confirm the last 4 digits
    of your Social Security number?"

    Parameters
    ----------
    member        : dict returned by a successful resolve_member() call.
    spoken_digits : Whisper transcript of the member's spoken response.
                    Normalised internally; tolerates "1 2 3 4", "12 34", etc.

    Returns
    -------
    True  if the digits match the stored bcrypt hash.
          Also sets member["ssn_verified"] = True in-place.
    False on mismatch, bad input length, missing hash, or bcrypt not installed.

    Security
    --------
    • The actual digits are NEVER logged — only the boolean outcome.
    • bcrypt's constant-time comparison prevents timing attacks.
    • Three consecutive failures should trigger call escalation in coach.py
      (implement a counter in the call session; not handled here).

    Disabled by default
    -------------------
    This function is a no-op (returns False with a warning) unless
    REQUIRE_SSN_VERIFICATION=true is set in the environment.  Enables a
    clear audit trail: if SSN_VERIFICATION is off and this is called,
    something is misconfigured.
    """
    if not REQUIRE_SSN_VERIFICATION:
        logger.warning(
            "verify_ssn_last4 called but SSN_VERIFICATION env var is not enabled — "
            "returning False.  Set SSN_VERIFICATION=true to activate."
        )
        return False

    # bcrypt is an optional dependency (pip install bcrypt).
    try:
        import bcrypt  # type: ignore[import]
    except ImportError:
        logger.error(
            "bcrypt is not installed — cannot complete SSN verification. "
            "Run: pip install bcrypt"
        )
        return False

    # Normalise: extract only digit characters from the Whisper transcript
    digits = re.sub(r"\D", "", spoken_digits)

    if len(digits) != 4:
        logger.debug(
            "SSN verify: expected 4 digits from transcript, got %d  member=%s",
            len(digits), member.get("id"),
        )
        return False

    stored_hash: str = member.get("ssn_last4_hash", "")
    if not stored_hash:
        logger.warning(
            "Member %s has no ssn_last4_hash stored — cannot verify.",
            member.get("id"),
        )
        return False

    matched: bool = bcrypt.checkpw(
        digits.encode("ascii"),
        stored_hash.encode("ascii"),
    )

    # Log outcome only — never the actual digits
    logger.info(
        "SSN verification  member=%s  matched=%s",
        member.get("id"), matched,
    )

    if matched:
        member["ssn_verified"] = True

    return matched
