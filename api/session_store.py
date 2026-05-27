"""
api/session_store.py — Redis-backed call session management.

Lifecycle
---------
1. create_session()  — called when the WebSocket START event arrives
2. append_turn()     — called after each Whisper transcription and each
                       coach response is generated
3. update_sentiment() — called after guardrails.check_guardrails() so
                       we track member affect across the conversation
4. mark_escalated()  — called if guardrails trigger a transfer
5. end_session()     — called on WebSocket STOP; generates a Claude summary,
                       saves to Postgres, creates a CRM task if needed

Redis storage
-------------
Key:  call_session:{call_sid}
Value: JSON blob (see _empty_session())
TTL:  SESSION_TTL_SECONDS (default 14 400 s = 4 h) — auto-expire stale
      sessions left by crashed workers.

Database
--------
Summaries are persisted to the member_coaching_sessions table (created by
the Alembic migration; see below for the column layout).  The async
SQLAlchemy engine is injected at construction time (shared with
member_resolver.py to avoid duplicate pools).

CRM integration
---------------
If the summary flags follow_up_required=True the session store posts a task
to the configured CRM API.  The implementation targets Microsoft Dynamics 365
(D365) via its Web API, with P71 / generic REST as a fallback.  Set
CRM_PROVIDER=d365|p71|none in the environment.

Claude summary
--------------
generate_session_summary() calls Claude once per call-end to extract
structured data from the full conversation transcript.  Using claude-sonnet-4-6
(fast, cheap) with adaptive thinking off — this is a well-structured extraction
task, not reasoning.  Output is parsed as JSON.

Privacy
-------
• member_id and call_sid are written to the DB; full transcripts are stored
  as JSONB in raw_transcript and are subject to your CU's data retention
  policy.  Set STORE_RAW_TRANSCRIPT=false to omit them.
• Sentiment labels and question summaries are never logged at INFO level with
  member PII.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL:            str  = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS:  int  = int(os.getenv("SESSION_TTL_SECONDS", "14400"))   # 4 h
ANTHROPIC_MODEL:      str  = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
CRM_PROVIDER:         str  = os.getenv("CRM_PROVIDER", "none").lower()         # d365 | p71 | none
D365_BASE_URL:        str  = os.getenv("D365_BASE_URL", "")
D365_ACCESS_TOKEN:    str  = os.getenv("D365_ACCESS_TOKEN", "")
P71_BASE_URL:         str  = os.getenv("P71_API_BASE_URL", "https://api.p31financial.com/v1")
P71_API_KEY:          str  = os.getenv("P71_API_KEY", "")
STORE_RAW_TRANSCRIPT: bool = os.getenv("STORE_RAW_TRANSCRIPT", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# Database table (mirrors Alembic migration)
# ─────────────────────────────────────────────────────────────────────────────
#
# CREATE TABLE member_coaching_sessions (
#   id                    UUID PRIMARY KEY,
#   call_sid              VARCHAR UNIQUE NOT NULL,
#   member_id             VARCHAR REFERENCES members(id),
#   started_at            TIMESTAMP NOT NULL,
#   ended_at              TIMESTAMP,
#   duration_s            INTEGER,
#   topics_covered        JSONB,        -- list[str]
#   member_questions      JSONB,        -- list[str]
#   information_provided  JSONB,        -- list[str]
#   action_items          JSONB,        -- list[{task: str, owner: str}]
#   member_sentiment      VARCHAR,      -- positive|neutral|concerned|distressed
#   follow_up_required    BOOLEAN,
#   follow_up_description TEXT,
#   escalated             BOOLEAN,
#   escalation_type       VARCHAR,
#   raw_transcript        JSONB         -- full turn-by-turn list (if enabled)
# );

_sessions_table = sa.Table(
    "member_coaching_sessions",
    sa.MetaData(),
    sa.Column("id",                    sa.String,   primary_key=True),
    sa.Column("call_sid",              sa.String,   nullable=False),
    sa.Column("member_id",             sa.String),
    sa.Column("started_at",            sa.DateTime),
    sa.Column("ended_at",              sa.DateTime),
    sa.Column("duration_s",            sa.Integer),
    sa.Column("topics_covered",        sa.JSON),
    sa.Column("member_questions",      sa.JSON),
    sa.Column("information_provided",  sa.JSON),
    sa.Column("action_items",          sa.JSON),
    sa.Column("member_sentiment",      sa.String),
    sa.Column("follow_up_required",    sa.Boolean),
    sa.Column("follow_up_description", sa.Text),
    sa.Column("escalated",             sa.Boolean),
    sa.Column("escalation_type",       sa.String),
    sa.Column("raw_transcript",        sa.JSON),
)

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight live-sentiment heuristic (keyword-based per-turn signal)
# Full Claude analysis runs only at call end.
# ─────────────────────────────────────────────────────────────────────────────

_SENTIMENT_UPGRADE: dict[str, list[str]] = {
    # Escalate current level toward a worse state when these appear.
    # Only the *highest* level triggered wins (distressed > concerned > positive).
    "distressed": [
        "kill myself", "hurt myself", "can't go on", "don't want to live",
        "want to die", "no reason", "end it", "can't take it",
    ],
    "concerned": [
        "worried", "scared", "nervous", "don't understand", "confused",
        "not sure", "anxious", "frustrated", "this is hard", "struggling",
        "can't afford", "losing", "can't pay",
    ],
    "positive": [
        "thank you", "that helps", "that's helpful", "great", "perfect",
        "makes sense", "wonderful", "appreciate", "good to know", "love that",
    ],
}

# Ordered from worst to best so we can short-circuit at first match
_SENTIMENT_PRIORITY = ["distressed", "concerned", "neutral", "positive"]


def _infer_live_sentiment(text: str, current: str) -> str:
    """
    Lightweight keyword-based sentiment update for a single utterance.

    Sentiment only escalates toward 'distressed' — it never improves mid-call
    from 'concerned' to 'positive' based on a single upbeat phrase.  This is
    intentional: once a member has shown concern, hold that signal until the
    Claude end-of-call analysis provides the authoritative assessment.
    """
    lower = text.lower()
    for label in ["distressed", "concerned"]:     # don't downgrade
        for phrase in _SENTIMENT_UPGRADE[label]:
            if phrase in lower:
                # Escalate only if it's worse than current
                if _SENTIMENT_PRIORITY.index(label) < _SENTIMENT_PRIORITY.index(current):
                    return label
    # Check for positive only if we're still at neutral
    if current == "neutral":
        for phrase in _SENTIMENT_UPGRADE["positive"]:
            if phrase in lower:
                return "positive"
    return current


# ─────────────────────────────────────────────────────────────────────────────
# Session dict factory
# ─────────────────────────────────────────────────────────────────────────────

def _empty_session(
    call_sid:   str,
    member:     Optional[dict],
    phone_e164: str,
) -> dict:
    """Return a fresh session dict with all fields initialised."""
    return {
        "call_sid":          call_sid,
        "member_id":         member["id"]         if member else None,
        "member_name":       (
            f"{member.get('first_name', '')} {member.get('last_name', '')}".strip()
            if member else "Guest"
        ),
        "phone_e164":        phone_e164,
        "start_time":        time.time(),
        "transcript":        [],          # list of {role, content, ts}
        "topics_discussed":  [],          # list of str (appended by coach.py callers)
        "actions_promised":  [],          # list of str
        "member_sentiment":  "neutral",
        "escalated":         False,
        "escalation_type":   None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SessionStore class
# ─────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    Manages per-call session state in Redis and persists end-of-call summaries
    to Postgres.

    Usage
    -----
    # In FastAPI lifespan:
    store = SessionStore(engine=_get_engine())
    await store.open()
    ...
    await store.close()
    """

    def __init__(self, engine: Optional[AsyncEngine] = None) -> None:
        self._engine  = engine
        self._redis   = None            # initialised in open()
        self._claude  = None            # lazy

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def open(self) -> None:
        """Open the Redis connection pool.  Call from FastAPI lifespan startup."""
        try:
            from redis.asyncio import from_url as redis_from_url  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "redis package is not installed. Run: pip install redis"
            )
        self._redis = await redis_from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("SessionStore: Redis connection pool opened  url=%s", REDIS_URL.split("@")[-1])

    async def close(self) -> None:
        """Close the Redis connection pool.  Call from FastAPI lifespan shutdown."""
        if self._redis:
            await self._redis.aclose()
            logger.info("SessionStore: Redis connection pool closed.")

    # ── Redis helpers ─────────────────────────────────────────────────────────

    def _key(self, call_sid: str) -> str:
        return f"call_session:{call_sid}"

    async def _load(self, call_sid: str) -> Optional[dict]:
        if not self._redis:
            return None
        raw = await self._redis.get(self._key(call_sid))
        return json.loads(raw) if raw else None

    async def _save(self, call_sid: str, session: dict) -> None:
        if not self._redis:
            return
        await self._redis.setex(
            self._key(call_sid),
            SESSION_TTL_SECONDS,
            json.dumps(session, default=str),
        )

    async def _delete(self, call_sid: str) -> None:
        if self._redis:
            await self._redis.delete(self._key(call_sid))

    # ── Public API ────────────────────────────────────────────────────────────

    async def create_session(
        self,
        call_sid:   str,
        member:     Optional[dict],
        phone_e164: str,
    ) -> None:
        """
        Initialise a new session for an inbound call.

        Parameters
        ----------
        call_sid   : Twilio call SID (unique per call leg)
        member     : dict from member_resolver.resolve_member(), or None for
                     guest callers
        phone_e164 : normalised caller phone number (for correlation; not logged)
        """
        session = _empty_session(call_sid, member, phone_e164)
        await self._save(call_sid, session)
        logger.info(
            "Session created  call_sid=%s  member=%s",
            call_sid,
            session["member_id"] or "guest",
        )

    async def get_session(self, call_sid: str) -> Optional[dict]:
        """Return the current session dict, or None if not found."""
        return await self._load(call_sid)

    async def append_turn(
        self,
        call_sid: str,
        role:     str,   # "user" | "assistant"
        content:  str,
    ) -> None:
        """
        Append a turn to the session transcript and refresh the TTL.

        Also updates live sentiment if role == "user".
        """
        session = await self._load(call_sid)
        if session is None:
            logger.warning("append_turn: session not found  call_sid=%s", call_sid)
            return

        turn = {"role": role, "content": content, "ts": time.time()}
        session["transcript"].append(turn)

        if role == "user":
            session["member_sentiment"] = _infer_live_sentiment(
                content, session["member_sentiment"]
            )

        await self._save(call_sid, session)

    async def update_sentiment(self, call_sid: str, sentiment: str) -> None:
        """
        Explicitly set the session sentiment (called from guardrails or coach).

        Only escalates toward more negative states (same rule as live heuristic).
        """
        session = await self._load(call_sid)
        if session is None:
            return
        current = session.get("member_sentiment", "neutral")
        if _SENTIMENT_PRIORITY.index(sentiment) < _SENTIMENT_PRIORITY.index(current):
            session["member_sentiment"] = sentiment
            await self._save(call_sid, session)

    async def mark_escalated(self, call_sid: str, escalation_type: str) -> None:
        """Record that this call was escalated to a human agent."""
        session = await self._load(call_sid)
        if session is None:
            return
        session["escalated"]       = True
        session["escalation_type"] = escalation_type
        # Crisis escalation always sets sentiment to distressed
        if escalation_type == "crisis":
            session["member_sentiment"] = "distressed"
        elif escalation_type == "financial_hardship" and session["member_sentiment"] == "neutral":
            session["member_sentiment"] = "concerned"
        await self._save(call_sid, session)

    async def add_topic(self, call_sid: str, topic: str) -> None:
        """Append a topic string to topics_discussed (de-duplicated)."""
        session = await self._load(call_sid)
        if session is None:
            return
        if topic and topic not in session["topics_discussed"]:
            session["topics_discussed"].append(topic)
            await self._save(call_sid, session)

    async def add_action(self, call_sid: str, action: str) -> None:
        """Append a promised action item string to actions_promised."""
        session = await self._load(call_sid)
        if session is None:
            return
        if action:
            session["actions_promised"].append(action)
            await self._save(call_sid, session)

    async def end_session(self, call_sid: str) -> Optional[dict]:
        """
        Finalise the session:
        1. Load session from Redis.
        2. Generate a structured summary via Claude.
        3. Persist summary to Postgres.
        4. Create a CRM follow-up task if required.
        5. Delete the Redis key.

        Returns the summary dict on success, None if the session was not found.
        """
        session = await self._load(call_sid)
        if session is None:
            logger.warning("end_session: session not found  call_sid=%s", call_sid)
            return None

        end_time  = time.time()
        duration  = int(end_time - session["start_time"])

        logger.info(
            "Session ending  call_sid=%s  member=%s  duration=%ds  turns=%d",
            call_sid,
            session["member_id"] or "guest",
            duration,
            len(session["transcript"]),
        )

        # ── 1. Generate Claude summary ────────────────────────────────────────
        summary = await self.generate_session_summary(session, duration)

        # ── 2. Persist to DB ──────────────────────────────────────────────────
        if self._engine:
            await self._save_to_db(call_sid, session, summary, end_time, duration)
        else:
            logger.warning("end_session: no DB engine — summary not persisted.")

        # ── 3. CRM follow-up task ─────────────────────────────────────────────
        if summary.get("follow_up_required") and session.get("member_id"):
            await self._create_crm_task(
                member_id           = session["member_id"],
                member_name         = session.get("member_name", ""),
                call_sid            = call_sid,
                follow_up_description = summary.get("follow_up_description", ""),
                action_items        = summary.get("action_items", []),
            )

        # ── 4. Clean up Redis ─────────────────────────────────────────────────
        await self._delete(call_sid)

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # Claude session summary
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_session_summary(
        self,
        session:  dict,
        duration: int = 0,
    ) -> dict:
        """
        Use Claude to extract a structured summary from the call transcript.

        Returns a dict matching the contract:
        {
            call_duration:        int,          # seconds
            topics_covered:       list[str],
            member_questions:     list[str],
            information_provided: list[str],
            action_items:         list[{task: str, owner: str}],
            member_sentiment:     "positive"|"neutral"|"concerned"|"distressed",
            follow_up_required:   bool,
            follow_up_description: str | null,
        }

        Falls back to a minimal dict built from session fields if the Claude
        call fails (DB insert always succeeds).
        """
        transcript = session.get("transcript", [])

        if not transcript:
            logger.debug("generate_session_summary: empty transcript — returning minimal summary.")
            return _minimal_summary(session, duration)

        # Format transcript as a readable dialogue for Claude
        dialogue = "\n".join(
            f"{'MEMBER' if t['role'] == 'user' else 'COACH'}: {t['content']}"
            for t in transcript
        )

        prompt = f"""You are analysing a call transcript from a credit union AI financial coach.

CALL TRANSCRIPT:
{dialogue}

LIVE SENTIMENT SIGNAL: {session.get("member_sentiment", "neutral")}
ESCALATED: {session.get("escalated", False)} ({session.get("escalation_type") or "n/a"})
CALL DURATION: {duration} seconds

Extract the following and respond ONLY with a valid JSON object — no markdown, no explanation:

{{
  "topics_covered":        ["<topic>", ...],
  "member_questions":      ["<verbatim or paraphrased question>", ...],
  "information_provided":  ["<key piece of info the coach gave>", ...],
  "action_items":          [{{"task": "<what needs to happen>", "owner": "member|CU|specialist"}}],
  "member_sentiment":      "positive"|"neutral"|"concerned"|"distressed",
  "follow_up_required":    true|false,
  "follow_up_description": "<one sentence describing the follow-up, or null>"
}}

Rules:
- topics_covered: 1–8 short labels (e.g. "CD renewal", "loan payment", "savings rate")
- member_questions: only actual questions the member asked; omit small talk
- action_items: only concrete tasks with a clear owner; omit vague statements
- member_sentiment: choose the WORST sentiment displayed at any point in the call
- follow_up_required: true if any action item has owner "CU" or "specialist", or if
  the member mentioned a time-sensitive event (payment due, CD maturing) without resolution
- follow_up_description: null if follow_up_required is false
"""

        try:
            client = self._get_claude_client()
            response = await client.messages.create(
                model      = ANTHROPIC_MODEL,
                max_tokens = 800,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip accidental markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data: dict = json.loads(raw)

        except json.JSONDecodeError as exc:
            logger.warning(
                "generate_session_summary: JSON parse failed  call_sid=%s: %s",
                session.get("call_sid"), exc,
            )
            data = {}
        except Exception as exc:
            logger.error(
                "generate_session_summary: Claude error  call_sid=%s: %s",
                session.get("call_sid"), exc,
            )
            data = {}

        # Merge Claude output with known session fields; fall back for missing keys
        return {
            "call_duration":         duration,
            "topics_covered":        data.get("topics_covered",        session.get("topics_discussed", [])),
            "member_questions":      data.get("member_questions",       []),
            "information_provided":  data.get("information_provided",   []),
            "action_items":          data.get("action_items",           []),
            "member_sentiment":      data.get("member_sentiment",       session.get("member_sentiment", "neutral")),
            "follow_up_required":    data.get("follow_up_required",     bool(session.get("actions_promised"))),
            "follow_up_description": data.get("follow_up_description",  None),
        }

    def _get_claude_client(self) -> AsyncAnthropic:
        if self._claude is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._claude = AsyncAnthropic(api_key=api_key)
        return self._claude

    # ─────────────────────────────────────────────────────────────────────────
    # Database persistence
    # ─────────────────────────────────────────────────────────────────────────

    async def _save_to_db(
        self,
        call_sid:   str,
        session:    dict,
        summary:    dict,
        end_time:   float,
        duration:   int,
    ) -> None:
        """Insert the session summary row into member_coaching_sessions."""
        from datetime import datetime, timezone

        row = {
            "id":                    str(uuid.uuid4()),
            "call_sid":              call_sid,
            "member_id":             session.get("member_id"),
            "started_at":            datetime.fromtimestamp(session["start_time"], tz=timezone.utc),
            "ended_at":              datetime.fromtimestamp(end_time,              tz=timezone.utc),
            "duration_s":            duration,
            "topics_covered":        summary.get("topics_covered",        []),
            "member_questions":      summary.get("member_questions",       []),
            "information_provided":  summary.get("information_provided",   []),
            "action_items":          summary.get("action_items",           []),
            "member_sentiment":      summary.get("member_sentiment",       "neutral"),
            "follow_up_required":    summary.get("follow_up_required",     False),
            "follow_up_description": summary.get("follow_up_description",  None),
            "escalated":             session.get("escalated",              False),
            "escalation_type":       session.get("escalation_type",        None),
            "raw_transcript":        session.get("transcript", []) if STORE_RAW_TRANSCRIPT else None,
        }

        try:
            async with self._engine.begin() as conn:
                await conn.execute(sa.insert(_sessions_table).values(**row))
            logger.info(
                "Session saved to DB  call_sid=%s  sentiment=%s  follow_up=%s",
                call_sid,
                row["member_sentiment"],
                row["follow_up_required"],
            )
        except Exception as exc:
            logger.error("DB insert failed  call_sid=%s: %s", call_sid, exc)

    # ─────────────────────────────────────────────────────────────────────────
    # CRM follow-up task
    # ─────────────────────────────────────────────────────────────────────────

    async def _create_crm_task(
        self,
        member_id:             str,
        member_name:           str,
        call_sid:              str,
        follow_up_description: str,
        action_items:          list[dict],
    ) -> None:
        """
        Create a follow-up task in the configured CRM system.

        Supported providers (CRM_PROVIDER env var):
          d365  — Microsoft Dynamics 365 Web API
          p71   — P71 CRM REST API (same base URL as P31)
          none  — disabled (default)
        """
        if CRM_PROVIDER == "none":
            logger.debug("CRM follow-up skipped  member=%s  provider=none", member_id)
            return

        description = (
            follow_up_description or
            "; ".join(f"{a.get('task','')} ({a.get('owner','')})" for a in action_items)
        )
        subject = f"Voice Coach Follow-Up — {member_name or member_id}"

        try:
            if CRM_PROVIDER == "d365":
                await self._create_d365_task(member_id, subject, description, call_sid)
            elif CRM_PROVIDER == "p71":
                await self._create_p71_task(member_id, subject, description, call_sid)
            else:
                logger.warning("Unknown CRM_PROVIDER=%s — task not created.", CRM_PROVIDER)
        except Exception as exc:
            # CRM failure must never crash the call-end flow
            logger.error(
                "CRM task creation failed  member=%s  provider=%s: %s",
                member_id, CRM_PROVIDER, exc,
            )

    async def _create_d365_task(
        self,
        member_id:   str,
        subject:     str,
        description: str,
        call_sid:    str,
    ) -> None:
        """Post a Task activity to Dynamics 365 Web API."""
        if not D365_BASE_URL or not D365_ACCESS_TOKEN:
            logger.warning("D365_BASE_URL / D365_ACCESS_TOKEN not set — task skipped.")
            return

        payload = {
            "subject":        subject,
            "description":    description,
            "regardingobjectid_account@odata.bind": f"/contacts({member_id})",
            "scheduledend":   _iso_tomorrow(),
            "prioritycode":   2,    # 0=low 1=normal 2=high
            "statecode":      0,    # open
        }
        url = f"{D365_BASE_URL.rstrip('/')}/api/data/v9.2/tasks"
        headers = {
            "Authorization":   f"Bearer {D365_ACCESS_TOKEN}",
            "Content-Type":    "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version":   "4.0",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        logger.info(
            "D365 task created  member=%s  subject=%r  call_sid=%s",
            member_id, subject, call_sid,
        )

    async def _create_p71_task(
        self,
        member_id:   str,
        subject:     str,
        description: str,
        call_sid:    str,
    ) -> None:
        """Post a follow-up task to the P71 CRM REST API."""
        payload = {
            "member_id":   member_id,
            "subject":     subject,
            "description": description,
            "source":      "voice_coach",
            "call_sid":    call_sid,
            "due_date":    _iso_tomorrow(),
        }
        url = f"{P71_BASE_URL.rstrip('/')}/members/{member_id}/tasks"
        headers = {
            "Authorization": f"Bearer {P71_API_KEY}",
            "Content-Type":  "application/json",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        logger.info(
            "P71 task created  member=%s  subject=%r  call_sid=%s",
            member_id, subject, call_sid,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iso_tomorrow() -> str:
    """Return tomorrow's date as an ISO 8601 string (used for CRM due dates)."""
    from datetime import date, timedelta
    return (date.today() + timedelta(days=1)).isoformat()


def _minimal_summary(session: dict, duration: int) -> dict:
    """Return a minimal summary dict built purely from session fields (no Claude)."""
    return {
        "call_duration":         duration,
        "topics_covered":        session.get("topics_discussed", []),
        "member_questions":      [],
        "information_provided":  [],
        "action_items":          [{"task": a, "owner": "CU"} for a in session.get("actions_promised", [])],
        "member_sentiment":      session.get("member_sentiment", "neutral"),
        "follow_up_required":    bool(session.get("actions_promised")),
        "follow_up_description": None,
    }
