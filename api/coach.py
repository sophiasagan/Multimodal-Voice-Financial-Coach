"""
api/coach.py — Claude response generation for the CU voice financial coach.

Responsibilities
----------------
1. Load the coach system prompt from prompts/coach_system.txt once at startup.
2. Accept a member utterance + pre-built account context + conversation history.
3. Call Claude (claude-sonnet-4-6 by default; override with ANTHROPIC_MODEL).
4. Enforce the ≤ 80-word hard limit — truncate at a sentence boundary if needed.
5. Strip Markdown and HTML so OpenAI TTS doesn't read asterisks aloud.
6. Return plain prose ready for immediate TTS synthesis.

Prompt caching
--------------
The system prompt (which includes the member's account context) is annotated
with cache_control {"type": "ephemeral"}.  Sonnet 4.6 requires a 2 048-token
minimum prefix to activate cache writes; our prompt is typically 400–600 tokens,
so cache hits will only fire once the account context grows that block to the
threshold.  The annotation is a no-op below the threshold and costs nothing.

History management
------------------
history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
We pass only the last MAX_HISTORY_TURNS pairs (default 6) to keep token counts
low and avoid stale context confusing the model.

Word-limit enforcement
-----------------------
Claude's max_tokens ceiling (150) keeps the raw output small, but the model may
still occasionally run 90–110 words.  _trim_to_word_limit() does a final pass:
  1. Split on sentence-ending punctuation, accumulate until > 80 words.
  2. Return the last complete sentence that kept us at or under 80 words.
  3. If even the first sentence is > 80 words, hard-truncate at word 80 and
     append "…" so TTS doesn't cut off mid-thought.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
CU_NAME:         str = os.getenv("CU_NAME", "your credit union")

# Hard word limit for phone delivery.  Callers cannot re-read; concise wins.
WORD_LIMIT: int = 80

# Sliding-window history: keep only the most recent N *pairs* (user + assistant)
# to cap context length and avoid stale exchanges confusing the model.
MAX_HISTORY_TURNS: int = int(os.getenv("COACH_MAX_HISTORY_TURNS", "6"))

# Claude's per-response ceiling in tokens.  80 words ≈ 110 tokens; 150 gives
# a comfortable buffer for multi-syllabic financial terms without risking
# run-on answers.
MAX_TOKENS: int = 150

# Path to the prompt template (resolved relative to this file so the module
# works regardless of the working directory uvicorn is launched from).
_PROMPT_DIR   = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_TMPL  = _PROMPT_DIR / "coach_system.txt"

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — loaded once at module import
# ─────────────────────────────────────────────────────────────────────────────

def _load_system_template() -> str:
    """
    Read coach_system.txt.  Falls back to a minimal inline prompt if the file
    is missing so unit tests and first-run imports don't hard-crash.
    """
    try:
        return _SYSTEM_TMPL.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            "coach.py: %s not found — using minimal inline system prompt.",
            _SYSTEM_TMPL,
        )
        return (
            "You are a friendly financial coach for {cu_name} speaking on the "
            "phone. Member context: {context} "
            "Respond in under 80 words. Be concise and helpful."
        )


_SYSTEM_TEMPLATE: str = _load_system_template()


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic client (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[AsyncAnthropic] = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        _client = AsyncAnthropic(api_key=api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Text cleanup helpers
# ─────────────────────────────────────────────────────────────────────────────

# Remove Markdown emphasis, headers, code, HTML tags — anything TTS would read
# literally and sound wrong (e.g. "asterisk asterisk important asterisk asterisk")
_MARKDOWN_RE = re.compile(r"[*_`#~]|<[^>]+>|\[([^\]]+)\]\([^)]+\)")

# Sentence-boundary pattern: split after . ! ? followed by whitespace or end.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_markdown(text: str) -> str:
    """Remove Markdown/HTML tokens that would be spoken literally by TTS."""
    # Replace [link text](url) with just the link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip remaining punctuation markers
    text = re.sub(r"[*_`#~]|<[^>]+>", "", text)
    # Collapse extra whitespace
    return re.sub(r" {2,}", " ", text).strip()


def _trim_to_word_limit(text: str, limit: int = WORD_LIMIT) -> str:
    """
    Trim text to at most `limit` words, preserving sentence boundaries.

    Strategy
    --------
    1. If the full text is within the limit — return as-is.
    2. Split into sentences; accumulate until adding the next sentence would
       exceed the limit; return what we have so far.
    3. If the very first sentence exceeds the limit — hard-truncate at word
       `limit` and append "…".

    Parameters
    ----------
    text  : cleaned plain-text string
    limit : word count ceiling (default: WORD_LIMIT = 80)
    """
    words = text.split()
    if len(words) <= limit:
        return text

    sentences = _SENTENCE_SPLIT_RE.split(text)
    accumulated: list[str] = []
    word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if word_count + sentence_words > limit:
            break
        accumulated.append(sentence)
        word_count += sentence_words

    if accumulated:
        return " ".join(accumulated)

    # First sentence alone is too long — hard truncate
    return " ".join(words[:limit]) + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def generate_response(
    transcript:      str,
    account_context: str,
    history:         list[dict],
    call_sid:        Optional[str] = None,
) -> str:
    """
    Generate a phone-ready financial coaching response from Claude.

    Parameters
    ----------
    transcript
        The member's transcribed utterance for this turn (from Whisper).
    account_context
        Context block built by context_builder.build_context(); contains the
        member's product summary, financial health score, and AI twin insights.
    history
        Alternating list of {"role": "user"|"assistant", "content": "..."}
        dicts representing the conversation so far (excluding this turn).
        Passed through a sliding-window filter (last MAX_HISTORY_TURNS pairs).
    call_sid
        Twilio call SID; included in log messages for correlation only.

    Returns
    -------
    str
        Plain prose, ≤ 80 words, Markdown-free, ready for TTS synthesis.
        Falls back to a generic error string if the API call fails.

    Raises
    ------
    Does not raise — all exceptions are caught and logged.  The fallback
    string returned is safe to speak aloud on the phone.
    """
    # ── 1. Build system prompt ────────────────────────────────────────────────
    system_text = _SYSTEM_TEMPLATE.format(
        cu_name=CU_NAME,
        context=account_context,
    )

    # Annotate the full system prompt for prompt caching.
    # Sonnet 4.6 minimum cacheable prefix is 2 048 tokens; this annotation is
    # a no-op below that threshold and costs nothing when not hit.
    system_block: list[dict] = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # ── 2. Sliding-window history ─────────────────────────────────────────────
    # Keep at most MAX_HISTORY_TURNS user+assistant pairs.
    # Clip from the front (oldest) so the most recent context is always present.
    max_messages = MAX_HISTORY_TURNS * 2   # each "turn" = 1 user + 1 assistant
    recent_history = history[-max_messages:] if len(history) > max_messages else history

    # Validate: Claude requires strictly alternating roles starting with "user".
    # Drop leading assistant messages if history was mis-assembled upstream.
    sanitized: list[dict] = []
    expected_role = "user"
    for msg in recent_history:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role not in ("user", "assistant") or not content:
            continue
        if role != expected_role:
            # Skip out-of-order messages rather than sending a malformed request
            logger.debug(
                "coach: skipping out-of-order history message role=%s expected=%s",
                role, expected_role,
            )
            continue
        sanitized.append({"role": role, "content": str(content)})
        expected_role = "assistant" if expected_role == "user" else "user"

    # Append the current user turn
    messages = sanitized + [{"role": "user", "content": transcript}]

    # ── 3. Claude API call ────────────────────────────────────────────────────
    logger.debug(
        "coach: calling Claude  model=%s  history_turns=%d  call_sid=%s",
        ANTHROPIC_MODEL,
        len(sanitized) // 2,
        call_sid or "—",
    )

    try:
        response = await _get_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_block,
            messages=messages,
        )
    except Exception as exc:
        logger.error(
            "Claude API error  call_sid=%s: %s", call_sid or "—", exc
        )
        return (
            "I'm sorry, I'm having trouble connecting right now. "
            "Please hold for a moment, or press zero to speak with a representative."
        )

    # ── 4. Extract text ───────────────────────────────────────────────────────
    raw_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_text += block.text

    raw_text = raw_text.strip()

    if not raw_text:
        logger.warning(
            "Claude returned empty content  call_sid=%s  stop_reason=%s",
            call_sid or "—",
            response.stop_reason,
        )
        return "Could you say that again? I didn't quite catch it."

    # ── 5. Clean for TTS + enforce word limit ─────────────────────────────────
    clean = _strip_markdown(raw_text)
    final = _trim_to_word_limit(clean)

    word_count = len(final.split())
    logger.info(
        "coach: response  call_sid=%s  words=%d  stop=%s  text=%r",
        call_sid or "—",
        word_count,
        response.stop_reason,
        final[:120],
    )

    return final
