"""
api/guardrails.py — Safety and escalation checks for the CU voice coach.

Architecture
------------
check_guardrails(transcript) is called on every member utterance BEFORE the
transcript is sent to Claude.  If it returns escalate=True the caller (main.py
/ call_handler.py) must:

  1. Speak the response_override directly (no Claude call).
  2. Initiate the appropriate transfer action (human agent, counseling line,
     crisis line) based on escalation_type.
  3. Do NOT call generate_response() for this turn.

This ordering is intentional: guardrails must fire before Claude, not after.
A harmful or mis-classified reply from Claude should never reach the member.

Check hierarchy (highest priority first)
-----------------------------------------
1. crisis            — personal distress / self-harm → 988 + immediate transfer
2. financial_hardship — inability to pay, bankruptcy  → financial counseling
3. complaint         — escalation / complaint language → member services
4. investment_advice — securities, crypto, specific stock picks → licensed advisor
5. escalation_phrases — catch-all from prompts/escalation_phrases.txt (loaded
                         dynamically; supplements the hard-coded lists above)

The checks are intentionally over-inclusive (false-positives are safer than
false-negatives).  A member who mentions "my uncle went bankrupt" will trigger
the financial_hardship path — the counselor can quickly determine it was
off-topic and redirect.  The alternative (missing a genuine hardship case) is
much worse.

Return contract
---------------
{
    "escalate":         bool,           # True → stop pipeline, use override
    "escalation_type":  str | None,     # category string or None
    "response_override": str | None,    # verbatim text to speak to member
}

Adding new categories
---------------------
Add a new block to _RULES (order matters — first match wins) and a
corresponding spoken response in _RESPONSES.  Phrase lists support plain
substrings (case-insensitive) and basic regex patterns prefixed with "re:".
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Spoken response overrides
# ─────────────────────────────────────────────────────────────────────────────

_RESPONSES: dict[str, str] = {
    "crisis": (
        "I hear that things are really hard right now. "
        "Please know you're not alone. "
        "I'm connecting you with someone who can help. "
        "You can also call or text 988, the Suicide and Crisis Lifeline, anytime."
    ),
    "financial_hardship": (
        "I want to make sure you get the right help. "
        "Let me connect you with one of our financial counselors "
        "who can walk through your options with you."
    ),
    "complaint": (
        "I'm sorry to hear you're not satisfied. "
        "Let me connect you with a member services representative "
        "who can address this directly."
    ),
    "investment_advice": (
        "For investment decisions I'd recommend speaking with a licensed "
        "financial advisor who knows your full picture. "
        "I can connect you with our investment services team — would that be helpful?"
    ),
    "escalation": (
        "Of course. Let me connect you with a team member right away."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Trigger phrase lists
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is either a plain substring (matched case-insensitively) or a
# string starting with "re:" followed by a Python regex pattern.

_RULES: list[tuple[str, list[str]]] = [
    # ── 1. Personal crisis (highest priority) ─────────────────────────────────
    (
        "crisis",
        [
            "hurt myself",
            "kill myself",
            "end my life",
            "don't want to live",
            "thinking about suicide",
            "no reason to go on",
            "can't go on",
            "want to die",
            "take my own life",
            "re:i\\s+(don'?t|do\\s+not)\\s+want\\s+to\\s+(be\\s+here|live)",
            "don't know what to do anymore",   # catch-all distress
        ],
    ),

    # ── 2. Financial hardship ─────────────────────────────────────────────────
    (
        "financial_hardship",
        [
            "can't pay",
            "cannot pay",
            "can't afford",
            "cannot afford",
            "losing my home",
            "going to lose my home",
            "going to lose the house",
            "about to be evicted",
            "being evicted",
            "lost my job",
            "laid off",
            "i was fired",
            "behind on my",
            "behind on payments",
            "drowning in debt",
            "buried in debt",
            "bankruptcy",
            "filing for bankruptcy",
            "can't make my payment",
            "cannot make my payment",
            "re:i\\s+(am|'?m)\\s+(struggling|really\\s+struggling)",
        ],
    ),

    # ── 3. Complaint / escalation request ─────────────────────────────────────
    (
        "complaint",
        [
            "speak to a manager",
            "talk to a manager",
            "talk to your supervisor",
            "speak to your supervisor",
            "this is unacceptable",
            "this is ridiculous",
            "i want to complain",
            "make a complaint",
            "file a complaint",
            "contact the cfpb",
            "contact your regulator",
            "this is illegal",
            "my lawyer",
            "small claims",
            "report you",
        ],
    ),

    # ── 4. Investment / securities advice ─────────────────────────────────────
    (
        "investment_advice",
        [
            "should i invest in",
            "should i buy",
            "is this stock",
            "what stock should",
            "crypto",
            "bitcoin",
            "ethereum",
            "nft",
            "hedge fund",
            "options trading",
            "day trading",
            "short selling",
            "pick a stock",
            "stock tip",
            "re:is\\s+\\w+\\s+(a\\s+good\\s+)?(stock|investment|buy)",
        ],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Optional: load extra escalation phrases from prompts/escalation_phrases.txt
# ─────────────────────────────────────────────────────────────────────────────

_PHRASES_FILE = (
    Path(__file__).resolve().parent.parent / "prompts" / "escalation_phrases.txt"
)

# Extra plain-string phrases loaded from the flat file, all mapped to the
# generic "escalation" category.
_extra_escalation_phrases: list[str] = []


def _load_escalation_phrases() -> None:
    """
    Parse prompts/escalation_phrases.txt into _extra_escalation_phrases.

    Lines starting with '#' or empty are ignored.  Called once at module load.
    """
    global _extra_escalation_phrases
    try:
        lines = _PHRASES_FILE.read_text(encoding="utf-8").splitlines()
        phrases = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        _extra_escalation_phrases = phrases
        logger.debug(
            "guardrails: loaded %d escalation phrases from %s",
            len(phrases), _PHRASES_FILE.name,
        )
    except FileNotFoundError:
        logger.debug("guardrails: %s not found — skipping extra phrases.", _PHRASES_FILE.name)
    except Exception as exc:
        logger.warning("guardrails: could not load escalation phrases: %s", exc)


_load_escalation_phrases()

# ─────────────────────────────────────────────────────────────────────────────
# Matching engine
# ─────────────────────────────────────────────────────────────────────────────

def _compile_pattern(phrase: str) -> re.Pattern:
    """
    Compile a phrase into a regex.

    Phrases starting with "re:" are treated as raw regex patterns.
    All others are converted to word-boundary-aware case-insensitive patterns.
    """
    if phrase.startswith("re:"):
        return re.compile(phrase[3:], re.IGNORECASE)
    # Escape and allow word-boundary matching around the phrase
    escaped = re.escape(phrase)
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


# Pre-compile all rule patterns at module load for fast per-call matching.
_compiled_rules: list[tuple[str, list[re.Pattern]]] = [
    (category, [_compile_pattern(p) for p in phrases])
    for category, phrases in _RULES
]

_compiled_extra: list[re.Pattern] = [
    _compile_pattern(p) for p in _extra_escalation_phrases
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_guardrails(transcript: str) -> dict:
    """
    Evaluate a member utterance against all safety and escalation rules.

    Must be called BEFORE generate_response() on every turn.

    Parameters
    ----------
    transcript : str
        The member's verbatim (or Whisper-transcribed) utterance for this turn.

    Returns
    -------
    dict with keys:
        escalate         (bool)         — True if pipeline should halt and
                                          speak response_override instead of
                                          calling Claude.
        escalation_type  (str | None)   — Category string ("crisis",
                                          "financial_hardship", "complaint",
                                          "investment_advice", "escalation")
                                          or None if no trigger matched.
        response_override (str | None)  — Verbatim text to speak to the member,
                                          or None if escalate is False.

    Examples
    --------
    >>> check_guardrails("I can't pay my mortgage")
    {'escalate': True, 'escalation_type': 'financial_hardship', 'response_override': '...'}

    >>> check_guardrails("What's my checking balance?")
    {'escalate': False, 'escalation_type': None, 'response_override': None}
    """
    if not transcript or not transcript.strip():
        return {"escalate": False, "escalation_type": None, "response_override": None}

    # ── Check primary rule set (ordered by priority) ──────────────────────────
    for category, patterns in _compiled_rules:
        for pattern in patterns:
            if pattern.search(transcript):
                logger.info(
                    "guardrails: TRIGGERED  category=%s  pattern=%r  text=%.80r",
                    category, pattern.pattern, transcript,
                )
                return {
                    "escalate":          True,
                    "escalation_type":   category,
                    "response_override": _RESPONSES[category],
                }

    # ── Check extra phrases from escalation_phrases.txt ───────────────────────
    for pattern in _compiled_extra:
        if pattern.search(transcript):
            logger.info(
                "guardrails: TRIGGERED (extra)  pattern=%r  text=%.80r",
                pattern.pattern, transcript,
            )
            return {
                "escalate":          True,
                "escalation_type":   "escalation",
                "response_override": _RESPONSES["escalation"],
            }

    # ── No trigger matched ────────────────────────────────────────────────────
    logger.debug("guardrails: clear  text=%.80r", transcript)
    return {
        "escalate":          False,
        "escalation_type":   None,
        "response_override": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience accessors (used by call_handler.py / tests)
# ─────────────────────────────────────────────────────────────────────────────

def get_response_for(escalation_type: str) -> Optional[str]:
    """
    Return the pre-written spoken response for an escalation type.

    Useful when call_handler.py determines the escalation_type independently
    (e.g. from a DTMF press) and needs the corresponding script.
    Returns None if the type is unknown.
    """
    return _RESPONSES.get(escalation_type)


def list_categories() -> list[str]:
    """Return the names of all recognised escalation categories."""
    return list(_RESPONSES.keys())
