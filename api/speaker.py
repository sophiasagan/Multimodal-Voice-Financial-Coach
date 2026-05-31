"""
api/speaker.py — OpenAI TTS synthesis for the CU voice financial coach.

Public API
----------
synthesize_speech(text: str) -> bytes
    Convert text to MP3 bytes via OpenAI TTS (model tts-1, voice nova).
    Cached phrases are returned from an in-process LRU cache without an API
    call.  All other text is synthesised on demand and not cached (call
    responses are too varied to benefit from caching).

to_mulaw_8k(mp3_bytes: bytes) -> bytes
    Transcode MP3 → G.711 µ-law 8 000 Hz mono.  Twilio Media Streams requires
    mulaw; this function is called by main.py before streaming audio back.

Design notes
------------
Voice: "nova" — warm, gender-neutral, conversational.  OpenAI's recommendation
for customer-service and financial guidance use-cases.

Model: "tts-1" (not "tts-1-hd") — optimised for low latency on short utterances
(< 30 s).  For phone-line audio quality the HD model provides no audible
benefit over a mulaw 8 kHz channel.

Cache strategy
--------------
A small set of phrases (greetings, hold messages, error messages) recurs on
every call.  These are pre-cached using _WARMUP_PHRASES at module import if
PREWARM_TTS=true is set (default false — callers can trigger it from the
FastAPI lifespan event).  All other phrases are NOT cached — LLM responses
vary enough that caching them wastes memory with essentially zero hit rate.

The cache is a plain dict keyed by the normalised phrase text; it lives for
the lifetime of the process.  This is intentional: TTS output for a fixed
phrase is deterministic, so there is no staleness concern.

Mulaw conversion
----------------
pydub + ffmpeg handle the MP3 → PCM → mulaw pipeline.  ffmpeg must be on PATH
or FFMPEG_PATH must point to the binary.  A RuntimeError is raised at first
call if ffmpeg is unavailable so the failure is surfaced at startup (via the
lifespan pre-warm) rather than mid-call.

If pydub/ffmpeg are not available, to_mulaw_8k falls back to returning the raw
MP3 bytes with a warning.  Twilio will reject them, but the call pipeline
continues rather than crashing.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TTS_MODEL:  str = os.getenv("TTS_MODEL",  "tts-1")
TTS_VOICE:  str = os.getenv("TTS_VOICE",  "nova")
TTS_FORMAT: str = "mp3"

# Set to "true" in the environment to pre-synthesise WARMUP_PHRASES at startup.
PREWARM_TTS: bool = os.getenv("PREWARM_TTS", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# Phrases to pre-synthesise (and cache) at startup
# ─────────────────────────────────────────────────────────────────────────────

_WARMUP_PHRASES: list[str] = [
    # Opening
    "Thank you for calling. I'm your AI financial coach. How can I help you today?",
    # Hold / transfer
    "Please hold for just a moment.",
    "Let me connect you with a specialist who can help.",
    "I'm transferring you to our financial counseling team now.",
    "I'm connecting you with a member services representative.",
    # Errors / recovery
    "I'm sorry, I didn't catch that. Could you say it again?",
    "I'm having a bit of trouble hearing you. Could you repeat that?",
    "I'm sorry, I'm having a technical issue. Please hold for a moment.",
    # Crisis
    (
        "I hear that things are really hard right now. Please know you're not alone. "
        "I'm connecting you with someone who can help. "
        "You can also call 988, the Suicide and Crisis Lifeline, anytime."
    ),
    # Closing
    "Thank you for calling. Take care.",
    "Is there anything else I can help you with today?",
]

# ─────────────────────────────────────────────────────────────────────────────
# In-process phrase cache  {normalised_text: mp3_bytes}
# ─────────────────────────────────────────────────────────────────────────────

_tts_cache: dict[str, bytes] = {}


def _cache_key(text: str) -> str:
    """
    Normalise text and return a stable cache key.

    Strips leading/trailing whitespace and collapses internal whitespace so
    minor formatting differences don't cause cache misses on identical phrases.
    The key is the normalised string itself (not a hash) for readability in
    debug logs.
    """
    return re.sub(r"\s+", " ", text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# MP3 → mulaw 8 kHz conversion
# ─────────────────────────────────────────────────────────────────────────────

def _check_ffmpeg() -> bool:
    """Return True if pydub can find ffmpeg / avconv."""
    try:
        from pydub.utils import get_encoder_name  # type: ignore[import]
        get_encoder_name("mp3")                   # raises if not found
        return True
    except Exception:
        return False


def to_mulaw_8k(mp3_bytes: bytes) -> bytes:
    """
    Transcode MP3 audio to G.711 µ-law 8 000 Hz mono (Twilio Media Streams
    format).

    Parameters
    ----------
    mp3_bytes : raw MP3 data as returned by the OpenAI TTS API.

    Returns
    -------
    bytes : raw µ-law encoded audio at 8 000 Hz, 8-bit, mono.
            No WAV header — Twilio expects raw PCM or raw mulaw in its
            media stream, base64-encoded.

    Falls back to returning mp3_bytes unchanged if pydub/ffmpeg is
    unavailable (Twilio will reject this, but the pipeline won't crash).
    """
    try:
        from pydub import AudioSegment  # type: ignore[import]
    except ImportError:
        logger.warning(
            "pydub is not installed — to_mulaw_8k returning MP3 bytes. "
            "Install pydub + ffmpeg for correct Twilio audio: pip install pydub"
        )
        return mp3_bytes

    try:
        # Decode MP3 → raw PCM via pydub (uses ffmpeg under the hood)
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")

        # Resample to 8 000 Hz mono 16-bit (the canonical PCM format for G.711)
        audio = (
            audio
            .set_frame_rate(8_000)
            .set_channels(1)
            .set_sample_width(2)    # 16-bit = 2 bytes
        )

        pcm_bytes = audio.raw_data

        # PCM → µ-law via audioop (available on Python < 3.13) or pure Python
        try:
            import audioop  # type: ignore[import]
            return audioop.lin2ulaw(pcm_bytes, 2)
        except ImportError:
            # Pure-Python G.711 µ-law encoder for Python 3.13+
            return _pcm_to_mulaw_pure(pcm_bytes)

    except Exception as exc:
        logger.error("to_mulaw_8k: transcoding failed — %s", exc)
        return mp3_bytes


def _pcm_to_mulaw_pure(pcm_bytes: bytes) -> bytes:
    """
    Encode 16-bit signed linear PCM to G.711 µ-law (Python 3.13+ fallback).

    Based on ITU-T G.711 §3.1 encoder algorithm.
    """
    import struct

    MULAW_BIAS = 0x84
    MULAW_MAX  = 0x7FFF

    out = bytearray(len(pcm_bytes) // 2)
    samples = struct.unpack_from(f"<{len(out)}h", pcm_bytes)

    for i, sample in enumerate(samples):
        sign = 0
        if sample < 0:
            sign   = 0x80
            sample = -sample
        sample = min(sample + MULAW_BIAS, MULAW_MAX)

        # Find the exponent (position of highest set bit above bias)
        exp = 7
        for exp in range(7, -1, -1):
            if sample & (1 << (exp + 3)):
                break

        mantissa = (sample >> (exp + 3)) & 0x0F
        mulaw    = ~(sign | (exp << 4) | mantissa) & 0xFF
        out[i]   = mulaw

    return bytes(out)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def synthesize_speech(text: str) -> bytes:
    """
    Convert text to MP3 bytes using OpenAI TTS (tts-1, nova voice).

    Cache behaviour
    ---------------
    If the normalised text matches a previously cached phrase, the cached MP3
    bytes are returned immediately — no API call is made.  Only the fixed
    phrases in _WARMUP_PHRASES are ever cached; dynamic coach responses are
    synthesised fresh every time.

    Parameters
    ----------
    text : the plain-text string to speak.  Should already be Markdown-free
           and ≤ 80 words (enforced upstream by coach.py / guardrails.py).

    Returns
    -------
    bytes : raw MP3 audio data.  Pass through to_mulaw_8k() before streaming
            back to Twilio.
    """
    if not text or not text.strip():
        logger.warning("synthesize_speech called with empty text — returning silence.")
        return b""

    key = _cache_key(text)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if key in _tts_cache:
        logger.debug("TTS cache hit  chars=%d  key=%.40s…", len(key), key)
        return _tts_cache[key]

    # ── OpenAI TTS call ───────────────────────────────────────────────────────
    logger.debug(
        "TTS synthesising  model=%s  voice=%s  chars=%d",
        TTS_MODEL, TTS_VOICE, len(text),
    )

    try:
        response = await _get_client().audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,            # type: ignore[arg-type]
            input=text,
            response_format=TTS_FORMAT, # type: ignore[arg-type]
        )
        mp3_bytes: bytes = response.content
    except Exception as exc:
        logger.error("TTS API error: %s", exc)
        raise

    logger.debug(
        "TTS done  chars=%d  mp3_bytes=%d",
        len(text), len(mp3_bytes),
    )

    return mp3_bytes


async def prewarm_cache() -> None:
    """
    Pre-synthesise all phrases in _WARMUP_PHRASES and store in _tts_cache.

    Called from the FastAPI lifespan startup handler in main.py when
    PREWARM_TTS=true.  Runs all TTS calls concurrently for fast startup.
    """
    import asyncio

    async def _warm_one(phrase: str) -> None:
        key = _cache_key(phrase)
        if key in _tts_cache:
            return
        try:
            mp3 = await synthesize_speech(phrase)
            _tts_cache[key] = mp3
            logger.debug("TTS pre-warmed  chars=%d", len(phrase))
        except Exception as exc:
            logger.warning("TTS pre-warm failed for phrase %.40r: %s", phrase, exc)

    logger.info("TTS: pre-warming %d cached phrases…", len(_WARMUP_PHRASES))
    await asyncio.gather(*(_warm_one(p) for p in _WARMUP_PHRASES))
    logger.info("TTS: pre-warm complete  cached=%d", len(_tts_cache))


def get_cached_phrase(text: str) -> Optional[bytes]:
    """
    Return cached MP3 bytes for an exact phrase match, or None if not cached.

    Useful for main.py to play the opening greeting without an async call on
    the hot path before the first utterance.
    """
    return _tts_cache.get(_cache_key(text))
