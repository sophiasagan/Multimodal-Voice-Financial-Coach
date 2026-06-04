"""
api/speaker.py — OpenAI TTS synthesis for the CU voice financial coach.

Public API
----------
synthesize_speech(text) -> bytes
    Returns raw 16-bit PCM audio at 24 000 Hz mono.
    Cached for warmup phrases; fresh synthesis for all other text.

to_mulaw_8k(pcm_bytes) -> bytes
    Downsample 24 kHz PCM → 8 kHz and encode to G.711 µ-law.
    Pure Python — no pydub, no ffmpeg, no audioop.

synthesize_to_wav(text) -> bytes
    Returns a complete WAV file (RIFF header + PCM).
    Used for the static opening greeting served to Twilio via <Play>.

Design
------
OpenAI TTS supports response_format="pcm" which returns raw signed 16-bit
little-endian PCM at 24 000 Hz mono.  Receiving PCM directly eliminates the
MP3 → PCM decode step and removes the pydub/ffmpeg dependency entirely.

Downsampling 24 kHz → 8 kHz uses simple averaging of every 3 samples
(factor = 3).  This is sufficient for telephone-quality voice — the bandwidth
of a mulaw phone channel is ~3.5 kHz, well below the 4 kHz Nyquist limit of
8 kHz sampling.

G.711 µ-law encoding is implemented from the ITU-T G.711 §3.1 algorithm.
"""

from __future__ import annotations

import io
import logging
import os
import re
import struct
import wave
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TTS_MODEL:       str = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE:       str = os.getenv("TTS_VOICE", "nova")

PCM_SAMPLE_RATE: int = 24_000   # OpenAI TTS PCM output rate (fixed)
MULAW_RATE:      int = 8_000    # Twilio mulaw rate (fixed)
DOWNSAMPLE_FACTOR = PCM_SAMPLE_RATE // MULAW_RATE   # = 3

# ─────────────────────────────────────────────────────────────────────────────
# Warmup phrases — cached as PCM on first call (or at startup if PREWARM_TTS)
# ─────────────────────────────────────────────────────────────────────────────

_WARMUP_PHRASES: list[str] = [
    "Thank you for calling. I'm your AI financial coach. How can I help you today?",
    "Please hold for just a moment.",
    "Let me connect you with a specialist who can help.",
    "I'm transferring you to our financial counseling team now.",
    "I'm connecting you with a member services representative.",
    "I'm sorry, I didn't catch that. Could you say it again?",
    "I'm having a bit of trouble hearing you. Could you repeat that?",
    "I'm sorry, I'm having a technical issue. Please hold for a moment.",
    (
        "I hear that things are really hard right now. Please know you're not alone. "
        "I'm connecting you with someone who can help. "
        "You can also call 988, the Suicide and Crisis Lifeline, anytime."
    ),
    "Thank you for calling. Take care.",
    "Is there anything else I can help you with today?",
]

# In-process cache: normalised phrase text → raw PCM bytes (24 kHz)
_tts_cache: dict[str, bytes] = {}


def _cache_key(text: str) -> str:
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
# PCM → mulaw (pure Python, no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def _downsample(pcm_24k: bytes) -> bytes:
    """
    Downsample 24 kHz 16-bit mono PCM to 8 kHz by averaging groups of 3 samples.

    Returns 16-bit signed little-endian PCM at 8 000 Hz.
    """
    n = len(pcm_24k) // 2
    samples = struct.unpack_from(f"<{n}h", pcm_24k)
    out = bytearray()
    for i in range(0, n - DOWNSAMPLE_FACTOR + 1, DOWNSAMPLE_FACTOR):
        avg = sum(samples[i : i + DOWNSAMPLE_FACTOR]) // DOWNSAMPLE_FACTOR
        avg = max(-32768, min(32767, avg))
        out += struct.pack("<h", avg)
    return bytes(out)


# Exponent lookup table — matches the ITU-T G.711 reference and CPython audioop.
# Indexed by (sample >> 7) & 0xFF after bias is added (14-bit space).
_EXP_LUT: list[int] = [
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
]


def _pcm_to_mulaw(pcm_8k: bytes) -> bytes:
    """
    Encode 16-bit signed linear PCM at 8 kHz to raw G.711 µ-law bytes.

    Matches the ITU-T G.711 reference algorithm and CPython's audioop.lin2ulaw.
    Works in 14-bit space (right-shifts input by 2) so the segment table is
    correct — operating on raw 16-bit values with the same table produces the
    wrong exponent for most samples, causing heavy distortion.
    """
    BIAS  = 0x84 >> 2   # = 33  (bias scaled to 14-bit)
    CLIP  = 32767

    n       = len(pcm_8k) // 2
    samples = struct.unpack_from(f"<{n}h", pcm_8k)
    out     = bytearray(n)

    for i, s in enumerate(samples):
        s >>= 2                         # 16-bit → 14-bit
        if s < 0:
            s    = -s
            sign = 0                    # sign bit 0 → negative in µ-law
        else:
            sign = 0x80
        s    = min(s, CLIP)
        s   += BIAS
        exp  = _EXP_LUT[(s >> 7) & 0xFF]
        mant = (s >> (exp + 3)) & 0x0F
        out[i] = (~(sign | (exp << 4) | mant)) & 0xFF

    return bytes(out)


def to_mulaw_8k(pcm_24k: bytes) -> bytes:
    """
    Convert raw 24 kHz 16-bit PCM (from OpenAI TTS) to G.711 µ-law at 8 kHz.

    Pure Python — no pydub, no ffmpeg, no audioop.
    Called synchronously from main.py after synthesize_speech().
    """
    if not pcm_24k:
        return b""
    pcm_8k = _downsample(pcm_24k)
    return _pcm_to_mulaw(pcm_8k)


def _pcm_to_wav(pcm_24k: bytes) -> bytes:
    """Wrap raw 24 kHz PCM in a standard WAV container (for static file serving)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)           # 16-bit
        wf.setframerate(PCM_SAMPLE_RATE)
        wf.writeframes(pcm_24k)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def synthesize_speech(text: str) -> bytes:
    """
    Convert text to raw 16-bit PCM at 24 000 Hz mono via OpenAI TTS.

    Returns cached PCM for warmup phrases; calls the API for all other text.
    Pass the result to to_mulaw_8k() before streaming to Twilio.
    """
    if not text or not text.strip():
        logger.warning("synthesize_speech: empty text — returning silence.")
        return b""

    key = _cache_key(text)
    if key in _tts_cache:
        logger.debug("TTS cache hit  chars=%d", len(key))
        return _tts_cache[key]

    logger.debug("TTS  model=%s  voice=%s  chars=%d", TTS_MODEL, TTS_VOICE, len(text))

    try:
        response = await _get_client().audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,       # type: ignore[arg-type]
            input=text,
            response_format="pcm", # type: ignore[arg-type]
        )
        pcm_bytes: bytes = response.content
    except Exception as exc:
        logger.error("TTS API error: %s", exc)
        raise

    logger.debug("TTS done  chars=%d  pcm_bytes=%d", len(text), len(pcm_bytes))
    return pcm_bytes


async def synthesize_to_wav(text: str) -> bytes:
    """
    Synthesise text and return a complete WAV file.

    Used by main.py to generate the static opening greeting
    (served to Twilio via <Play> over HTTP).
    """
    pcm = await synthesize_speech(text)
    return _pcm_to_wav(pcm)


async def prewarm_cache() -> None:
    """Pre-synthesise warmup phrases concurrently at startup."""
    import asyncio

    async def _warm_one(phrase: str) -> None:
        key = _cache_key(phrase)
        if key in _tts_cache:
            return
        try:
            _tts_cache[key] = await synthesize_speech(phrase)
            logger.debug("TTS pre-warmed  chars=%d", len(phrase))
        except Exception as exc:
            logger.warning("TTS pre-warm failed %.40r: %s", phrase, exc)

    logger.info("TTS: pre-warming %d phrases…", len(_WARMUP_PHRASES))
    await asyncio.gather(*(_warm_one(p) for p in _WARMUP_PHRASES))
    logger.info("TTS: pre-warm complete  cached=%d", len(_tts_cache))


def get_cached_phrase(text: str) -> Optional[bytes]:
    """Return cached PCM for a warmup phrase, or None."""
    return _tts_cache.get(_cache_key(text))
