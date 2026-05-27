"""
api/transcriber.py — Whisper transcription for the CU Voice Financial Coach.

Converts raw Twilio mulaw 8 kHz audio bytes to text.

Pipeline
--------
1. mulaw 8-bit → 16-bit linear PCM
     • array + pre-built decode table (primary; works on Python 3.13+)
     • audioop.ulaw2lin             (fast-path fallback on Python < 3.13)
2. PCM → in-memory WAV              (stdlib wave; no temp files on disk)
3. WAV → text                       (OpenAI Whisper whisper-1, language="en")
4. Filter hallucinations / silence  (return None → caller skips the turn)

Notes
-----
• Twilio sends mulaw at 8 000 Hz.  Whisper was trained on 16 kHz but handles
  8 kHz well for clear telephone speech.  If accuracy degrades on a specific
  deployment, upsample PCM to 16 kHz before wrapping in WAV (resampy / scipy).
• audioop was deprecated in Python 3.11 and removed in 3.13.  The pure-Python
  path is the primary implementation; audioop is opportunistically used only
  when available for a small speed-up on older runtimes.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import time
import wave
from array import array
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 8_000   # Hz  — Twilio mulaw sample rate
SAMPLE_WIDTH = 2       # bytes — 16-bit signed linear PCM after decode

# Reject frames shorter than this before calling the Whisper API.
# 300 ms ≈ the shortest recognisable word on a phone line.
MIN_AUDIO_MS    = 300
MIN_AUDIO_BYTES = (MIN_AUDIO_MS * SAMPLE_RATE) // 1_000   # = 2 400

# Whisper sometimes emits these strings for silence or line noise at 8 kHz.
# Matching transcripts are discarded rather than sent to the coach.
_HALLUCINATION_RE = re.compile(
    r"^\s*(?:"
    r"thank\s+you\.?"        # "Thank you."   — very common for silence
    r"|you\.?"               # "you."
    r"|\.+"                  # "..." or "."
    r"|\s+"                  # whitespace only
    r")\s*$",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# G.711 µ-law → 16-bit linear PCM
# ─────────────────────────────────────────────────────────────────────────────

def _build_mulaw_decode_table() -> list[int]:
    """
    Pre-compute the 256-entry G.711 µ-law → 16-bit signed linear PCM table.

    Algorithm (ITU-T G.711 §3.1)
    ----------------------------
    1. Invert all 8 bits of the compressed byte.
    2. Extract sign (bit 7), exponent (bits 6-4), mantissa (bits 3-0).
    3. magnitude = ((mantissa << 3 | 0x04) << exponent) − 0x84
    4. Apply sign.

    Built once at import time; reused for every frame.
    """
    table: list[int] = []
    for i in range(256):
        u         = (~i) & 0xFF
        sign      = u & 0x80
        exponent  = (u >> 4) & 0x07
        mantissa  = u & 0x0F
        magnitude = (((mantissa << 3) | 0x04) << exponent) - 0x84
        table.append(-magnitude if sign else magnitude)
    return table


# Always build the table — used by both the pure-Python path and any
# module-level RMS callers that share this table via import.
_MULAW_TABLE: list[int] = _build_mulaw_decode_table()


# ── choose the fastest available PCM decoder ─────────────────────────────────

try:
    import audioop as _audioop  # type: ignore[import]  # removed in Py 3.13

    def _mulaw_to_pcm(mulaw_bytes: bytes) -> bytes:
        """audioop path (Python < 3.13): single C call, fastest."""
        return _audioop.ulaw2lin(mulaw_bytes, SAMPLE_WIDTH)

    logger.debug("transcriber: using audioop for µ-law decode")

except ImportError:

    def _mulaw_to_pcm(mulaw_bytes: bytes) -> bytes:   # type: ignore[misc]
        """
        Pure-Python path (Python 3.13+): array-based for reasonable speed.

        array("h") packs 16-bit signed integers; byteswap ensures little-endian
        WAV on big-endian hosts (rare, but correct).
        """
        pcm = array("h", (_MULAW_TABLE[b] for b in mulaw_bytes))
        if sys.byteorder == "big":
            pcm.byteswap()
        return pcm.tobytes()

    logger.debug("transcriber: audioop unavailable — using pure-Python µ-law decode")


# ─────────────────────────────────────────────────────────────────────────────
# mulaw → WAV (in-memory)
# ─────────────────────────────────────────────────────────────────────────────

def _mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    """
    Decode mulaw audio and wrap in a standard WAV container.

    Parameters
    ----------
    mulaw_bytes : raw G.711 µ-law bytes at 8 000 Hz mono

    Returns
    -------
    bytes : complete WAV file (RIFF header + PCM payload) in memory.
            No files are written to disk.
    """
    pcm_bytes  = _mulaw_to_pcm(mulaw_bytes)
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)            # mono
        wf.setsampwidth(SAMPLE_WIDTH) # 16-bit
        wf.setframerate(SAMPLE_RATE)  # 8 000 Hz
        wf.writeframes(pcm_bytes)

    return wav_buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client (lazy singleton — instantiated on first transcribe() call)
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
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def transcribe(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe raw mulaw 8 kHz audio bytes to English text via Whisper.

    Returns the transcript string on success, or None if the audio should be
    skipped (too short, silent, line noise, or a known Whisper hallucination).
    Callers should treat None as "no utterance detected — wait for next".

    Parameters
    ----------
    audio_bytes : bytes
        Raw G.711 µ-law encoded audio at 8 000 Hz, as accumulated by the
        silence-detection loop in main.py's voice_stream handler.

    Returns
    -------
    str  — cleaned transcript text on success
    None — on silence, noise, min-duration rejection, or API error

    Logs
    ----
    INFO  — utterance duration, word count, Whisper latency, full transcript
    DEBUG — rejection reason when returning None
    ERROR — Whisper API exceptions (call continues; pipeline skips the turn)
    """
    # ── 1. Pre-flight: too short to be real speech ────────────────────────────
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        logger.debug(
            "Audio too short — skipping Whisper call  "
            "bytes=%d  min=%d  duration_ms=%.0f",
            len(audio_bytes),
            MIN_AUDIO_BYTES,
            len(audio_bytes) / SAMPLE_RATE * 1_000,
        )
        return None

    duration_s = len(audio_bytes) / SAMPLE_RATE

    # ── 2. mulaw → WAV ────────────────────────────────────────────────────────
    try:
        wav_bytes = _mulaw_to_wav(audio_bytes)
    except Exception as exc:
        logger.error("mulaw → WAV conversion failed: %s", exc)
        return None

    # ── 3. Whisper API call ───────────────────────────────────────────────────
    # Pass as (filename, bytes, content-type) tuple so the SDK infers WAV
    # without needing a BytesIO object with a .name attribute.
    t0 = time.perf_counter()
    try:
        response = await _get_client().audio.transcriptions.create(
            model="whisper-1",
            file=("utterance.wav", wav_bytes, "audio/wav"),
            language="en",
        )
    except Exception as exc:
        logger.error(
            "Whisper API error  duration=%.2fs: %s", duration_s, exc
        )
        return None

    latency_ms = (time.perf_counter() - t0) * 1_000
    transcript  = response.text.strip()

    # ── 4. Filter empty / hallucinated output ─────────────────────────────────
    if not transcript:
        logger.debug(
            "Whisper returned empty transcript  "
            "duration=%.2fs  latency=%.0fms",
            duration_s, latency_ms,
        )
        return None

    if _HALLUCINATION_RE.match(transcript):
        logger.debug(
            "Whisper hallucination filtered  "
            "text=%r  duration=%.2fs  latency=%.0fms",
            transcript, duration_s, latency_ms,
        )
        return None

    # ── 5. Log success metrics ────────────────────────────────────────────────
    word_count = len(transcript.split())
    logger.info(
        "Whisper  duration=%.2fs  words=%d  latency=%.0fms  text=%r",
        duration_s, word_count, latency_ms, transcript,
    )

    return transcript
