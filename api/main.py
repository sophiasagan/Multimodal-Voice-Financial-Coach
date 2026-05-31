"""
api/main.py — FastAPI entry point for the CU Voice Financial Coach.

Routes
------
POST /voice/incoming   Twilio webhook for inbound calls.
                       Returns TwiML: plays the opening greeting, then opens
                       a bidirectional Media Streams WebSocket.

WS   /voice/stream     Real-time audio pipe (Twilio Media Streams v2).
                       Receives mulaw 8 kHz chunks, runs silence detection,
                       feeds each utterance through the full coach pipeline,
                       and streams mulaw audio back to the member's phone.

GET  /health           Liveness probe (load-balancer / k8s).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .coach import generate_response
from .context_builder import build_context
from .guardrails import check_guardrails
from .member_resolver import resolve_member
from .session_store import SessionStore
from .speaker import synthesize_speech, to_mulaw_8k
from .transcriber import transcribe

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (env vars → sensible dev defaults)
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL   = os.getenv("REDIS_URL",   "redis://localhost:6379/0")
PUBLIC_HOST = os.getenv("PUBLIC_HOST", "")     # e.g. abc123.ngrok.io
CU_NAME     = os.getenv("CU_NAME",     "your credit union")

OPENING_MESSAGE = (
    f"Hi, I'm the financial coach from {CU_NAME}. "
    "I can help you understand your account, answer questions about your "
    "finances, and talk through your options. "
    "What's on your mind?"
)

# Generated once at startup; served as a static file so Twilio can <Play> it.
OPENING_AUDIO_PATH = Path("static/opening.mp3")

# ── Silence-detection parameters ─────────────────────────────────────────────
# Twilio Media Streams sends 20 ms mulaw frames (160 bytes at 8 kHz).
# We treat 200 ms of consecutive sub-threshold audio as end-of-utterance.

FRAME_MS            = 20     # ms per Twilio audio frame
SILENCE_MS          = 200    # ms of silence → end of utterance
SILENCE_FRAMES_REQ  = SILENCE_MS // FRAME_MS   # = 10 frames

# G.711 µ-law 16-bit linear RMS value below which a frame is "silent".
# Calibrated for normal telephone handset background noise.
SILENCE_RMS_FLOOR   = 300


# ─────────────────────────────────────────────────────────────────────────────
# G.711 µ-law helpers  (pure-Python; no audioop dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mulaw_decode_table() -> list[int]:
    """
    Pre-compute the 256-entry G.711 µ-law → 16-bit signed linear PCM table.

    Algorithm (ITU-T G.711 §3.1):
      1. Invert all 8 bits of the compressed byte.
      2. Extract sign (bit 7), exponent (bits 6-4), mantissa (bits 3-0).
      3. Magnitude = ((mantissa << 3 | 0x04) << exponent) − bias
         where bias = 0x84 (132).
    """
    table: list[int] = []
    for i in range(256):
        u        = (~i) & 0xFF
        sign     = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        magnitude = (((mantissa << 3) | 0x04) << exponent) - 0x84
        table.append(-magnitude if sign else magnitude)
    return table


_MULAW_TABLE: list[int] = _build_mulaw_decode_table()


def _frame_rms(mulaw_bytes: bytes) -> float:
    """Return the RMS amplitude (16-bit linear scale) of a mulaw audio frame."""
    if not mulaw_bytes:
        return 0.0
    total = sum(_MULAW_TABLE[b] ** 2 for b in mulaw_bytes)
    return (total / len(mulaw_bytes)) ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup
    -------
    1. Generate the opening greeting audio via TTS (once; cached to disk).
    2. Open a shared async Redis connection pool.

    Shutdown
    --------
    Gracefully close the Redis pool.
    """
    # 1. Opening audio ──────────────────────────────────────────────────────
    OPENING_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not OPENING_AUDIO_PATH.exists():
        logger.info("Generating opening audio via TTS…")
        audio_bytes = await synthesize_speech(OPENING_MESSAGE)
        OPENING_AUDIO_PATH.write_bytes(audio_bytes)
        logger.info("Opening audio saved → %s", OPENING_AUDIO_PATH)
    else:
        logger.info("Opening audio already exists at %s — skipping TTS.", OPENING_AUDIO_PATH)

    # 2. Redis ──────────────────────────────────────────────────────────────
    redis_client = await aioredis.from_url(
        REDIS_URL, encoding="utf-8", decode_responses=True
    )
    app.state.redis         = redis_client
    app.state.session_store = SessionStore(redis_client)
    logger.info("Redis connected: %s", REDIS_URL)

    yield  # ← application serves requests here

    # Shutdown ──────────────────────────────────────────────────────────────
    await redis_client.aclose()
    logger.info("Redis connection closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CU Voice Financial Coach",
    description=(
        "Multimodal voice financial coach: "
        "Twilio → Whisper STT → Claude → OpenAI TTS → member's phone."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Serve static assets only when the directory exists (it may be empty on a
# fresh deploy before the opening audio is generated at first startup).
if Path("static").is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe — returns 200 when the server is up."""
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /voice/incoming  —  Twilio webhook for inbound calls
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/voice/incoming", response_class=Response)
async def voice_incoming(request: Request) -> Response:
    """
    Twilio calls this endpoint (HTTP POST) when a member dials the CU's number.

    Returns TwiML that:
      1. Plays the pre-generated opening greeting from /static/opening.mp3.
      2. Opens a bidirectional Media Streams WebSocket to /voice/stream.

    The caller's phone number is injected into the Stream as a custom parameter
    so the WebSocket handler can identify the member without an additional lookup.
    """
    form         = await request.form()
    call_sid     = form.get("CallSid", "unknown")
    caller_phone = form.get("From",    "unknown")
    logger.info("Inbound call  CallSid=%s  From=%s", call_sid, caller_phone)

    # Absolute URL for the opening audio (Twilio needs a publicly reachable URL)
    base_url  = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/static/opening.mp3"

    # WebSocket URL — prefer PUBLIC_HOST (ngrok / prod domain) over request host
    domain     = PUBLIC_HOST or request.headers.get("host", "localhost:8000")
    stream_url = f"wss://{domain}/voice/stream"

    # Build TwiML manually (avoids a hard twilio SDK dep at the import level)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        f'  <Play>{audio_url}</Play>\n'
        '  <Connect>\n'
        f'    <Stream url="{stream_url}">\n'
        f'      <Parameter name="caller_phone" value="{caller_phone}"/>\n'
        f'      <Parameter name="call_sid"     value="{call_sid}"/>\n'
        '    </Stream>\n'
        '  </Connect>\n'
        "</Response>\n"
    )

    return Response(content=twiml, media_type="application/xml")


# ─────────────────────────────────────────────────────────────────────────────
# WS /voice/stream  —  real-time bidirectional audio (Twilio Media Streams v2)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/voice/stream")
async def voice_stream(ws: WebSocket) -> None:
    """
    Bidirectional audio WebSocket endpoint for Twilio Media Streams.

    Twilio frame types (JSON over WebSocket text):
      connected  — initial handshake; contains protocol version.
      start      — stream metadata: callSid, accountSid, customParameters.
      media      — 20 ms mulaw 8 kHz audio encoded as base64 payload.
      stop       — call ended (member hung up or Twilio hung up).
      dtmf       — key-press digit (reserved for future menu navigation).

    To play audio back to the member we send `media` frames with base64-encoded
    mulaw 8 kHz audio, followed by a `mark` frame signalling end-of-playback.

    Pipeline per utterance (fired as an async task so the receive loop stays live):
      mulaw bytes → transcriber.transcribe()
                  → guardrails.check_guardrails()   [MUST run before Claude]
                  → member_resolver / context_builder
                  → coach.generate_response()        [≤ 80 words]
                  → speaker.synthesize_speech()      [MP3]
                  → speaker.to_mulaw_8k()            [MP3 → mulaw 8 kHz]
                  → _send_audio()                    [stream back to Twilio]
    """
    await ws.accept()

    # ── per-connection state ───────────────────────────────────────────────
    call_sid:     Optional[str] = None
    stream_sid:   Optional[str] = None
    caller_phone: Optional[str] = None

    audio_buffer:  bytearray = bytearray()
    silent_frames: int       = 0
    in_utterance:  bool      = False   # True once speech energy detected

    # Prevents concurrent pipeline runs from overlapping turns.
    # If a pipeline is already running when the next utterance ends,
    # the new utterance is dropped (preferred over queuing stale audio).
    processing_lock = asyncio.Lock()

    session_store: SessionStore = ws.app.state.session_store

    # ── inner helpers ──────────────────────────────────────────────────────

    async def _send_audio(mulaw_bytes: bytes) -> None:
        """
        Stream mulaw 8 kHz audio back to Twilio as 20 ms media frames.

        Twilio expects the same format it sends: mulaw, 8 kHz, mono.
        Frame size matches the 20 ms / 160-byte Twilio default to avoid
        jitter at the phone handset.
        """
        if not stream_sid:
            logger.warning("Cannot send audio — stream_sid not yet known.")
            return

        CHUNK = 160  # 20 ms @ 8 kHz mulaw
        for offset in range(0, len(mulaw_bytes), CHUNK):
            payload = base64.b64encode(mulaw_bytes[offset : offset + CHUNK]).decode()
            await ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": stream_sid,
                "media":     {"payload": payload},
            }))

        # Mark frame lets Twilio (and us) know playback is complete.
        await ws.send_text(json.dumps({
            "event":     "mark",
            "streamSid": stream_sid,
            "mark":      {"name": "response_end"},
        }))

    async def _handle_utterance(raw_mulaw: bytes) -> None:
        """
        Full coach pipeline for one complete utterance.
        Runs under processing_lock so turns never overlap.

        Steps
        -----
        1. Transcribe mulaw → text (Whisper).
        2. Guardrails: topic filter, crisis detection, escalation check.
           Crisis path: inject 988 hotline before any transfer.
        3. Resolve member by caller_phone; build account context.
        4. Generate Claude response (≤ 80 words hard limit).
        5. Synthesise speech (OpenAI TTS → MP3 → mulaw 8 kHz).
        6. Persist turn to Redis conversation history.
        7. Stream audio back to member.
        """
        async with processing_lock:
            try:
                # ── 1. Transcribe ────────────────────────────────────────
                logger.debug("Transcribing %d mulaw bytes…", len(raw_mulaw))
                transcript: str = await transcribe(raw_mulaw)
                if not transcript or not transcript.strip():
                    logger.debug("Empty transcript — skipping turn.")
                    return
                logger.info("[%s] Member: %r", call_sid, transcript)

                # ── 2. Guardrails ────────────────────────────────────────
                # MUST run before generate_response (CLAUDE.md constraint).
                guard = await check_guardrails(transcript, call_sid=call_sid)

                if guard.escalate:
                    # Crisis / out-of-scope / human-agent transfer
                    logger.info(
                        "[%s] Guardrail triggered: %s", call_sid, guard.reason
                    )
                    response_text = guard.escalation_message
                else:
                    # ── 3. Member lookup + account context ───────────────
                    member  = await resolve_member(caller_phone)
                    context = await build_context(member) if member else ""

                    # ── 4. Claude response (≤ 80 words) ─────────────────
                    history = await session_store.get_history(call_sid)
                    response_text = await generate_response(
                        transcript=transcript,
                        account_context=context,
                        history=history,
                        call_sid=call_sid,
                    )

                logger.info(
                    "[%s] Coach (%d words): %r",
                    call_sid,
                    len(response_text.split()),
                    response_text,
                )

                # ── 5. TTS → mulaw 8 kHz ────────────────────────────────
                mp3_bytes   = await synthesize_speech(response_text)
                mulaw_bytes = await to_mulaw_8k(mp3_bytes)

                # ── 6. Persist turn ──────────────────────────────────────
                await session_store.append_turn(
                    call_sid, role="user",      content=transcript
                )
                await session_store.append_turn(
                    call_sid, role="assistant", content=response_text
                )

                # ── 7. Stream audio back ─────────────────────────────────
                await _send_audio(mulaw_bytes)

            except Exception as exc:
                logger.exception("[%s] Pipeline error: %s", call_sid, exc)
                # Keep the call alive with a graceful fallback utterance.
                try:
                    fallback_mp3   = await synthesize_speech(
                        "I'm sorry, I didn't catch that. Could you say it again?"
                    )
                    fallback_mulaw = await to_mulaw_8k(fallback_mp3)
                    await _send_audio(fallback_mulaw)
                except Exception:
                    pass  # silence is better than a crash

    # ── main WebSocket receive loop ────────────────────────────────────────
    try:
        async for raw_message in ws.iter_text():
            msg   = json.loads(raw_message)
            event = msg.get("event")

            # ── connected ───────────────────────────────────────────────
            if event == "connected":
                logger.info(
                    "Twilio WebSocket connected  protocol=%s  version=%s",
                    msg.get("protocol"),
                    msg.get("version"),
                )

            # ── start ────────────────────────────────────────────────────
            elif event == "start":
                start_data   = msg.get("start", {})
                stream_sid   = msg.get("streamSid") or start_data.get("streamSid")
                call_sid     = start_data.get("callSid")
                custom       = start_data.get("customParameters", {})
                caller_phone = custom.get("caller_phone")
                media_fmt    = start_data.get("mediaFormat", {})

                logger.info(
                    "Stream started  CallSid=%s  StreamSid=%s  From=%s  "
                    "encoding=%s  sampleRate=%s",
                    call_sid, stream_sid, caller_phone,
                    media_fmt.get("encoding"),
                    media_fmt.get("sampleRate"),
                )

                # Initialise Redis session record
                await session_store.create_session(
                    call_sid=call_sid,
                    stream_sid=stream_sid,
                    caller_phone=caller_phone,
                    start_time=time.time(),
                )

            # ── media ────────────────────────────────────────────────────
            elif event == "media":
                media = msg.get("media", {})
                if media.get("track", "inbound") != "inbound":
                    continue  # ignore reflected outbound frames

                raw_bytes = base64.b64decode(media.get("payload", ""))
                rms       = _frame_rms(raw_bytes)

                if rms > SILENCE_RMS_FLOOR:
                    # Active speech — accumulate and reset silence counter
                    in_utterance  = True
                    silent_frames = 0
                    audio_buffer.extend(raw_bytes)
                else:
                    # Silent frame
                    if in_utterance:
                        silent_frames += 1
                        # Continue buffering through the pause so we don't
                        # clip trailing words; silence frames are kept so
                        # Whisper gets natural audio boundaries.
                        audio_buffer.extend(raw_bytes)

                        if silent_frames >= SILENCE_FRAMES_REQ:
                            # ─ end of utterance ─────────────────────────
                            utterance_audio = bytes(audio_buffer)
                            audio_buffer    = bytearray()   # reset
                            silent_frames   = 0
                            in_utterance    = False

                            logger.debug(
                                "[%s] Utterance complete  %d bytes",
                                call_sid, len(utterance_audio),
                            )

                            # Drop utterance if a pipeline is already running
                            # (prevents stale audio queueing mid-response).
                            if processing_lock.locked():
                                logger.debug(
                                    "[%s] Pipeline busy — dropping utterance.",
                                    call_sid,
                                )
                            else:
                                asyncio.create_task(
                                    _handle_utterance(utterance_audio)
                                )

            # ── stop  ────────────────────────────────────────────────────
            elif event == "stop":
                logger.info("Stream stopped  CallSid=%s", call_sid)
                break

            # ── dtmf ─────────────────────────────────────────────────────
            elif event == "dtmf":
                digit = msg.get("dtmf", {}).get("digit")
                logger.info("[%s] DTMF digit=%s", call_sid, digit)
                # Reserved: future DTMF-driven menu navigation

            # ── unknown ──────────────────────────────────────────────────
            else:
                logger.debug("Unknown Twilio event: %s", event)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected  CallSid=%s", call_sid)
    except Exception as exc:
        logger.exception("Fatal WebSocket error  CallSid=%s: %s", call_sid, exc)
    finally:
        # ── session teardown ─────────────────────────────────────────────────
        # Saves conversation summary and flags follow-up items.
        if call_sid:
            try:
                await session_store.close_session(call_sid)
                logger.info("Session closed  CallSid=%s", call_sid)
            except Exception as exc:
                logger.warning(
                    "Failed to close session %s: %s", call_sid, exc
                )
