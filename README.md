# 🏦 CU Voice Financial Coach

> A real-time AI financial coach delivered over a phone call — no app required.  
> Members call their credit union's dedicated number and speak naturally with an AI that knows their accounts.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Twilio](https://img.shields.io/badge/Twilio-Media%20Streams-F22F46?logo=twilio&logoColor=white)](https://www.twilio.com/docs/voice/media-streams)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper%20STT-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/speech-to-text)
[![OpenAI TTS](https://img.shields.io/badge/OpenAI-TTS%20nova-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/text-to-speech)
[![Claude](https://img.shields.io/badge/Anthropic-Claude%20Sonnet%204.6-CC785C?logo=anthropic&logoColor=white)](https://docs.anthropic.com)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## Architecture

```
Member's phone
     │
     │  PSTN / Twilio Voice
     ▼
┌──────────────────────────────────────────────────────────────────┐
│  Twilio                                                          │
│  POST /voice/incoming  ──►  TwiML: <Stream url="wss://…">       │
│  WebSocket frames (mulaw 8kHz, base64 JSON)  ◄──►  FastAPI      │
└──────────────┬───────────────────────────────────────┬──────────┘
               │ audio frames                           │ mulaw audio
               ▼                                        ▲
┌─────────────────────────────────┐      ┌─────────────────────────┐
│  FastAPI  (api/main.py)         │      │  OpenAI TTS  (nova)     │
│                                 │      │  api/speaker.py         │
│  Silence detection (RMS)        │      │  MP3 → mulaw 8kHz       │
│  Audio accumulation             │      └──────────▲──────────────┘
│  WebSocket session mgmt         │                 │ text
└──────┬──────────────────────────┘      ┌──────────┴──────────────┐
       │ mulaw bytes                      │  Claude Sonnet 4.6      │
       ▼                                  │  api/coach.py           │
┌─────────────────────┐                  │  ≤ 80-word response     │
│  OpenAI Whisper     │                  │  Prompt caching on      │
│  api/transcriber.py │                  │  system prompt          │
│  → transcript text  │                  └──────────▲──────────────┘
└──────┬──────────────┘                             │ context
       │ transcript                    ┌────────────┴────────────────┐
       ▼                               │  api/context_builder.py     │
┌─────────────────────┐                │  P31 Financial Twin API     │
│  api/guardrails.py  │                │  Account data (parallel)    │
│  Crisis / hardship  │                └────────────▲────────────────┘
│  Escalation check   │                             │ member dict
└──────┬──────────────┘                ┌────────────┴────────────────┐
       │ (if clear)                    │  api/member_resolver.py     │
       └───────────────────────────────│  Phone → member lookup      │
                                       │  SQLAlchemy + Postgres      │
                                       └─────────────────────────────┘

Session layer
─────────────
api/session_store.py
  Redis (active call state, transcript, sentiment)
  ──► end-of-call: Claude summary extraction
  ──► Postgres: member_coaching_sessions table
  ──► CRM: D365 / P71 follow-up task (if needed)

Analytics
─────────
dashboard/app.py (Streamlit)
  Reads member_coaching_sessions
  Topics / questions feed → P52 RAG knowledge base updates
```

### Call flow — step by step

| Step | What happens |
|------|-------------|
| 1 | Member dials the CU's AI coach number |
| 2 | Twilio → `POST /voice/incoming` → FastAPI returns TwiML with opening audio + `<Stream>` WebSocket |
| 3 | WebSocket opens; Twilio streams mulaw 8 kHz audio frames as base64 JSON |
| 4 | FastAPI accumulates frames; 200 ms RMS silence = end of utterance |
| 5 | `transcriber.py` — mulaw → WAV → Whisper → transcript text |
| 6 | `guardrails.py` — checks for crisis / hardship / escalation triggers before Claude |
| 7 | `member_resolver.py` — identifies caller by E.164 phone number |
| 8 | `context_builder.py` — fetches account data + P31 Financial Twin insights (parallel) |
| 9 | `coach.py` — sends transcript + context + history to Claude; ≤ 80-word response |
| 10 | `speaker.py` — Claude text → OpenAI TTS (nova) → MP3 → mulaw → streamed to Twilio |
| 11 | Steps 4–10 repeat for each member utterance |
| 12 | On hangup: `session_store.end_session()` → Claude summary → DB + CRM task |

---

## Tech stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Telephony** | Twilio Voice + Media Streams | Industry-standard; WebSocket audio streaming at 8 kHz mulaw |
| **STT** | OpenAI Whisper (`whisper-1`) | Best accuracy on financial terminology at 8 kHz telephone quality |
| **LLM** | Anthropic Claude Sonnet 4.6 | Low latency, prompt caching on system+context block, 80-word outputs |
| **TTS** | OpenAI TTS (`tts-1`, voice `nova`) | Warm, natural, gender-neutral — appropriate for financial guidance |
| **API framework** | FastAPI + uvicorn | Async WebSocket support; clean dependency injection |
| **Session cache** | Redis 7 (via `redis.asyncio`) | Sub-millisecond read/write per audio frame; auto-TTL cleanup |
| **Database** | PostgreSQL 16 + SQLAlchemy 2 async | Member records + session summaries with JSONB analytics fields |
| **Member data** | P31 Financial Twin API | Churn score, health score, propensities, behavioural summary |
| **CRM** | Microsoft D365 or P71 REST | Follow-up task creation when action items identified |
| **Analytics** | Streamlit + Plotly | Self-hosted dashboard; no BI tool required |
| **Audio codec** | G.711 µ-law (pure Python, 3.13-safe) | `audioop` removed in Python 3.13; full decode table at import |

---

## Project structure

```
cu_voice_coach/
├── api/
│   ├── main.py               # FastAPI app: /voice/incoming + /voice/stream (WebSocket)
│   ├── transcriber.py        # mulaw 8kHz → WAV → Whisper → transcript
│   ├── member_resolver.py    # Phone number → member record (Postgres)
│   ├── context_builder.py    # P31 Financial Twin + account data → context string
│   ├── coach.py              # Claude response generation (≤ 80 words, prompt caching)
│   ├── speaker.py            # OpenAI TTS → MP3 → mulaw; common-phrase cache
│   ├── guardrails.py         # Crisis / hardship / escalation detection (fires before Claude)
│   ├── session_store.py      # Redis session + Claude summary + DB/CRM on call end
│   └── __init__.py
├── prompts/
│   ├── coach_system.txt      # Claude system prompt template ({cu_name}, {context})
│   └── escalation_phrases.txt # Substring-match escalation triggers
├── dashboard/
│   ├── app.py                # Streamlit analytics dashboard
│   └── requirements.txt
├── tests/
│   └── test_call_flow.py     # Simulate a full call without Twilio
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For mulaw transcoding (MP3 → 8kHz):
# macOS:  brew install ffmpeg
# Ubuntu: apt install ffmpeg
# Windows: winget install ffmpeg
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials:
```

```dotenv
# Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token

# OpenAI (Whisper + TTS)
OPENAI_API_KEY=sk-...

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# P31 Financial Twin
P31_API_BASE_URL=https://api.p31financial.com/v1
P31_API_KEY=your_p31_key

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/cu_coach

# Redis
REDIS_URL=redis://localhost:6379/0

# CU identity
CU_NAME=Hometown Credit Union

# Optional
CRM_PROVIDER=none               # d365 | p71 | none
SSN_VERIFICATION=false          # true for high-security CUs
PREWARM_TTS=true                # pre-synthesise greeting phrases at startup
STORE_RAW_TRANSCRIPT=true
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Expose to Twilio (development)

```bash
ngrok http 8000
# Copy the https URL → Twilio Console → Voice → Webhook URL:
# https://xxxx.ngrok.io/voice/incoming   (HTTP POST)
```

### 5. Run the analytics dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

## What members can ask

The coach handles **any conversational financial question** — not a menu of fixed intents.  
Example topics from real credit union deployments:

**Account & balance questions**
- *"What's my current checking balance?"*
- *"Did my direct deposit come in yet?"*
- *"When does my CD mature and what's the rate?"*

**Loans & payments**
- *"When is my next loan payment due, and how much?"*
- *"Can I skip a payment this month?"*
- *"What would my payment be if I refinanced at today's rate?"*
- *"How do I pay off my auto loan faster?"*

**Savings & financial health**
- *"Am I saving enough for an emergency fund?"*
- *"What's the difference between your savings account and a money market?"*
- *"My financial health score is 62 — what can I do to improve it?"*

**Products & rates**
- *"What CD rates are you offering right now?"*
- *"Do you have a high-yield savings account?"*
- *"I'm thinking about buying a home — where do I start?"*

**Financial hardship** *(escalates to counseling)*
- *"I lost my job and I'm worried about my loan payment."*
- *"Is there a skip-a-payment program I can use?"*
- *"What happens if I can't make my mortgage payment?"*

**General financial guidance**
- *"How much should I have in my emergency fund?"*
- *"What's the 50/30/20 budgeting rule?"*
- *"Should I pay down debt or build my savings first?"*
- *"I'm 28 — when should I start thinking about retirement?"*

> **Hard limits by design:** The coach never gives specific investment advice, never quotes securities, and never accesses accounts without phone-number verification. High-security CUs can require last-4 SSN verbal confirmation (see `member_resolver.py`).

---

## Guardrails

All member utterances are checked **before** reaching Claude:

| Category | Examples | Action |
|----------|----------|--------|
| `crisis` | "I can't go on", "hurt myself" | Provide **988** Lifeline + transfer to human |
| `financial_hardship` | "losing my home", "can't pay" | Transfer to financial counseling team |
| `complaint` | "speak to a manager", "file a complaint" | Transfer to member services |
| `investment_advice` | "should I buy this stock", "crypto" | Refer to licensed financial advisor |
| `escalation` | "transfer me", "talk to a person" | Connect to live agent |

---

## Analytics dashboard

`streamlit run dashboard/app.py`

| Panel | What it shows |
|-------|--------------|
| KPI strip | Calls, avg duration, escalation %, completion %, follow-up % with period-over-period deltas |
| Sentiment gauge | Donut of positive / neutral / concerned / distressed + inferred satisfaction score |
| Daily volume | Bar chart of call count by day |
| Topics frequency | Top-20 discussed topics (extracted by Claude at call end) |
| Escalation breakdown | By type and over time |
| **Most common questions** | Frequency-filtered table with CSV export → feeds P52 RAG knowledge base review |
| Follow-up actions | Pending CRM tasks with owner and sentiment context |

---

## Security & compliance notes

- **Phone number privacy** — only the last-4 digits appear in any log line
- **SSN digits** — never logged; only the boolean match outcome is recorded
- **No balances in logs** — member financial data flows only through the Claude context string and is never written to disk by the API layer
- **Guardrails fire before Claude** — harmful or misdirected responses cannot reach the member
- **Generic mode fallback** — if member lookup fails or DB is down, the call continues with general guidance only (no PII, no account data)
- **Raw transcript retention** — controlled by `STORE_RAW_TRANSCRIPT` env var; set to `false` to store only the Claude-extracted summary

---

## License

Internal / proprietary. Not for public distribution.
# Multimodal-Voice-Financial-Coach
