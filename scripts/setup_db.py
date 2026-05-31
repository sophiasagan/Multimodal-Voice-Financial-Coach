"""
scripts/setup_db.py — Create all tables in the target database.

Run once against a fresh database (local or Railway):

    python scripts/setup_db.py

DATABASE_URL is read from the environment or .env file.
Uses psycopg2 directly (no ORM) so it works without the full async stack.
"""

import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set.")
    sys.exit(1)

# psycopg2 doesn't accept the +asyncpg or +psycopg2 driver prefix
url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
url = url.replace("postgresql+psycopg2://", "postgresql://")

DDL = """
-- ── Members ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS members (
    id                  VARCHAR     PRIMARY KEY,
    member_number       VARCHAR,
    first_name          VARCHAR,
    last_name           VARCHAR,
    phone_e164          VARCHAR     UNIQUE,
    email               VARCHAR,
    status              VARCHAR     NOT NULL DEFAULT 'active',
    ssn_last4_hash      VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_members_phone  ON members (phone_e164);
CREATE INDEX IF NOT EXISTS idx_members_status ON members (status);

-- ── Coaching sessions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS member_coaching_sessions (
    id                      VARCHAR     PRIMARY KEY,
    call_sid                VARCHAR     UNIQUE NOT NULL,
    member_id               VARCHAR     REFERENCES members (id),
    started_at              TIMESTAMP WITH TIME ZONE,
    ended_at                TIMESTAMP WITH TIME ZONE,
    duration_s              INTEGER,
    topics_covered          JSONB,
    member_questions        JSONB,
    information_provided    JSONB,
    action_items            JSONB,
    member_sentiment        VARCHAR,
    follow_up_required      BOOLEAN     DEFAULT FALSE,
    follow_up_description   TEXT,
    escalated               BOOLEAN     DEFAULT FALSE,
    escalation_type         VARCHAR,
    raw_transcript          JSONB
);

CREATE INDEX IF NOT EXISTS idx_sessions_member_id  ON member_coaching_sessions (member_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON member_coaching_sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_sentiment  ON member_coaching_sessions (member_sentiment);
"""

def main() -> None:
    print(f"Connecting to database…")
    try:
        conn = psycopg2.connect(url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()
        conn.close()
        print("Done. Tables created (or already existed):")
        print("  ✓ members")
        print("  ✓ member_coaching_sessions")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
