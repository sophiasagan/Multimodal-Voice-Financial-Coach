"""
scripts/seed_test_data.py — Insert test members into the database.

Run once (safe to re-run — uses ON CONFLICT DO NOTHING):

    python scripts/seed_test_data.py

DATABASE_URL is read from the environment or .env file.

After seeding, update the `phone_e164` column for any member you want to
reach from a REAL phone number:

    UPDATE members SET phone_e164 = '+15551234567'
    WHERE id = 'mem_test_001';

See TEST_DATA.md for the full test scenarios and suggested conversation flows.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set.")
    sys.exit(1)

url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
url = url.replace("postgresql+psycopg2://", "postgresql://")

# ─────────────────────────────────────────────────────────────────────────────
# Test members
# Update phone_e164 to your real test phone numbers before calling.
# ─────────────────────────────────────────────────────────────────────────────

MEMBERS = [
    {
        "id":            "mem_test_001",
        "member_number": "M-10001",
        "first_name":    "Sarah",
        "last_name":     "Chen",
        "phone_e164":    "+15550001001",   # ← replace with your test number
        "email":         "sarah.chen.test@example.com",
        "status":        "active",
        "ssn_last4_hash": None,
    },
    {
        "id":            "mem_test_002",
        "member_number": "M-10002",
        "first_name":    "Marcus",
        "last_name":     "Thompson",
        "phone_e164":    "+15550001002",
        "email":         "marcus.thompson.test@example.com",
        "status":        "active",
        "ssn_last4_hash": None,
    },
    {
        "id":            "mem_test_003",
        "member_number": "M-10003",
        "first_name":    "Linda",
        "last_name":     "Rodriguez",
        "phone_e164":    "+15550001003",
        "email":         "linda.rodriguez.test@example.com",
        "status":        "active",
        "ssn_last4_hash": None,
    },
    {
        "id":            "mem_test_004",
        "member_number": "M-10004",
        "first_name":    "David",
        "last_name":     "Kim",
        "phone_e164":    "+15550001004",
        "email":         "david.kim.test@example.com",
        "status":        "active",
        "ssn_last4_hash": None,
    },
    {
        "id":            "mem_test_005",
        "member_number": "M-10005",
        "first_name":    "Priya",
        "last_name":     "Patel",
        "phone_e164":    "+15550001005",
        "email":         "priya.patel.test@example.com",
        "status":        "active",
        "ssn_last4_hash": None,
    },
    {
        "id":            "mem_test_006",
        "member_number": "M-10006",
        "first_name":    "Robert",
        "last_name":     "Walsh",
        "phone_e164":    "+15550001006",
        "email":         "robert.walsh.test@example.com",
        "status":        "active",
        "ssn_last4_hash": None,
    },
]

INSERT_SQL = """
    INSERT INTO members
        (id, member_number, first_name, last_name, phone_e164, email, status, ssn_last4_hash)
    VALUES
        (%(id)s, %(member_number)s, %(first_name)s, %(last_name)s,
         %(phone_e164)s, %(email)s, %(status)s, %(ssn_last4_hash)s)
    ON CONFLICT (id) DO UPDATE SET
        first_name    = EXCLUDED.first_name,
        last_name     = EXCLUDED.last_name,
        email         = EXCLUDED.email,
        status        = EXCLUDED.status;
"""


def main() -> None:
    print(f"Connecting to database…")
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()

        for m in MEMBERS:
            cur.execute(INSERT_SQL, m)
            print(f"  ✓ {m['first_name']} {m['last_name']}  ({m['id']})  {m['phone_e164']}")

        cur.close()
        conn.close()
        print(f"\nDone. {len(MEMBERS)} test members upserted.")
        print("\nTo assign your real phone number to a test member:")
        print("  UPDATE members SET phone_e164 = '+1XXXXXXXXXX' WHERE id = 'mem_test_001';")
        print("\nSee TEST_DATA.md for full scenario guide.")

    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
