"""
scripts/assign_phone.py — Assign a real phone number to a test member.

Usage:
    python scripts/assign_phone.py mem_test_001 +15869443943

DATABASE_URL is read from the environment or .env file.
"""

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

if len(sys.argv) != 3:
    print("Usage: python scripts/assign_phone.py <member_id> <phone_e164>")
    print("  e.g. python scripts/assign_phone.py mem_test_001 +15869443943")
    sys.exit(1)

member_id = sys.argv[1]
phone     = sys.argv[2]

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set.")
    print("Set it in a .env file or run: $env:DATABASE_URL = 'postgresql://...'")
    sys.exit(1)

url = DATABASE_URL.replace("+asyncpg", "").replace("+psycopg2", "")

try:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        "UPDATE members SET phone_e164 = NULL WHERE phone_e164 = %s AND id != %s",
        (phone, member_id)
    )
    cur.execute(
        "UPDATE members SET phone_e164 = %s WHERE id = %s RETURNING first_name, last_name",
        (phone, member_id)
    )
    row = cur.fetchone()
    if row:
        print(f"Done: {row[0]} {row[1]} ({member_id}) -> {phone}")
    else:
        print(f"ERROR: member '{member_id}' not found. Run seed_test_data.py first.")
        sys.exit(1)

    cur.close()
    conn.close()

except Exception as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
