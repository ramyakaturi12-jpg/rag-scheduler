"""
ingest.py
---------
One-time (or repeatable) script to seed ChromaDB with sample schedule data.

Run:
    python ingest.py            # seed with today as day-0
    python ingest.py --reset    # wipe collection and re-seed
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

import vector_store as vs
from sample_schedule import generate_sample_events


def main(reset: bool = False, start_date: date | None = None) -> None:
    if start_date is None:
        start_date = date.today()

    print(f"[ingest] Generating sample events starting {start_date.isoformat()} …")
    events = generate_sample_events(start_date=start_date)
    print(f"[ingest] {len(events)} events generated.")

    if reset:
        print("[ingest] Resetting collection …")

    count = vs.ingest_events(events, reset=reset)
    total = vs.collection_count()
    print(f"[ingest] ✅ Upserted {count} events. Collection now has {total} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed ChromaDB with sample schedule data.")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate the collection before ingesting.")
    parser.add_argument("--date", default=None,
                        help="Override start date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    start = date.fromisoformat(args.date) if args.date else None
    main(reset=args.reset, start_date=start)
