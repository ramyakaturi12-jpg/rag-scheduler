"""
tools.py
--------
LangChain-compatible tools for the Schedule Assistant agent.

Tools:
  1. get_schedule   – Retrieve schedule info by date, date range, or free-text query.
  2. update_schedule – Add, update, or delete schedule entries.
"""

from __future__ import annotations

import json
from datetime import date, timedelta, datetime
from typing import Literal

from langchain_core.tools import tool

import vector_store as vs

# ── Helpers ───────────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def _tomorrow() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def _day_name_to_date(day_name: str) -> str | None:
    """
    Convert a day name like 'Friday' or 'next Monday' to YYYY-MM-DD.
    Looks forward up to 14 days from today.
    """
    day_name = day_name.strip().lower()
    days = ["monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"]
    prefix = ""
    if day_name.startswith("next "):
        prefix = "next"
        day_name = day_name[5:].strip()
    if day_name not in days:
        return None

    target_weekday = days.index(day_name)
    today = date.today()
    delta = (target_weekday - today.weekday()) % 7
    if delta == 0 and prefix != "next":
        delta = 0   # today
    elif delta == 0 and prefix == "next":
        delta = 7
    elif delta == 0:
        delta = 7
    return (today + timedelta(days=delta)).isoformat()


def _format_events(events: list[dict], context: str = "") -> str:
    """Format a list of events into a readable string."""
    if not events:
        return f"No events found{' for ' + context if context else ''}."

    lines = [f"Found {len(events)} event(s){' for ' + context if context else ''}:\n"]
    for e in events:
        lines.append(
            f"  [{e.get('id', '?')}] {e['title']}\n"
            f"      Date     : {e['date']}\n"
            f"      Time     : {e['start_time']} – {e['end_time']}\n"
            f"      Type     : {e['type']}\n"
            f"      Location : {e.get('location') or 'N/A'}\n"
            f"      Attendees: {e.get('attendees') or 'N/A'}\n"
            f"      Notes    : {e.get('description') or ''}\n"
        )
    return "\n".join(lines)


def _parse_natural_date(date_str: str) -> str | None:
    """
    Try to parse various date formats and keywords into YYYY-MM-DD.
    Handles: 'today', 'tomorrow', day names, ISO dates, 'Aug 15', 'August 15', etc.
    """
    s = date_str.strip().lower()
    if s == "today":
        return _today()
    if s == "tomorrow":
        return _tomorrow()
    if s == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()

    # Day name
    day_result = _day_name_to_date(s)
    if day_result:
        return day_result

    # Try common formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
                "%B %d", "%b %d", "%B %d %Y", "%b %d %Y",
                "%d %B %Y", "%d %b %Y"):
        try:
            parsed = datetime.strptime(date_str.strip(), fmt)
            # For formats without year, use current year
            if parsed.year == 1900:
                parsed = parsed.replace(year=date.today().year)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


# ── Tool 1: get_schedule ──────────────────────────────────────────────────────

@tool
def get_schedule(query: str) -> str:
    """
    Retrieve schedule information based on a natural-language query.

    This tool understands queries like:
      - "What do I have tomorrow?"
      - "Show me my schedule for Friday"
      - "Am I free on August 20 afternoon?"
      - "What meetings do I have this week?"
      - "Do I have any workshops scheduled?"
      - "What's on my calendar for August 15?"
      - A specific date in any common format.

    Args:
        query: Natural-language question or date/time reference about the schedule.

    Returns:
        A formatted string listing relevant schedule events.
    """
    q = query.strip()

    # ── 1. Check for "today" / "tomorrow" keywords ───────────────────────────
    q_lower = q.lower()

    if "today" in q_lower:
        events = vs.get_events_by_date(_today())
        return _format_events(events, "today")

    if "tomorrow" in q_lower:
        events = vs.get_events_by_date(_tomorrow())
        return _format_events(events, "tomorrow")

    # ── 2. Check for "this week" ──────────────────────────────────────────────
    if "this week" in q_lower:
        today = date.today()
        week_start = (today - timedelta(days=today.weekday())).isoformat()
        week_end = (today + timedelta(days=6 - today.weekday())).isoformat()
        events = vs.get_events_by_date_range(week_start, week_end)
        return _format_events(events, "this week")

    if "next week" in q_lower:
        today = date.today()
        next_mon = today + timedelta(days=7 - today.weekday())
        next_sun = next_mon + timedelta(days=6)
        events = vs.get_events_by_date_range(next_mon.isoformat(), next_sun.isoformat())
        return _format_events(events, "next week")

    # ── 3. Try to extract a specific date ─────────────────────────────────────
    # Look for day names in the query
    day_names = ["monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday"]
    found_day = next((d for d in day_names if d in q_lower), None)
    if found_day:
        resolved = _day_name_to_date(found_day)
        if resolved:
            events = vs.get_events_by_date(resolved)
            return _format_events(events, f"{found_day.capitalize()} ({resolved})")

    # Try parsing the whole query as a date
    resolved_date = _parse_natural_date(q)
    if resolved_date:
        events = vs.get_events_by_date(resolved_date)
        return _format_events(events, resolved_date)

    # ── 4. Semantic / free-text search ────────────────────────────────────────
    events = vs.query_events(q, top_k=8)
    return _format_events(events, f'query "{q}"')


# ── Tool 2: update_schedule ───────────────────────────────────────────────────

@tool
def update_schedule(action: str, event_data: str) -> str:
    """
    Add, update, or remove a schedule entry.

    Args:
        action:     One of "add", "update", or "delete".
        event_data: A JSON string describing the event. Fields:
                    - For "add":
                        title       (required) – event name
                        date        (required) – YYYY-MM-DD or natural date like "August 15"
                        start_time  (required) – HH:MM  e.g. "15:00"
                        end_time    (optional) – HH:MM; defaults to start_time + 1 hour
                        type        (optional) – meeting|workshop|task|appointment
                        location    (optional)
                        description (optional)
                        attendees   (optional)
                    - For "update":
                        id OR (title + date) to identify the event (required)
                        Any fields you want to change.
                    - For "delete":
                        id OR (title + date) to identify the event (required)

    Returns:
        Confirmation message or error description.

    Examples:
        action="add",    event_data='{"title":"Team Meeting","date":"2026-08-20","start_time":"15:00","type":"meeting"}'
        action="update", event_data='{"title":"Team Meeting","date":"2026-08-20","start_time":"16:00"}'
        action="delete", event_data='{"title":"Team Meeting","date":"2026-08-20"}'
    """
    action = action.strip().lower()

    # Parse the event_data JSON
    try:
        data: dict = json.loads(event_data)
    except json.JSONDecodeError as exc:
        return f"Error: event_data is not valid JSON. {exc}"

    # ── ADD ───────────────────────────────────────────────────────────────────
    if action == "add":
        if not data.get("title"):
            return "Error: 'title' is required to add an event."
        if not data.get("date"):
            return "Error: 'date' is required to add an event."
        if not data.get("start_time"):
            return "Error: 'start_time' is required to add an event."

        # Normalise date
        norm_date = _parse_natural_date(data["date"])
        if not norm_date:
            return f"Error: Could not parse date '{data['date']}'. Use YYYY-MM-DD or a natural date."
        data["date"] = norm_date

        # Default end_time = start_time + 1 hour
        if not data.get("end_time"):
            try:
                st = datetime.strptime(data["start_time"], "%H:%M")
                et = (st + timedelta(hours=1)).strftime("%H:%M")
                data["end_time"] = et
            except ValueError:
                data["end_time"] = data["start_time"]

        # Default type
        data.setdefault("type", "meeting")

        event_id = vs.add_event(data)
        return (
            f"✅ Event added successfully!\n"
            f"   ID       : {event_id}\n"
            f"   Title    : {data['title']}\n"
            f"   Date     : {data['date']}\n"
            f"   Time     : {data['start_time']} – {data['end_time']}\n"
            f"   Type     : {data['type']}\n"
            f"   Location : {data.get('location') or 'N/A'}"
        )

    # ── UPDATE ────────────────────────────────────────────────────────────────
    elif action == "update":
        event_id = data.get("id")

        if not event_id:
            # Locate by title + date
            title = data.get("title")
            raw_date = data.get("date")
            if not title:
                return "Error: Provide 'id' or 'title' (and optionally 'date') to identify the event."
            norm_date = _parse_natural_date(raw_date) if raw_date else None
            candidates = vs.find_event_by_title_and_date(title, norm_date)
            if not candidates:
                return f"Error: No event found matching title '{title}'" + (f" on {norm_date}" if norm_date else "") + "."
            if len(candidates) > 1:
                ids = ", ".join(c["id"] for c in candidates)
                return (
                    f"Multiple events match '{title}'. Please specify the event ID. Matches: {ids}"
                )
            event_id = candidates[0]["id"]

        # Normalise date in updates if present
        if "date" in data:
            norm_date = _parse_natural_date(data["date"])
            if norm_date:
                data["date"] = norm_date

        updates = {k: v for k, v in data.items() if k not in ("id",)}
        success = vs.update_event(event_id, updates)
        if success:
            updated_fields = ", ".join(f"{k}={v}" for k, v in updates.items())
            return f"✅ Event {event_id} updated. Changes: {updated_fields}"
        else:
            return f"Error: Event '{event_id}' not found."

    # ── DELETE ────────────────────────────────────────────────────────────────
    elif action == "delete":
        event_id = data.get("id")

        if not event_id:
            title = data.get("title")
            raw_date = data.get("date")
            if not title:
                return "Error: Provide 'id' or 'title' (and optionally 'date') to identify the event."
            norm_date = _parse_natural_date(raw_date) if raw_date else None
            candidates = vs.find_event_by_title_and_date(title, norm_date)
            if not candidates:
                return f"Error: No event found matching title '{title}'" + (f" on {norm_date}" if norm_date else "") + "."
            if len(candidates) > 1:
                ids = ", ".join(c["id"] for c in candidates)
                return (
                    f"Multiple events match '{title}'. Please specify the event ID. Matches: {ids}"
                )
            event_id = candidates[0]["id"]

        success = vs.delete_event(event_id)
        if success:
            return f"✅ Event {event_id} deleted successfully."
        else:
            return f"Error: Event '{event_id}' not found."

    else:
        return f"Error: Unknown action '{action}'. Use 'add', 'update', or 'delete'."
