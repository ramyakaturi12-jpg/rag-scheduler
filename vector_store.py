"""
vector_store.py
---------------
ChromaDB-backed schedule store.

Uses ChromaDB purely as a persistent document + metadata store.
Embeddings are disabled — all retrieval is done via:
  1. Exact metadata filters  (date, type)
  2. Keyword search in documents (for free-text queries)

This approach requires zero external embedding APIs and works on any free tier.
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME: str = "schedule_events"
DEFAULT_TOP_K: int = 8

# ── Dummy embedding function (stores zero vectors — we never query by vector) ──
class DummyEmbedder(EmbeddingFunction):
    """
    Returns a fixed 1-dim zero vector for every document.
    We never use vector similarity search — all queries are metadata/keyword based.
    This lets us use ChromaDB as a pure document store without any embedding API.
    """
    def __call__(self, input: list[str]) -> Embeddings:
        return [[0.0] for _ in input]


# ── Singleton client & collection ─────────────────────────────────────────────
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, initializing it on first call."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DummyEmbedder(),
        )
    return _collection


# ── Document builder ──────────────────────────────────────────────────────────
def _event_to_doc(event: dict) -> str:
    """Build a searchable text document from an event dict."""
    parts = [
        f"Event: {event.get('title', '')}",
        f"Date: {event.get('date', '')}",
        f"Time: {event.get('start_time', '')} to {event.get('end_time', '')}",
        f"Type: {event.get('type', '')}",
    ]
    if event.get("location"):
        parts.append(f"Location: {event['location']}")
    if event.get("attendees"):
        parts.append(f"Attendees: {event['attendees']}")
    if event.get("description"):
        parts.append(f"Description: {event['description']}")
    return ". ".join(parts)


def _event_to_metadata(event: dict) -> dict:
    """Flatten event fields into ChromaDB-compatible metadata (strings only)."""
    return {
        "title":       event.get("title", ""),
        "date":        event.get("date", ""),
        "start_time":  event.get("start_time", ""),
        "end_time":    event.get("end_time", ""),
        "type":        event.get("type", ""),
        "location":    event.get("location", ""),
        "attendees":   event.get("attendees", ""),
        "description": event.get("description", ""),
    }


# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest_events(events: list[dict], reset: bool = False) -> int:
    """Upsert a list of event dicts into ChromaDB."""
    global _collection

    if reset and _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None

    col = get_collection()
    col.upsert(
        ids=[e["id"] for e in events],
        documents=[_event_to_doc(e) for e in events],
        metadatas=[_event_to_metadata(e) for e in events],
    )
    return len(events)


# ── Retrieval ─────────────────────────────────────────────────────────────────
def query_events(query_text: str, top_k: int = DEFAULT_TOP_K,
                 where: dict | None = None) -> list[dict]:
    """
    Keyword search over all stored events.
    Scores each document by how many query words appear in its text.
    Falls back to returning all events if no keywords match.
    """
    col = get_collection()
    kwargs: dict[str, Any] = {"include": ["metadatas", "documents"]}
    if where:
        kwargs["where"] = where

    results = col.get(**kwargs)
    if not results["ids"]:
        return []

    query_words = set(query_text.lower().split())
    scored: list[tuple[int, str, dict]] = []

    for i, eid in enumerate(results["ids"]):
        doc  = (results["documents"][i] or "").lower()
        meta = results["metadatas"][i]
        score = sum(1 for w in query_words if w in doc)
        scored.append((score, eid, meta))

    # Sort by score desc, then date asc
    scored.sort(key=lambda x: (-x[0], x[2].get("date", ""), x[2].get("start_time", "")))

    events = []
    for score, eid, meta in scored[:top_k]:
        events.append({
            "id": eid,
            "title":        meta.get("title", ""),
            "date":         meta.get("date", ""),
            "start_time":   meta.get("start_time", ""),
            "end_time":     meta.get("end_time", ""),
            "type":         meta.get("type", ""),
            "location":     meta.get("location", ""),
            "attendees":    meta.get("attendees", ""),
            "description":  meta.get("description", ""),
            "relevance_score": score,
        })
    return events


def get_events_by_date(date_str: str) -> list[dict]:
    """Exact-match retrieval for a specific date (YYYY-MM-DD)."""
    col = get_collection()
    results = col.get(where={"date": date_str}, include=["metadatas"])

    events = []
    for i, eid in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        events.append({"id": eid, **meta})

    events.sort(key=lambda e: e.get("start_time", ""))
    return events


def get_events_by_date_range(start_date: str, end_date: str) -> list[dict]:
    """Retrieve all events between start_date and end_date inclusive."""
    col = get_collection()
    results = col.get(include=["metadatas"])

    events = []
    for i, eid in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        d = meta.get("date", "")
        if start_date <= d <= end_date:
            events.append({"id": eid, **meta})

    events.sort(key=lambda e: (e.get("date", ""), e.get("start_time", "")))
    return events


# ── CRUD ──────────────────────────────────────────────────────────────────────
def add_event(event: dict) -> str:
    """Add a new event. Generates an ID if not provided. Returns the event ID."""
    col = get_collection()

    if not event.get("id"):
        existing_ids = col.get()["ids"]
        idx = len(existing_ids) + 1
        event["id"] = f"evt_{idx:03d}"
        while event["id"] in existing_ids:
            idx += 1
            event["id"] = f"evt_{idx:03d}"

    col.upsert(
        ids=[event["id"]],
        documents=[_event_to_doc(event)],
        metadatas=[_event_to_metadata(event)],
    )
    return event["id"]


def update_event(event_id: str, updates: dict) -> bool:
    """Update fields of an existing event. Returns True if found and updated."""
    col = get_collection()
    result = col.get(ids=[event_id], include=["metadatas"])

    if not result["ids"]:
        return False

    existing_meta = result["metadatas"][0]
    existing_meta.update({k: v for k, v in updates.items() if k != "id"})

    col.upsert(
        ids=[event_id],
        documents=[_event_to_doc(existing_meta)],
        metadatas=[_event_to_metadata(existing_meta)],
    )
    return True


def delete_event(event_id: str) -> bool:
    """Delete an event by ID. Returns True if deleted."""
    col = get_collection()
    if not col.get(ids=[event_id])["ids"]:
        return False
    col.delete(ids=[event_id])
    return True


def find_event_by_title_and_date(title_fragment: str,
                                  date_str: str | None = None) -> list[dict]:
    """Find events whose title contains title_fragment on an optional date."""
    col = get_collection()
    kwargs: dict[str, Any] = {"include": ["metadatas"]}
    if date_str:
        kwargs["where"] = {"date": date_str}
    results = col.get(**kwargs)

    matches = []
    frag = title_fragment.lower()
    for i, eid in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        if frag in meta.get("title", "").lower():
            matches.append({"id": eid, **meta})
    return matches


def collection_count() -> int:
    """Return total number of events stored."""
    return get_collection().count()
