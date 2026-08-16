"""
vector_store.py
---------------
ChromaDB setup and RAG pipeline.

Responsibilities:
  - Initialize / load a persistent ChromaDB collection.
  - Ingest schedule events (create document text + metadata).
  - Semantic search: retrieve the k most relevant events for a query.
  - CRUD helpers used by the agent tools (add / update / delete).
"""

from __future__ import annotations

import os
import json
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME: str = "schedule_events"
EMBED_MODEL: str = "gemini-embedding-exp-03-07"  # Google free embedding model (new SDK)
DEFAULT_TOP_K: int = 5

# ── Singleton client & collection ─────────────────────────────────────────────
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


class GeminiEmbedder(EmbeddingFunction):
    """Embedding function using the new google-genai SDK (free tier)."""

    def __init__(self):
        self._genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def __call__(self, input: list[str]) -> Embeddings:
        result = []
        for text in input:
            response = self._genai_client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                ),
            )
            result.append(response.embeddings[0].values)
        return result


def _embed_fn() -> GeminiEmbedder:
    """Return a GeminiEmbedder instance."""
    return GeminiEmbedder()


def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, initializing it on first call."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Document text builder ──────────────────────────────────────────────────────
def _event_to_doc(event: dict) -> str:
    """
    Convert an event dict to a rich natural-language document.
    This text is what gets embedded and searched semantically.
    """
    parts = [
        f"Event: {event['title']}",
        f"Date: {event['date']}",
        f"Time: {event['start_time']} to {event['end_time']}",
        f"Type: {event['type']}",
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
        "title": event.get("title", ""),
        "date": event.get("date", ""),
        "start_time": event.get("start_time", ""),
        "end_time": event.get("end_time", ""),
        "type": event.get("type", ""),
        "location": event.get("location", ""),
        "attendees": event.get("attendees", ""),
        "description": event.get("description", ""),
    }


# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest_events(events: list[dict], reset: bool = False) -> int:
    """
    Upsert a list of event dicts into ChromaDB.

    Args:
        events: List of event dicts (from sample_schedule.py or elsewhere).
        reset:  If True, drop and recreate the collection first.

    Returns:
        Number of events ingested.
    """
    global _collection

    if reset and _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None   # force re-init

    col = get_collection()

    ids = [e["id"] for e in events]
    documents = [_event_to_doc(e) for e in events]
    metadatas = [_event_to_metadata(e) for e in events]

    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(events)


# ── Retrieval (RAG) ───────────────────────────────────────────────────────────
def query_events(query_text: str, top_k: int = DEFAULT_TOP_K,
                 where: dict | None = None) -> list[dict]:
    """
    Semantic search over the schedule collection.

    Args:
        query_text: Natural-language query (e.g. "meetings on Friday afternoon").
        top_k:      Number of results to return.
        where:      Optional ChromaDB metadata filter, e.g. {"date": "2026-08-17"}.

    Returns:
        List of event dicts (metadata + id + distance).
    """
    col = get_collection()

    kwargs: dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": min(top_k, col.count() or 1),
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    events: list[dict] = []
    if not results["ids"] or not results["ids"][0]:
        return events

    for i, eid in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        events.append(
            {
                "id": eid,
                "title": meta.get("title", ""),
                "date": meta.get("date", ""),
                "start_time": meta.get("start_time", ""),
                "end_time": meta.get("end_time", ""),
                "type": meta.get("type", ""),
                "location": meta.get("location", ""),
                "attendees": meta.get("attendees", ""),
                "description": meta.get("description", ""),
                "relevance_score": round(1 - results["distances"][0][i], 4),
            }
        )
    return events


def get_events_by_date(date_str: str) -> list[dict]:
    """Exact-match retrieval for a specific date (YYYY-MM-DD)."""
    col = get_collection()
    results = col.get(where={"date": date_str}, include=["metadatas"])

    events: list[dict] = []
    for i, eid in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        events.append({"id": eid, **meta})

    # Sort by start_time
    events.sort(key=lambda e: e.get("start_time", ""))
    return events


def get_events_by_date_range(start_date: str, end_date: str) -> list[dict]:
    """
    Retrieve events between start_date and end_date (inclusive, YYYY-MM-DD).
    ChromaDB doesn't support range queries natively, so we fetch all and filter.
    """
    col = get_collection()
    results = col.get(include=["metadatas"])

    events: list[dict] = []
    for i, eid in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        d = meta.get("date", "")
        if start_date <= d <= end_date:
            events.append({"id": eid, **meta})

    events.sort(key=lambda e: (e.get("date", ""), e.get("start_time", "")))
    return events


# ── CRUD ──────────────────────────────────────────────────────────────────────
def add_event(event: dict) -> str:
    """
    Add a new event. Generates an ID if not provided.
    Returns the event ID.
    """
    col = get_collection()

    if not event.get("id"):
        # Generate a simple unique ID
        existing = col.get()
        existing_ids = existing["ids"]
        idx = len(existing_ids) + 1
        event["id"] = f"evt_{idx:03d}"
        # Ensure uniqueness
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
    """
    Update fields of an existing event.
    Returns True if the event was found and updated, False otherwise.
    """
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
    """
    Delete an event by ID.
    Returns True if deleted, False if not found.
    """
    col = get_collection()
    result = col.get(ids=[event_id])
    if not result["ids"]:
        return False
    col.delete(ids=[event_id])
    return True


def find_event_by_title_and_date(title_fragment: str,
                                  date_str: str | None = None) -> list[dict]:
    """
    Fuzzy find events whose title contains title_fragment on an optional date.
    Used by the update tool to locate events without knowing their ID.
    """
    col = get_collection()
    where = {"date": date_str} if date_str else None
    kwargs: dict[str, Any] = {"include": ["metadatas"]}
    if where:
        kwargs["where"] = where
    results = col.get(**kwargs)

    matches: list[dict] = []
    fragment_lower = title_fragment.lower()
    for i, eid in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        if fragment_lower in meta.get("title", "").lower():
            matches.append({"id": eid, **meta})
    return matches


def collection_count() -> int:
    """Return total number of events stored."""
    return get_collection().count()
