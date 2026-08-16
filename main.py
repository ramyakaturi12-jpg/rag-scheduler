"""
main.py
-------
FastAPI application for the Agentic RAG Schedule Assistant.

Endpoints:
  POST /chat          – Send a message to the agent (supports conversation history).
  POST /chat/reset    – Clear conversation history (server-side session by session_id).
  GET  /schedule      – Quick REST endpoint to fetch today's schedule.
  GET  /health        – Health check.
  GET  /              – Root info.

Run locally:
    uvicorn main:app --reload --port 8000

Render start command:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# ── Lifespan: seed ChromaDB if empty ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: ensure ChromaDB is seeded with sample data."""
    import vector_store as vs
    from sample_schedule import generate_sample_events

    count = vs.collection_count()
    if count == 0:
        print("[startup] ChromaDB is empty — seeding with sample schedule data …")
        events = generate_sample_events()
        vs.ingest_events(events)
        print(f"[startup] ✅ Seeded {vs.collection_count()} events.")
    else:
        print(f"[startup] ChromaDB already has {count} events. Skipping seed.")

    yield  # application runs here

    print("[shutdown] Goodbye.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Schedule Assistant API",
    description=(
        "An Agentic RAG-based Schedule Assistant powered by LangGraph 1.x and ChromaDB. "
        "Manage your 30-day schedule with natural language."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (conversation history) ────────────────────────────
# For production, replace with Redis or a DB-backed store.
_sessions: dict[str, list[dict]] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's natural-language message.")
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. A new one is created if omitted.",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    turn: int


class ResetRequest(BaseModel):
    session_id: str


class ScheduleEvent(BaseModel):
    id: str
    title: str
    date: str
    start_time: str
    end_time: str
    type: str
    location: str
    attendees: str
    description: str


class ScheduleResponse(BaseModel):
    date: str
    events: list[ScheduleEvent]
    count: int


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    today: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Schedule Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /chat": "Chat with the schedule agent",
            "POST /chat/reset": "Clear conversation history",
            "GET /schedule": "Fetch today's schedule (optional ?date=YYYY-MM-DD)",
            "GET /health": "Health check",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    import vector_store as vs
    return HealthResponse(
        status="ok",
        vector_store_count=vs.collection_count(),
        today=date.today().isoformat(),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat_endpoint(req: ChatRequest):
    """
    Send a message to the Schedule Assistant agent.

    The agent uses RAG over a ChromaDB vector store and two tools:
    - **get_schedule**: Retrieves events by date, day name, or semantic query.
    - **update_schedule**: Adds, updates, or deletes events.

    Conversation history is maintained per `session_id`.

    **Example queries:**
    - "What do I have scheduled tomorrow?"
    - "Am I free Friday afternoon?"
    - "Add a meeting on August 25 at 3 PM"
    - "Move my morning stand-up to 10 AM tomorrow"
    - "Cancel my doctor appointment"
    """
    from agent import chat

    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])

    try:
        reply, updated_history = chat(req.message, history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _sessions[session_id] = updated_history
    turn = sum(1 for m in updated_history if m["role"] == "human")

    return ChatResponse(reply=reply, session_id=session_id, turn=turn)


@app.post("/chat/reset", tags=["Agent"])
def reset_session(req: ResetRequest):
    """Clear conversation history for a given session."""
    if req.session_id in _sessions:
        del _sessions[req.session_id]
        return {"message": f"Session {req.session_id} cleared."}
    return {"message": f"Session {req.session_id} not found (already cleared or never started)."}


@app.get("/schedule", response_model=ScheduleResponse, tags=["Schedule"])
def get_today_schedule(date_param: Optional[str] = None):
    """
    REST endpoint to fetch the schedule for a specific date.
    Defaults to today if no `date` query param is provided.

    Query params:
      - date (optional): YYYY-MM-DD format.
    """
    import vector_store as vs

    target_date = date_param or date.today().isoformat()

    # Validate format
    try:
        date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{target_date}'. Use YYYY-MM-DD.",
        )

    events = vs.get_events_by_date(target_date)
    return ScheduleResponse(
        date=target_date,
        events=[ScheduleEvent(**{k: e.get(k, "") for k in ScheduleEvent.model_fields}) for e in events],
        count=len(events),
    )
