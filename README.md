# Agentic RAG Schedule Assistant

A production-ready Schedule Assistant powered by **LangGraph 1.x**, **ChromaDB**, and **FastAPI**. It manages a user's 30-day calendar using an agentic RAG pipeline that retrieves and modifies schedule data with natural language.

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────┐
│           FastAPI  (main.py)            │
│  POST /chat  ─►  agent.chat()           │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       LangGraph 1.x Agent Graph         │
│                                         │
│  START → [agent node] ──────────────►  END
│               │  ▲                      │
│          tool │  │ result               │
│               ▼  │                      │
│          [tools node]                   │
│    ┌──────────┴───────────┐             │
│    │  get_schedule        │             │
│    │  update_schedule     │             │
│    └──────────┬───────────┘             │
└──────────────-│─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│        ChromaDB  (vector_store.py)      │
│  Persistent  •  Cosine similarity       │
│  all-MiniLM-L6-v2 embeddings (local)   │
└─────────────────────────────────────────┘
```

### Components

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — HTTP endpoints, lifespan (auto-seed) |
| `agent.py` | LangGraph 1.x graph — agent + tools nodes, `chat()` helper |
| `tools.py` | `get_schedule` and `update_schedule` LangChain tools |
| `vector_store.py` | ChromaDB init, ingest, semantic search, CRUD |
| `sample_schedule.py` | Generates 30-day sample events |
| `ingest.py` | CLI script to seed ChromaDB |

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd schedule-assistant
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Seed the vector store

```bash
python ingest.py
# Reset and re-seed:
python ingest.py --reset
```

### 4. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

### `POST /chat`
Chat with the schedule agent.

```json
{
  "message": "What do I have scheduled tomorrow?",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "reply": "You have 3 events tomorrow: ...",
  "session_id": "abc-123",
  "turn": 1
}
```

### `GET /schedule?date=2026-08-20`
REST endpoint to fetch events for a date.

### `POST /chat/reset`
Clear conversation history for a session.

### `GET /health`
Health check — returns status and ChromaDB document count.

---

## Example Queries

| Query | What happens |
|---|---|
| "What do I have scheduled tomorrow?" | `get_schedule("tomorrow")` → returns events |
| "Am I free Friday afternoon?" | `get_schedule("Friday afternoon")` → semantic search |
| "Add a meeting on August 25 at 3 PM" | `update_schedule(action="add", ...)` |
| "Move my stand-up meeting to 10 AM tomorrow" | `get_schedule` to find it, then `update_schedule(action="update", ...)` |
| "Cancel my dentist appointment" | `get_schedule` to find it, then `update_schedule(action="delete", ...)` |
| "What workshops are coming up?" | `get_schedule("workshops")` → semantic search |

---

## Deploy to Render

### Using Docker (recommended)

1. Push your code to GitHub/GitLab.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Select **Docker** runtime.
4. Set environment variable `OPENAI_API_KEY` in the Render dashboard.
5. Deploy — the `render.yaml` handles the rest.

> **Note on persistence:** The free Render tier does not support persistent disks.  
> ChromaDB will be re-seeded from `sample_schedule.py` on every cold start.  
> For production, upgrade to the Starter plan and uncomment the `disk:` section in `render.yaml`.

### Using render.yaml (Infrastructure as Code)

```bash
# Install the Render CLI (optional)
render deploy
```

Or connect your repo to Render and it will auto-detect `render.yaml`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Path for ChromaDB storage |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 1.0.0 |
| LLM | OpenAI GPT-4o-mini (via langchain-openai) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Vector store | ChromaDB 0.5.x (persistent) |
| API | FastAPI 0.115 + Uvicorn |
| Deployment | Render (Docker) |
