"""
agent.py
--------
LangGraph 1.x Schedule Assistant Agent.

Architecture:
  StateGraph with two nodes:
    - "agent"  : LLM decides whether to respond or call a tool
    - "tools"  : Executes the chosen tool and returns result to agent

  Flow: START → agent → (tool call?) → tools → agent → ... → END

Uses:
  - langgraph.graph.StateGraph + MessageState  (v1.x canonical API)
  - langgraph.prebuilt.ToolNode               (handles tool dispatch)
  - ChatOpenAI with bind_tools                 (function-calling)
"""

from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessageState
from langgraph.prebuilt import ToolNode

from tools import get_schedule, update_schedule

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_llm = ChatOpenAI(
    model=_MODEL,
    temperature=0,
    streaming=False,
)

# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS = [get_schedule, update_schedule]
_llm_with_tools = _llm.bind_tools(TOOLS)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = f"""You are a smart, helpful personal Schedule Assistant.

Today's date is {date.today().isoformat()} ({date.today().strftime('%A, %B %d, %Y')}).

You have access to two tools:
1. get_schedule   – Retrieves schedule information. Use this whenever the user asks about
                    their calendar, upcoming events, free time, or anything schedule-related.
2. update_schedule – Adds, updates, or deletes schedule entries. Use this when the user
                    wants to create a new event, move/reschedule an event, or cancel/delete one.

Guidelines:
- Always use get_schedule before answering schedule-related questions — do not guess.
- When adding an event, collect all necessary details (title, date, time) before calling the tool.
  If the user's message already contains them, proceed immediately.
- When updating or deleting, first use get_schedule to confirm what exists, then act.
- After a tool call, interpret the raw result into a friendly, conversational response.
- If an event is not found, tell the user clearly and offer to add it or refine the search.
- Format times in 12-hour format (e.g. 3:00 PM) in your replies, even though the data uses 24-hour.
- Be concise — the user wants quick, accurate answers.
"""

# ── Graph nodes ───────────────────────────────────────────────────────────────

def agent_node(state: MessageState) -> dict:
    """
    LLM node: decides whether to respond directly or invoke a tool.
    Prepends the system message on every call.
    """
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessageState) -> str:
    """
    Routing function: if the last message has tool calls, go to 'tools';
    otherwise end the conversation turn.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph."""
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(MessageState)

    # Register nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")   # after tool execution, return to agent

    return graph.compile()


# Compiled graph (singleton, imported by main app)
schedule_agent = build_graph()


# ── Convenience chat function ─────────────────────────────────────────────────

def chat(user_message: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """
    Send a message to the agent and return (reply_text, updated_history).

    Args:
        user_message: The user's latest message.
        history:      Previous messages in LangChain format:
                      [{"role": "human"|"ai", "content": "..."}]

    Returns:
        Tuple of (assistant reply string, full updated message history).
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # Build message list
    lc_history = []
    for msg in (history or []):
        if msg["role"] == "human":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    lc_history.append(HumanMessage(content=user_message))

    result = schedule_agent.invoke({"messages": lc_history})

    # Extract the last AI message
    reply = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            reply = msg.content
            break

    # Build updated history (human + ai only, skip tool messages)
    updated_history = list(history or [])
    updated_history.append({"role": "human", "content": user_message})
    updated_history.append({"role": "ai", "content": reply})

    return reply, updated_history


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Schedule Assistant (type 'quit' to exit)\n")
    history: list[dict] = []

    demo_queries = [
        "What do I have scheduled tomorrow?",
        "Am I free Friday afternoon?",
        "What workshops are coming up?",
        'Add a team sync meeting on August 25 at 2 PM in the main conference room.',
        'Move my morning stand-up tomorrow to 10 AM.',
    ]

    for q in demo_queries:
        print(f"You: {q}")
        reply, history = chat(q, history)
        print(f"Assistant: {reply}\n{'-'*60}")
