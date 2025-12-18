#!/usr/bin/env python3
"""
Terminal Jira Agent (LangGraph + MCP + Gemini 2.5 Flash)
- Colorful, structured logs for chats and tool interactions
- Streams step-by-step agent updates (LLM & Tools) in real time
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
from rich.text import Text
from rich import box

# LangChain / LangGraph
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, HumanMessage

# Import LLM creation from centralized module
from llm_initial import create_jira_agent_llm, JIRA_AGENT_LLM_MODEL

# ---------- Config ----------
JIRA_MCP_URL = os.getenv("JIRA_MCP_URL", "http://localhost:9003/mcp")
MODEL_NAME = os.getenv("GEMINI_MODEL", JIRA_AGENT_LLM_MODEL)
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.5"))

AGENT_SYSTEM_PROMPT = """You are Blue AI Coacher, a helpful Jira assistant.
When the user asks anything about Jira (issues, sprints, boards, epics, versions),
use the Jira MCP tools when needed (e.g., jira_search, jira_get_agile_boards,
jira_get_sprints_from_board, jira_get_issue, jira_update_issue, jira_transition_issue).
Summarize results in neat tables. Prefer: key, summary, assignee, status, priority, due.
If a sprint is given by NAME only, try: sprint = "Name". If ambiguous, explain clearly.
"""

console = Console(highlight=False)

# ---------- Pretty helpers ----------
def banner():
    title = Text("Blue AI Coacher — Jira Agent", style="bold white")
    subtitle = Text("LangGraph ReAct + MCP tools + Gemini 2.5 Flash", style="italic cyan")
    console.print(Panel.fit(Text.assemble(title, "\n", subtitle),
                            border_style="blue", box=box.ROUNDED))

def log_user(text: str):
    console.print(Panel.fit(text, title="you", title_align="left",
                            border_style="bright_green", box=box.ROUNDED))

def log_assistant(text: str):
    console.print(Panel.fit(text, title="assistant", title_align="left",
                            border_style="bright_magenta", box=box.ROUNDED))

def log_event(title: str, payload: Any, border="cyan"):
    try:
        if isinstance(payload, (dict, list)):
            j = json.dumps(payload, ensure_ascii=False, indent=2)
            console.print(Panel(JSON.from_data(json.loads(j)),
                                title=title, border_style=border, box=box.ROUNDED))
        else:
            console.print(Panel(str(payload), title=title, border_style=border, box=box.ROUNDED))
    except Exception:
        console.print(Panel(str(payload), title=title, border_style=border, box=box.ROUNDED))

def msg_role(m: Any) -> str:
    if isinstance(m, dict):
        return m.get("role") or m.get("type") or "unknown"
    if isinstance(m, BaseMessage):
        return getattr(m, "type", "unknown")  # e.g., "human", "ai", "tool", "system"
    return type(m).__name__

def msg_content(m: Any) -> str:
    if isinstance(m, dict):
        return m.get("content", "")
    if isinstance(m, BaseMessage):
        return str(getattr(m, "content", ""))
    return str(m)

def pretty_messages_table(messages: List[Any]) -> Table:
    t = Table(title="Conversation (last 8)", title_style="bold cyan", box=box.SIMPLE, show_lines=False)
    t.add_column("Role", style="yellow", no_wrap=True)
    t.add_column("Content", style="white")
    for m in messages[-8:]:
        role = msg_role(m)
        content = msg_content(m)
        # Trim ultra-long blobs to keep UI readable
        if len(content) > 2000:
            content = content[:2000] + " … [truncated]"
        t.add_row(role, content)
    return t

# ---------- Build pieces ----------
def build_llm():
    """Build LLM instance using centralized configuration."""
    return create_jira_agent_llm(model=MODEL_NAME, temperature=TEMPERATURE)

async def load_mcp_tools():
    client = MultiServerMCPClient({
        "jira": {"url": JIRA_MCP_URL, "transport": "streamable_http"}
    })
    tools = await client.get_tools()
    return tools

async def build_agent():
    llm = build_llm()
    tools = await load_mcp_tools()

    tool_names = [getattr(t, "name", getattr(t, "__name__", type(t).__name__)) for t in tools]
    log_event("Loaded MCP Tools", tool_names, border="bright_blue")

    agent = create_react_agent(
        llm,
        tools,
        prompt=AGENT_SYSTEM_PROMPT,
        version="v2"
    )
    return agent

# ---------- Run one turn with streaming ----------
async def run_one_turn(agent, messages: List[Any]) -> List[Any]:
    console.rule("[bold]agent run[/bold]")

    # Stream per-step updates (LLM/tool) and show raw event payloads for full transparency
    async for update in agent.astream({"messages": messages}, stream_mode="updates"):
        log_event("step update", update, border="bright_cyan")

    # Final state (includes full message history with tool traces)
    state = await agent.ainvoke({"messages": messages})
    updated = state.get("messages", messages)

    # Find latest assistant message
    ai_text = ""
    for m in reversed(updated):
        role = msg_role(m)
        if role in ("assistant", "ai"):
            ai_text = msg_content(m)
            break

    log_assistant(ai_text or "[no assistant message]")
    return updated

# ---------- CLI loop ----------
async def chat_loop():
    banner()
    agent = await build_agent()

    messages: List[Any] = []

    console.print("[dim]Type your question, or 'exit' to quit.[/dim]")
    console.print("Examples:")
    console.print("  • list sprints for project BAP")
    console.print('  • show "Sprint 5" To Do issues in BAP (name-only)')
    console.print("  • show my BAP tasks due today\n")

    while True:
        try:
            user = console.input("[bright_green]you> [/bright_green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye![/dim]")
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            console.print("[dim]bye![/dim]")
            return

        log_user(user)
        # Keep conversation as LangChain messages
        messages.append(HumanMessage(content=user))

        console.print(pretty_messages_table(messages))

        try:
            messages = await run_one_turn(agent, messages)
        except Exception as e:
            console.print(Panel.fit(str(e), title="error", border_style="red"))
            continue

if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except Exception as e:
        Console().print(Panel.fit(str(e), title="fatal", border_style="red"))
        sys.exit(1)
