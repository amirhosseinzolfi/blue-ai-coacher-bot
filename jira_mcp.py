#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import re
import argparse
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, types

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:9000/mcp")
PROJECT_KEY = os.getenv("PROJECT_KEY", "BAP")
BOARD_ID = int(os.getenv("BOARD_ID", "18"))

console = Console()

def status_style(name: str) -> str:
    n = (name or "").lower()
    if "to do" in n or "todo" in n or "backlog" in n:
        return "bold bright_cyan"
    if "in progress" in n or "doing" in n or "selected" in n:
        return "bold yellow"
    if "done" in n or "closed" in n or "resolved" in n or "qa passed" in n:
        return "bold green"
    if "blocked" in n or "hold" in n:
        return "bold red"
    return "white"

def to_json(result: types.CallToolResult) -> Any:
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    for c in result.content:
        if isinstance(c, types.TextContent):
            try:
                return json.loads(c.text)
            except Exception:
                return {"text": c.text}
    return {}

# Tool resolver cache
_tool_cache: Dict[str, str] = {}

async def resolve_tool_name(session: ClientSession, preferred: str) -> str:
    global _tool_cache
    if not _tool_cache:
        tool_list = await session.list_tools()
        names = [t.name for t in tool_list.tools]
        for n in names:
            _tool_cache[n] = n

    # direct match
    if preferred in _tool_cache:
        return preferred

    # try without jira_ prefix
    if preferred.startswith("jira_"):
        alt = preferred[len("jira_"):]
        if alt in _tool_cache:
            return alt

    # try adding jira_ prefix
    alt2 = "jira_" + preferred
    if alt2 in _tool_cache:
        return alt2

    # fuzzy fallback: match ignoring underscores
    wanted = preferred.replace("_", "")
    for n in _tool_cache:
        if n.replace("_", "") == wanted:
            return n
    for n in _tool_cache:
        if wanted in n.replace("_", ""):
            return n

    # fallback to original
    return preferred

async def call_tool(session: ClientSession, tool: str, args: Dict[str, Any]) -> Any:
    actual = await resolve_tool_name(session, tool)
    # try calling with resolved name
    try:
        res = await session.call_tool(actual, args)
    except Exception as e:
        # if original had jira_ prefix, try without it
        alt = tool
        if tool.startswith("jira_"):
            alt = tool[len("jira_"):]
        if alt != actual:
            res = await session.call_tool(alt, args)
            actual = alt
        else:
            raise

    if res.isError:
        msg = ""
        for c in res.content:
            if isinstance(c, types.TextContent):
                msg += c.text + "\n"
        raise RuntimeError(f"Tool {actual} failed: {msg.strip() or 'unknown error'}")
    return to_json(res)

async def get_active_sprint_id(session: ClientSession, board_id: int) -> Optional[int]:
    data = await call_tool(session, "jira_get_sprints_from_board", {"board_id": board_id})
    sprints = data.get("values") if isinstance(data, dict) and "values" in data else data
    if not isinstance(sprints, list):
        raise RuntimeError(f"Unexpected sprints payload: {data}")
    for s in sprints:
        if (s.get("state") or "").lower() == "active":
            return int(s.get("id"))
    return None

def render_issues_table(issues: List[Dict[str, Any]], title: str) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Key", style="bold")
    table.add_column("Summary", overflow="fold", max_width=80)
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Assignee")
    table.add_column("Priority")
    table.add_column("Updated")

    for issue in issues:
        key = issue.get("key") or issue.get("id") or "?"
        fields = issue.get("fields") or {}
        summary = fields.get("summary", "")
        issuetype = (fields.get("issuetype") or {}).get("name", "")
        status_name = (fields.get("status") or {}).get("name", "")
        assignee = fields.get("assignee") or {}
        assignee_name = assignee.get("displayName") or assignee.get("name") or "-"
        priority = (fields.get("priority") or {}).get("name", "")
        updated = fields.get("updated") or fields.get("updatedDate") or ""
        style = status_style(status_name)
        table.add_row(
            str(key),
            str(summary),
            str(issuetype),
            f"[{style}]{status_name}[/{style}]",
            str(assignee_name),
            str(priority),
            str(updated),
        )

    console.print(table)

async def list_todo_in_active_sprint(server_url: str, project_key: str, board_id: int, headers: Optional[Dict[str, str]] = None) -> None:
    async with streamablehttp_client(server_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            sprint_id = await get_active_sprint_id(session, board_id)
            if not sprint_id:
                console.print(Panel.fit(f"No active sprint found on board {board_id}", style="red"))
                return

            data = await call_tool(session, "jira_get_sprint_issues", {"sprint_id": sprint_id})
            issues = data.get("issues") if isinstance(data, dict) and "issues" in data else data
            if not isinstance(issues, list):
                raise RuntimeError(f"Unexpected issues payload: {data}")

            # فیلتر To Do و پروژه
            todo = [
                i for i in issues
                if ((i.get("fields") or {}).get("status") or {}).get("name", "").lower() in ("to do", "todo")
                and (((i.get("fields") or {}).get("project") or {}).get("key", "").upper() == project_key.upper())
            ]

            if not todo:
                console.print(Panel.fit("No TODO issues in active sprint.", style="yellow"))
            else:
                render_issues_table(todo, f"TODO in Active Sprint (Board {board_id}, Sprint {sprint_id}) — Project {project_key}")

async def list_all_project_issues(server_url: str, project_key: str, headers: Optional[Dict[str, str]] = None) -> None:
    jql = f"project = {project_key} ORDER BY updated DESC"
    async with streamablehttp_client(server_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            data = await call_tool(session, "jira_search", {"jql": jql})
            issues = data.get("issues") if isinstance(data, dict) and "issues" in data else data
            if not isinstance(issues, list):
                raise RuntimeError(f"Unexpected issues payload: {data}")
            if not issues:
                console.print(Panel.fit("No issues found in project.", style="yellow"))
            else:
                render_issues_table(issues, f"All Issues in Project {project_key}")

async def print_tools(server_url: str, headers: Optional[Dict[str, str]] = None) -> None:
    async with streamablehttp_client(server_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tab = Table(title="MCP Tools", show_lines=True)
            tab.add_column("Name")
            tab.add_column("Description")
            for t in tools.tools:
                tab.add_row(t.name, (t.description or "")[:120])
            console.print(tab)

def parse_headers(header_kv: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for kv in header_kv or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            headers[k.strip()] = v.strip()
    return headers

def cli():
    parser = argparse.ArgumentParser(description="MCP Jira viewer")
    parser.add_argument("--server", default=os.getenv("SERVER_URL", SERVER_URL))
    parser.add_argument("--project", default=os.getenv("PROJECT_KEY", PROJECT_KEY))
    parser.add_argument("--board", type=int, default=int(os.getenv("BOARD_ID", BOARD_ID)))
    parser.add_argument("--mode", choices=["todo-active-sprint", "project-all"], default="todo-active-sprint")
    parser.add_argument("--header", action="append", default=[], help="Extra HTTP header, e.g. Authorization=Bearer <token>")
    parser.add_argument("--list-tools", action="store_true", help="List available tools on MCP server")

    args = parser.parse_args()
    headers = parse_headers(args.header)

    if args.list_tools:
        asyncio.run(print_tools(args.server, headers))
        return

    if args.mode == "todo-active-sprint":
        asyncio.run(list_todo_in_active_sprint(args.server, args.project, args.board, headers))
    else:
        asyncio.run(list_all_project_issues(args.server, args.project, headers))

if __name__ == "__main__":
    cli()
