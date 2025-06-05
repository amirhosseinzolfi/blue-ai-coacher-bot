#!/usr/bin/env python3
"""
Terminal-based LangGraph chatbot with SQLite-backed task management.
Synchronous mode only: each user message yields one full assistant response.
"""

import sqlite3
import uuid
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import logging
from rich.console import Console
from rich.logging import RichHandler
import os
from pathlib import Path

# configure rich logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("task_agent")
console = Console()

# --- Step 1: Initialize the SQLite tasks DB ---
def init_tasks_db(db_path: str = "tasks.db"):
    logger.info(f"[bold green]Initializing tasks DB[/] at [cyan]{db_path}[/]")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        username    TEXT,
        task_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name   TEXT NOT NULL,
        task_status TEXT CHECK(task_status IN ('to do','done'))
                       NOT NULL DEFAULT 'to do',
        description TEXT
    );
    """)
    conn.commit()
    conn.close()
    logger.info("[bold green]Tasks DB ready[/]")

# --- Step 2: Build the LangChain SQLDatabaseToolkit ---
def create_sql_toolkit(
    db_path: str = "tasks.db",
    ollama_url: str = "http://localhost:15203/v1",
    model_name: str = "gpt-4o",  # Changed default from gpt-4o to a more widely supported model
    api_key: str = "324",
    fallback_models: list = None,
) -> SQLDatabaseToolkit:
    logger.info("[bold yellow]Creating SQL toolkit[/]…")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False
    )
    db = SQLDatabase(engine)
    
    if fallback_models is None:
        fallback_models = ["gemini-1.5-flash", "gpt-4o", "gpt-4o-mini"]
    
    # If the specified model is not in fallback_models, add it to ensure it's tried first
    if model_name not in fallback_models:
        fallback_models.insert(0, model_name)
    else:
        # Move the specified model to the front if it's in the list
        fallback_models.remove(model_name)
        fallback_models.insert(0, model_name)
    
    # Try models in order until one works
    last_error = None
    for model in fallback_models:
        try:
            llm = ChatOpenAI(
                base_url=ollama_url,
                model_name=model,
                temperature=0.5,
                api_key=api_key,
                max_retries=2,
            )
            # Test the model with a simple prompt to ensure it's working
            llm.invoke("Say hello")
            logger.info(f"[bold green]Successfully connected[/] using model [magenta]{model}[/] at [cyan]{ollama_url}[/]")
            return SQLDatabaseToolkit(db=db, llm=llm)
        except Exception as e:
            logger.warning(f"[yellow]Failed to initialize model {model}:[/] {str(e)}")
            last_error = e
    
    # If we reached here, none of the models worked
    error_msg = f"Failed to initialize any AI model. Last error: {str(last_error)}"
    logger.error(f"[bold red]{error_msg}[/]")
    raise RuntimeError(error_msg)

# --- Step 3: Register CRUD tools via @tool decorator ---
def register_crud_tools(toolkit: SQLDatabaseToolkit):
    def get_checker():  return QuerySQLCheckerTool(db=toolkit.db, llm=toolkit.llm)
    def get_executor(): return QuerySQLDatabaseTool(db=toolkit.db)

    @tool("add_task", description="Add a new task for the given user.")
    def add_task(username: str, task_name: str, description: str) -> str:
        sql = """
        INSERT INTO tasks (username, task_name, task_status, description)
        VALUES (:username, :task_name, 'to do', :description);
        """
        valid_sql = get_checker().run({"dialect": "SQLite", "query": sql})
        get_executor().run({
            "query": valid_sql,
            "username": username,
            "task_name": task_name,
            "description": description
        })
        return f"✅ Added task “{task_name}”."

    @tool("complete_task", description="Mark an existing task as done.")
    def complete_task(username: str, task_id: int) -> str:
        sql = """
        UPDATE tasks
        SET task_status = 'done'
        WHERE username = :username AND task_id = :task_id;
        """
        valid_sql = get_checker().run({"dialect": "SQLite", "query": sql})
        get_executor().run({
            "query": valid_sql,
            "username": username,
            "task_id": task_id
        })
        return f"🎉 Marked task #{task_id} as done."

    @tool("delete_task", description="Delete a task by its ID.")
    def delete_task(username: str, task_id: int) -> str:
        sql = "DELETE FROM tasks WHERE username = :username AND task_id = :task_id;"
        valid_sql = get_checker().run({"dialect": "SQLite", "query": sql})
        get_executor().run({
            "query": valid_sql,
            "username": username,
            "task_id": task_id
        })
        return f"🗑️ Deleted task #{task_id}."

    @tool("list_tasks", description="List all tasks for the user.")
    def list_tasks(username: str) -> str:
        sql = """
        SELECT task_id, task_name, task_status, description
        FROM tasks
        WHERE username = :username
        ORDER BY task_id;
        """
        valid_sql = get_checker().run({"dialect": "SQLite", "query": sql})
        rows = get_executor().run({"query": valid_sql, "username": username})
        if not rows:
            return "You have no tasks."
        lines = [f"{r[0]}. [{r[2]}] {r[1]} — {r[3] or ''}" for r in rows]
        return "📝 Your tasks:\n" + "\n".join(lines)

    tools = ["add_task", "complete_task", "delete_task", "list_tasks"]
    logger.info(f"[bright_blue]Registered CRUD tools[/]: {', '.join(tools)}")
    return [add_task, complete_task, delete_task, list_tasks]

# --- Step 4: Build the LangGraph agent with proper checkpointing ---
def build_agent(username: str, thread_id: str, checkpointer):
    logger.info(f"[bold green]Building agent[/] for user [cyan]{username}[/] with thread [cyan]{thread_id}[/]")
    init_tasks_db()
    toolkit    = create_sql_toolkit()
    base_tools = toolkit.get_tools()
    crud_tools = register_crud_tools(toolkit)
    tools      = base_tools + crud_tools

    system_message = """
    You are a business-management assistant. When the user asks about tasks,
    choose exactly one of these tools by name: add_task, complete_task,
    delete_task, or list_tasks, and invoke it with the correct parameters.
    Otherwise, reply normally without calling any tool.
    """

    agent = create_react_agent(
        toolkit.llm,
        tools=tools,
        prompt=system_message,
        checkpointer=checkpointer
    )
    logger.info("[bold green]Agent built successfully[/]")
    return agent

# --- Step 5: Terminal chat loop (synchronous, with thread_id) ---
def chat_loop():
    username = input("👤 Your username: ").strip()
    thread_id = str(uuid.uuid4())
    logger.info(f"[bold green]Starting chat loop[/] for [cyan]{username}[/] with thread ID [cyan]{thread_id}[/]")
    
    # Create database directory if it doesn't exist
    db_dir = Path("./data")
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "agent_state.db"
    
    # Initialize SQLite database for checkpointer
    logger.info(f"[bold green]Initializing checkpointer DB[/] at [cyan]{db_path}[/]")
    
    # Use the proper context manager pattern
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        try:
            # Build agent with the checkpointer
            agent = build_agent(username, thread_id, checkpointer)

            console.print("\n🚀 [bold magenta]Agent ready![/] Type [yellow]'exit'[/] or [yellow]'quit'[/].\n")
            history = []
            while True:
                user_input = input("> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    console.print("[bold red]Goodbye![/]")
                    break

                logger.info(f"[bold blue]User:[/] {user_input}")
                history.append(("user", user_input))
                
                # Build chat messages
                msgs = [{"role": r, "content": c} for r, c in history]

                try:
                    logger.debug(f"Invoking agent with {len(msgs)} message(s)")
                    result = agent.invoke(
                        {"messages": msgs},
                        config={"configurable": {"thread_id": thread_id, "checkpoint_ns": "task_manager"}}
                    )

                    assistant_msg = result["messages"][-1]
                    reply_text = assistant_msg.content

                    console.print(f"[bold green]Assistant:[/] {reply_text}")
                    logger.info(f"[bold green]Assistant response:[/] {reply_text}")
                    history.append(("assistant", reply_text))

                    # Show tool usage if applicable
                    if "intermediate_steps" in result:
                        for step in result["intermediate_steps"]:
                            logger.info(f"[bold yellow]Tool used:[/] {step[0].tool}")
                            logger.debug(f"Tool input: {step[0].tool_input}")
                            logger.debug(f"Tool output: {step[1]}")
                
                except Exception as e:
                    logger.error(f"[bold red]Error during agent invocation:[/] {str(e)}")
                    console.print(f"[bold red]Sorry, I encountered an error:[/] {str(e)}")
        
        except Exception as e:
            logger.error(f"[bold red]Fatal error:[/] {str(e)}")
            console.print(f"[bold red]Fatal error:[/] {str(e)}")

if __name__ == "__main__":
    try:
        console.print("[bold blue]Starting Task Agent[/] [italic]v1.0[/]")
        chat_loop()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user. Exiting...[/]")
    except Exception as e:
        logger.exception("[bold red]Unhandled exception:[/]")
        console.print(f"[bold red]Unhandled exception:[/] {str(e)}")
