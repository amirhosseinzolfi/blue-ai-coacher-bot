from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, RemoveMessage
import json
import uuid
import operator
from typing import List, Optional, Annotated
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging
import threading
import datetime
import requests
import time
import os
try:
    from g4f.api import run_api
except ImportError:
    log.error("[bold red]g4f.api module not found. Install the 'g4f' package.[/bold red]")
    run_api = None

# --- Initialize Logging ---
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
log = logging.getLogger("task_agent")

# --- G4F API Server Functions ---
def start_g4f_api():
    """Starts the G4F API server in a daemon thread"""
    log.info("[bold cyan]Starting G4F API server on http://localhost:16201/v1 ...[/bold cyan]")
    run_api(bind="0.0.0.0:16201")

def wait_for_api_server(timeout=30):
    """Waits for the G4F API server to become available"""
    base_url = "http://localhost:16201/v1/chat/completions"
    start_time = datetime.datetime.now()
    log.info("[bold yellow]Waiting for the G4F API server to become available...[/bold yellow]")
    
    while True:
        try:
            response = requests.post(
                base_url, 
                json={"messages": [{"role": "system", "content": "ping"}]}, 
                timeout=5
            )
            if response.ok:
                log.info("[bold green]G4F API server responded successfully.[/bold green]")
                break
        except Exception:
            pass
            
        if (datetime.datetime.now() - start_time).seconds > timeout:
            log.error("[bold red]API server not available after waiting 30 seconds.[/bold red]")
            return False
        time.sleep(1)
    return True

# --- Initialize LLM ---
def initialize_llm():
    """Initialize LLM with local G4F API"""
    return ChatOpenAI(
        base_url="http://localhost:16201/v1",
        model_name="gemini-2.0-flash",  # or your preferred model
        temperature=0.5,
        api_key="not-needed"  # G4F doesn't require an API key
    )

# --- Define State ---
class State(TypedDict):
    conversation_history: Annotated[List[BaseMessage], operator.add]
    tasks: list
    summary: Optional[str]

# --- Task Persistence Functions ---
TASKS_FILE = "tasks.json"

def load_tasks() -> list:
    """Load tasks from tasks.json if exists, else return empty list."""
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error loading tasks: {e}")
    return []

def save_tasks(tasks: list):
    """Save tasks list to tasks.json."""
    try:
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=2)
    except Exception as e:
        log.error(f"Error saving tasks: {e}")

# --- Task Management Functions ---
def add_task(state: State, task: str):
    state['tasks'].append({"task": task, "done": False})
    save_tasks(state['tasks'])

def check_task_done(state: State, task_index: int):
    if 0 <= task_index < len(state['tasks']):
        state['tasks'][task_index]['done'] = True
        save_tasks(state['tasks'])

def refine_tasks(state: State):
    # Logic to refine tasks based on user input; for example, reloading and modifying tasks
    tasks = load_tasks()
    # ...existing refinement logic...
    state['tasks'] = tasks
    save_tasks(state['tasks'])

def display_tasks(state: State):
    """Display current tasks in a table format"""
    table = Table(title="Current Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Task", style="white")
    table.add_column("Status", style="green")
    
    for idx, task in enumerate(state['tasks']):
        status = "✓" if task['done'] else "○"
        table.add_row(str(idx), task['task'], status)
    
    console.print(table)

# --- Message Management Functions ---
def summarize_conversation(state: State):
    """Summarizes conversation and prunes old messages."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Summarizing conversation...", total=None)
        
        messages = state["conversation_history"]
        current_summary = state.get("summary")
        
        # Create summarization prompt
        summary_messages = []
        if current_summary:
            summary_messages.append(SystemMessage(content=f"Previous summary:\n{current_summary}"))
        summary_messages.extend(messages)
        summary_messages.append(HumanMessage(content="Summarize the conversation above, including any tasks discussed."))
        
        try:
            # Get new summary using proper invocation
            summary_response = llm.invoke(summary_messages)
            new_summary = summary_response.content
        except Exception as e:
            log.error(f"Error during summarization: {e}")
            new_summary = "Summary unavailable due to an error."
        
        # Display summary
        console.print(Panel(
            new_summary,
            title="[cyan]Conversation Summary[/cyan]",
            border_style="blue"
        ))

        # Keep only last 2 messages
        messages_to_keep = 2
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages[:-messages_to_keep]]
        log.info(f"Pruned {len(messages_to_remove)} messages, keeping last {messages_to_keep}")

        return {"summary": new_summary, "conversation_history": messages_to_remove}

def handle_user_input(state: State):
    """Process user input with context from summary."""
    user_input = console.input("[bold green]You:[/bold green] ")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing input...", total=None)
        
        # Prepare messages for LLM
        messages_to_send = []
        if state.get("summary"):
            messages_to_send.append(SystemMessage(content=f"Previous context:\n{state['summary']}"))
        messages_to_send.extend(state["conversation_history"])
        messages_to_send.append(HumanMessage(content=user_input))
        
        try:
            # Process with LLM using proper invocation
            response = llm.invoke(messages_to_send)
        except Exception as e:
            log.error(f"Error invoking LLM: {e}")
            response = AIMessage(content="Sorry, there was an error processing your request.")
        
        # Handle tasks
        if "add task" in user_input.lower():
            task_desc = user_input.split("add task")[-1].strip()
            add_task(state, task_desc)
            log.info(f"✓ Added task: {task_desc}")
            display_tasks(state)
        elif "check task" in user_input.lower():
            try:
                task_index = int(user_input.split("check task")[-1].strip())
                check_task_done(state, task_index)
                log.info(f"✓ Marked task {task_index} as done")
                display_tasks(state)
            except Exception as e:
                log.error(f"Error processing 'check task': {e}")
        
        # Display AI response
        console.print("[bold blue]Assistant:[/bold blue]", response.content)
        
        return {"conversation_history": [HumanMessage(content=user_input), response]}

def should_continue_or_summarize(state: State) -> Literal["summarize", "__end__"]:
    """Decide whether to summarize or end turn."""
    if len(state["conversation_history"]) > 6:  # Threshold for summarization
        return "summarize"
    return END

# --- Build the Graph ---
builder = StateGraph(State)
builder.add_node("agent", handle_user_input)
builder.add_node("summarize", summarize_conversation)

# Define edges
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue_or_summarize,
    {
        "summarize": "summarize",
        END: END
    }
)
builder.add_edge("summarize", END)

# --- Set Up Memory ---
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- Initialize State ---
initial_state = {
    "conversation_history": [],
    "tasks": load_tasks(),
    "summary": None
}

# --- Run the Agent ---
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "checkpoint_ns": "blue_business",
        "checkpoint_id": str(uuid.uuid4())
    }
}

def main():
    # Start G4F API server if available
    if run_api is not None:
        api_thread = threading.Thread(
            target=start_g4f_api, 
            daemon=True, 
            name="G4F-API-Thread"
        )
        api_thread.start()
        
        # Wait for API server to be ready
        if not wait_for_api_server():
            console.print("[bold red]Failed to start G4F API server. Exiting...[/bold red]")
            return
            
        # Initialize LLM with local API
        global llm
        llm = initialize_llm()
    else:
        console.print("[bold orange]G4F API server not started due to missing module.[/bold orange]")
        return

    console.print("[bold cyan]Task Management Agent[/bold cyan]", justify="center")
    console.print("Type 'exit' to quit, 'add task <description>' to add tasks\n", style="dim")
    
    while True:
        try:
            for event in graph.stream(initial_state, config, stream_mode="values"):
                if "messages" in event:
                    pass  # Response already printed in handle_user_input
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    main()
