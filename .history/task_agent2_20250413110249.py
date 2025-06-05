import json
import logging
import threading
import datetime
import time
import requests
import coloredlogs
import colorama
import os
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langchain_openai import ChatOpenAI

colorama.init()  # initialize colorama
console = Console()

# --- Enhanced Logging Configuration ---
log_format = "%(asctime)s │ %(levelname)-8s │ %(message)s"
coloredlogs.install(
    level='INFO',
    fmt=log_format,
    datefmt='%H:%M:%S',
    level_styles={
        'debug': {'color': 'green'},
        'info': {'color': 'blue'},
        'warning': {'color': 'yellow', 'bold': True},
        'error': {'color': 'red', 'bold': True},
    },
    field_styles={
        'asctime': {'color': 'white'},
        'levelname': {'color': 'white', 'bold': True},
        'message': {'color': 'white'}
    }
)

############################################
# G4F API Server Setup
############################################
try:
    from g4f.api import run_api
except ImportError:
    logging.error("[bold red]g4f.api module not found. Install the 'g4f' package.[/bold red]")
    run_api = None

if run_api is not None:
    def start_g4f_api():
        logging.info("Starting G4F Interference API server on http://localhost:16200/v1 ...")
        run_api(bind="0.0.0.0:16200")
    api_thread = threading.Thread(target=start_g4f_api, daemon=True, name="G4F-API-Thread")
    api_thread.start()
else:
    logging.warning("G4F API server not started due to missing module.")

def wait_for_api_server(timeout=30):
    base_url = "http://localhost:16200/v1/chat/completions"
    start_time = datetime.datetime.now()
    logging.info("Waiting for the G4F API server to become available...")
    while True:
        try:
            r = requests.post(base_url, json={"messages": [{"role": "system", "content": "ping"}]}, timeout=5)
            if r.ok:
                logging.info("G4F API server responded successfully.")
                break
        except Exception:
            pass
        if (datetime.datetime.now() - start_time).seconds > timeout:
            logging.error("API server not available after waiting 30 seconds.")
            break
        time.sleep(1)

wait_for_api_server()   # wait until the API server is available

# --- Initialize LLM ---  
llm = ChatOpenAI(  
    base_url="http://localhost:16200/v1",  # updated to use local G4F API server port
    model_name="gemini-2.0-flash",  
    temperature=0.5,  
    api_key="324"  
)

class TaskManager:
    def __init__(self, task_file='tasks/task2.json'):  # updated default file path
        os.makedirs(os.path.dirname(task_file), exist_ok=True)  # ensure "tasks" folder exists
        self.task_file = task_file
        self.load_tasks()
        console.print(Panel.fit("📋 Task Manager Initialized", style="blue"))

    def load_tasks(self):
        try:
            with open(self.task_file, 'r') as f:
                self.tasks = json.load(f)
            console.print(f"[green]✓ Loaded {len(self.tasks)} tasks from {self.task_file}[/green]")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            console.print(f"[yellow]⚠ No existing tasks found: {e}. Starting fresh.[/yellow]")
            self.tasks = []

    def save_tasks(self):
        with open(self.task_file, 'w') as f:
            json.dump(self.tasks, f, indent=4)
        console.print(f"[green]✓ Saved {len(self.tasks)} tasks to {self.task_file}[/green]")

    def add_task(self, title, description):
        task = {'title': title, 'description': description, 'status': 'pending'}
        self.tasks.append(task)
        self.save_tasks()
        console.print(f"[green]✓ Task added:[/green] {title}")

    def mark_task_done(self, task_title):
        for task in self.tasks:
            if task['title'] == task_title:
                task['status'] = 'done'
                self.save_tasks()
                console.print(f"[green]✓ Marked as done:[/green] {task_title}")
                return True
        console.print(f"[yellow]⚠ Task not found:[/yellow] {task_title}")
        return False

    def refine_task(self, task_title, new_description):
        for task in self.tasks:
            if task['title'] == task_title:
                task['description'] = new_description
                self.save_tasks()
                console.print(f"[green]✓ Refined task:[/green] {task_title}")
                return True
        console.print(f"[yellow]⚠ Task not found:[/yellow] {task_title}")
        return False
        
    def summarize_tasks(self):
        if not self.tasks:
            console.print("[yellow]⚠ No tasks available.[/yellow]")
            return "No tasks available."
        
        # Create rich table for tasks
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", style="dim")
        table.add_column("Title", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Description", style="green")

        for idx, task in enumerate(self.tasks, 1):
            status_style = "green" if task['status'] == 'done' else "yellow"
            status = f"[{status_style}]{task['status']}[/{status_style}]"
            table.add_row(
                str(idx),
                task['title'],
                status,
                task['description']
            )

        console.print(Panel(table, title="[bold]Tasks Summary[/bold]", border_style="blue"))
        return table

# --- Dummy Extractor Functions ---
def extract_title_from_input(user_input):
    # Simple extractor: assume title follows 'task' keyword
    parts = user_input.split('task')
    if len(parts) > 1:
        return parts[1].strip().split()[0]
    return "Untitled"

def extract_description_from_input(user_input):
    # Simple extractor: return the remaining text after title extraction
    parts = user_input.split('task')
    if len(parts) > 1:
        return parts[1].strip()
    return "No description provided"

# --- Main Interaction Loop ---  
if __name__ == '__main__':
    task_manager = TaskManager()
    console.print(Panel.fit("🤖 Welcome to Task Manager!", style="bold blue"))
    
    try:
        while True:
            user_input = input("\nYou → ")
            if user_input.lower() in ['quit', 'exit']:
                console.print(Panel.fit("👋 Goodbye!", style="bold blue"))
                break

            # Process LLM with enhanced structured logging of response components
            console.print("\n[bold blue]Processing...[/bold blue]")
            try:
                response = llm.invoke(user_input)
                if isinstance(response, dict):
                    content       = response.get('content', '')
                    model_name    = response.get('model_name', response.get('response_metadata', {}).get('model_name', ''))
                    finish_reason = response.get('finish_reason', response.get('response_metadata', {}).get('finish_reason',''))
                    # Extract reasoning tokens if available in completion_tokens_details
                    reasoning     = response.get('response_metadata', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 'N/A')
                    resp_id       = response.get('id', response.get('response_metadata', {}).get('id', ''))
                    additional    = response.get('additional_kwargs', {})
                    usage         = response.get('usage_metadata', {})
                else:
                    content       = str(response)
                    model_name    = ""
                    finish_reason = ""
                    reasoning     = ""
                    resp_id       = ""
                    additional    = {}
                    usage         = {}

                # Assemble structured output using rich Table
                resp_table = Table(show_header=False, box=None, expand=True)
                resp_table.add_row("[bold green]Content:[/bold green]", content)
                resp_table.add_row("[bold cyan]Model Name:[/bold cyan]", str(model_name))
                resp_table.add_row("[bold magenta]Finish Reason:[/bold magenta]", str(finish_reason))
                resp_table.add_row("[bold yellow]Reasoning Tokens:[/bold yellow]", str(reasoning))
                resp_table.add_row("[bold white]Response ID:[/bold white]", str(resp_id))
                resp_table.add_row("[bold blue]Additional Args:[/bold blue]", str(additional))
                resp_table.add_row("[bold red]Usage Metadata:[/bold red]", str(usage))
                console.print(Panel(resp_table, title="[bold]AI Response[/bold]", border_style="bright_green"))
            except Exception as llm_err:
                console.print(f"[bold red]Error:[/bold red] {llm_err}")

            # Process commands
            if 'add task' in user_input.lower():
                title = extract_title_from_input(user_input)
                description = extract_description_from_input(user_input)
                task_manager.add_task(title, description)
                
            elif 'mark task done' in user_input.lower():
                task_title = extract_title_from_input(user_input)
                task_manager.mark_task_done(task_title)
                    
            elif 'refine task' in user_input.lower():
                task_title = extract_title_from_input(user_input)
                new_description = extract_description_from_input(user_input)
                task_manager.refine_task(task_title, new_description)
                    
            elif 'summarize tasks' in user_input.lower():
                task_manager.summarize_tasks()  # Rich table will be printed automatically

    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] {str(e)}")
        console.print_exception()