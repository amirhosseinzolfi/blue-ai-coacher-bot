import json
import logging
import threading            # added for API thread
import datetime             # added for API wait
import time                 # added for API wait
import requests             # added for API calls
import coloredlogs          # added for colorful logging
import colorama             # added for Windows terminal support
from langchain_openai import ChatOpenAI

colorama.init()             # initialize colorama

# --- Initialize colored logging ---
coloredlogs.install(level='INFO', fmt='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
    def __init__(self, task_file='tasks2.json'):  # changed default from 'tasks.json'
        self.task_file = task_file
        self.load_tasks()

    def load_tasks(self):
        try:
            with open(self.task_file, 'r') as f:
                self.tasks = json.load(f)
            logging.info("Loaded tasks from %s", self.task_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning("Failed loading tasks: %s. Using an empty tasks list.", e)
            self.tasks = []

    def save_tasks(self):
        with open(self.task_file, 'w') as f:
            json.dump(self.tasks, f, indent=4)
        logging.info("Tasks saved to %s", self.task_file)

    def add_task(self, title, description):
        task = {'title': title, 'description': description, 'status': 'pending'}
        self.tasks.append(task)
        self.save_tasks()
        logging.info("Task added: %s", title)

    def mark_task_done(self, task_title):
        for task in self.tasks:
            if task['title'] == task_title:
                task['status'] = 'done'
                self.save_tasks()
                logging.info("Task marked as done: %s", task_title)
                return True
        logging.warning("Task not found: %s", task_title)
        return False

    def refine_task(self, task_title, new_description):
        for task in self.tasks:
            if task['title'] == task_title:
                task['description'] = new_description
                self.save_tasks()
                logging.info("Task refined: %s", task_title)
                return True
        logging.warning("Task not found: %s", task_title)
        return False
        
    def summarize_tasks(self):
        # Provides a structured summary of tasks
        if not self.tasks:
            return "No tasks available."
        summary_lines = []
        for idx, task in enumerate(self.tasks, 1):
            line = f"{idx}. Title: {task['title']} | Status: {task['status']} | Desc: {task['description']}"
            summary_lines.append(line)
        summary = "\n".join(summary_lines)
        logging.info("Tasks summary:\n%s", summary)
        return summary

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
    logging.info("Welcome to the Task Manager! Type your commands to manage your tasks.")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                logging.info("Session ended by user.")
                break

            # Wrap LLM call in try/except for error handling
            try:
                response = llm.invoke(user_input)  # updated to use invoke()
            except Exception as llm_err:
                logging.error("LLM invocation error: %s", llm_err)
                response = "Error calling agent. Please try again."
            logging.info("Agent response: %s", response)
            print("Agent:", response)

            # Process commands for managing tasks
            if 'add task' in user_input.lower():
                title = extract_title_from_input(user_input)
                description = extract_description_from_input(user_input)
                task_manager.add_task(title, description)
                print(f'[INFO] Task added: {title}')
            elif 'mark task done' in user_input.lower():
                task_title = extract_title_from_input(user_input)
                if task_manager.mark_task_done(task_title):
                    print(f'[INFO] Task marked as done: {task_title}')
                else:
                    print(f'[WARN] Task not found: {task_title}')
            elif 'refine task' in user_input.lower():
                task_title = extract_title_from_input(user_input)
                new_description = extract_description_from_input(user_input)
                if task_manager.refine_task(task_title, new_description):
                    print(f'[INFO] Task refined: {task_title}')
                else:
                    print(f'[WARN] Task not found: {task_title}')
            elif 'summarize tasks' in user_input.lower():
                summary = task_manager.summarize_tasks()
                # Print summary with coloring for better readability
                print("\n" + "="*40)
                print("TASKS SUMMARY:")
                print(summary)
                print("="*40 + "\n")
    except Exception as e:
        logging.error("An unhandled error occurred: %s", e)
        print("An unexpected error occurred. Please try again.")