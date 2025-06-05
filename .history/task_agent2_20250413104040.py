import json
import logging
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize LLM ---  
llm = ChatOpenAI(  
    base_url="http://localhost:15203/v1",  
    model_name="gemini-2.0-flash",  
    temperature=0.5,  
    api_key="324"  
)

class TaskManager:
    def __init__(self, task_file='tasks.json'):
        self.task_file = task_file
        self.load_tasks()

    def load_tasks(self):
        try:
            with open(self.task_file, 'r') as f:
                self.tasks = json.load(f)
                logging.info("Loaded tasks from %s", self.task_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning("Failed loading tasks: %s. Initializing new tasks list.", e)
            self.tasks = []

    def save_tasks(self):
        with open(self.task_file, 'w') as f:
            json.dump(self.tasks, f, indent=4)
        logging.info("Tasks saved to %s", self.task_file)

    def add_task(self, title, description):
        task = {'title': title, 'description': description, 'status': 'pending'}
        self.tasks.append(task)
        self.save_tasks()

    def mark_task_done(self, task_title):
        for task in self.tasks:
            if task['title'] == task_title:
                task['status'] = 'done'
                self.save_tasks()
                return True
        return False

    def refine_task(self, task_title, new_description):
        for task in self.tasks:
            if task['title'] == task_title:
                task['description'] = new_description
                self.save_tasks()
                return True
        return False

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
                response = llm(user_input)
            except Exception as llm_err:
                logging.error("LLM invocation error: %s", llm_err)
                response = "Error calling agent. Please try again."
            logging.info("Agent response: %s", response)
            print("Agent: ", response)

            # Parse the response for commands to manage tasks
            if 'add task' in user_input.lower():
                title = extract_title_from_input(user_input)
                description = extract_description_from_input(user_input)
                task_manager.add_task(title, description)
                logging.info("Task added: %s", title)
                print(f'Task added: {title}')
            elif 'mark task done' in user_input.lower():
                task_title = extract_title_from_input(user_input)
                if task_manager.mark_task_done(task_title):
                    logging.info("Task marked as done: %s", task_title)
                    print(f'Task marked as done: {task_title}')
                else:
                    logging.warning("Task not found: %s", task_title)
                    print(f'Task not found: {task_title}')
            elif 'refine task' in user_input.lower():
                task_title = extract_title_from_input(user_input)
                new_description = extract_description_from_input(user_input)
                if task_manager.refine_task(task_title, new_description):
                    logging.info("Task refined: %s", task_title)
                    print(f'Task refined: {task_title}')
                else:
                    logging.warning("Task not found: %s", task_title)
                    print(f'Task not found: {task_title}')
    except Exception as e:
        logging.error("An unhandled error occurred: %s", e)
        print("An unexpected error occurred. Please try again.")