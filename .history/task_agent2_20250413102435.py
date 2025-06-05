import json
from langchain_openai import ChatOpenAI

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
        except (FileNotFoundError, json.JSONDecodeError):
            self.tasks = []

    def save_tasks(self):
        with open(self.task_file, 'w') as f:
            json.dump(self.tasks, f, indent=4)

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

# --- Main Interaction Loop ---  
if __name__ == '__main__':
    task_manager = TaskManager()
    print("Welcome to the Task Manager! Type your commands to manage your tasks.")
    while True:
        user_input = input("You: ")  
        # Convert input to a string format expected by the LLM
        response = llm(user_input)  
        print("Agent: ", response)

        # Parse the response for commands to manage tasks
        if 'add task' in user_input.lower():
            title = extract_title_from_input(user_input)
            description = extract_description_from_input(user_input)
            task_manager.add_task(title, description)
            print(f'Task added: {title}')
        elif 'mark task done' in user_input.lower():
            task_title = extract_title_from_input(user_input)
            if task_manager.mark_task_done(task_title):
                print(f'Task marked as done: {task_title}')
            else:
                print(f'Task not found: {task_title}')
        elif 'refine task' in user_input.lower():
            task_title = extract_title_from_input(user_input)
            new_description = extract_description_from_input(user_input)
            if task_manager.refine_task(task_title, new_description):
                print(f'Task refined: {task_title}')
            else:
                print(f'Task not found: {task_title}')