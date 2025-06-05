import json
import os
from typing import TypedDict, List, Dict, Any, Annotated, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver  # Updated import path

# --- Configuration ---
TASKS_FILE = "tasks.json"

# --- Initialize LLM ---
# NOTE: Replace with your actual endpoint and API key if needed
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324" # Replace with your actual API key or load from env
)

# --- Agent State ---
class Task(BaseModel):
    id: int = Field(description="Unique identifier for the task")
    description: str = Field(description="Detailed description of the task")
    status: str = Field(description="Current status of the task (e.g., 'todo', 'done')", default="todo")

class AgentState(TypedDict):
    conversation_history: List[Dict[str, str]] # List of {"role": "user/assistant", "content": "message"}
    tasks: List[Task]
    user_input: str
    next_action: str # What the LLM decided the next step should be ('add', 'update', 'complete', 'refine', 'respond')
    response: str # The final response to the user

# --- Pydantic Models ---
class Task(BaseModel):
    id: int = Field(description="Unique identifier for the task")
    description: str = Field(description="Detailed description of the task")
    status: str = Field(description="Current status of the task (e.g., 'todo', 'done')", default="todo")

class TaskAction(BaseModel):
    action: str = Field(description="The action to perform: 'add', 'update', 'complete', 'list', 'clarify', 'respond'")
    task_id: Optional[int] = Field(None, description="The ID of the task to update or complete. Required for 'update' and 'complete'.")
    task_description: Optional[str] = Field(None, description="The description of the task. Required for 'add', potentially used for 'update'.")
    response: str = Field(description="A direct response to the user if no task action is needed, clarification is required, or confirming an action.")


# --- Agent State ---
class AgentState(TypedDict):
    conversation_history: List[Dict[str, str]] # List of {"role": "user/assistant", "content": "message"}
    tasks: List[Task]
    user_input: str
    next_action: str # What the LLM decided the next step should be ('add', 'update', 'complete', 'list', 'clarify', 'respond', 'error')
    response: str # The final response to the user
    # Fields to hold data extracted by LLM for other nodes
    extracted_task_id: Optional[int]
    extracted_task_description: Optional[str]


# --- Task Loading/Saving ---
def load_tasks() -> List[Task]:
    if not os.path.exists(TASKS_FILE):
        print(f"'{TASKS_FILE}' not found. Starting with empty task list.")
        return []
    try:
        with open(TASKS_FILE, 'r') as f:
            tasks_data = json.load(f)
            # Basic validation: check if it's a list
            if not isinstance(tasks_data, list):
                print(f"Warning: '{TASKS_FILE}' does not contain a list. Starting fresh.")
                return []
            # Validate each item - simple check for now
            valid_tasks = []
            for i, task_data in enumerate(tasks_data):
                 if isinstance(task_data, dict) and 'id' in task_data and 'description' in task_data:
                     # Ensure status exists, default if not
                     if 'status' not in task_data:
                         task_data['status'] = 'todo'
                     valid_tasks.append(Task(**task_data))
                 else:
                    print(f"Warning: Skipping invalid task data at index {i} in '{TASKS_FILE}'.")
            return valid_tasks
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from '{TASKS_FILE}'. Starting with empty task list.")
        return []
    except FileNotFoundError:
        print(f"'{TASKS_FILE}' not found. Starting with empty task list.")
        return []
    except Exception as e: # Catch other potential errors like invalid Pydantic parsing
        print(f"Warning: Error loading tasks from '{TASKS_FILE}': {e}. Starting with empty task list.")
        return []

def save_tasks(tasks: List[Task]):
    try:
        with open(TASKS_FILE, 'w') as f:
            json.dump([task.dict() for task in tasks], f, indent=4)
        print(f"Tasks saved to {TASKS_FILE}")
    except IOError as e:
        print(f"Error saving tasks to {TASKS_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving tasks: {e}")


# --- LLM Analysis Node ---
system_prompt = """You are a task management assistant. Your goal is to help the user manage their tasks based on the conversation.
Analyze the latest user input in the context of the conversation history and the current list of tasks.
Determine the appropriate action and extract necessary information.

Current Tasks:
{tasks}

Conversation History (most recent messages first):
{history}

User Request: {user_input}

Based *only* on the user's latest request and the provided context, decide the single most appropriate action:
- 'add': User wants to add a new task. Extract the task description. Provide a confirmation response.
- 'update': User wants to modify an existing task (refine description). Extract the task ID and the new description. Provide a confirmation response.
- 'complete': User wants to mark a task as done. Extract the task ID. Provide a confirmation response.
- 'list': User asks to see their tasks. Generate a response listing the tasks.
- 'clarify': The user's request is ambiguous, lacks necessary information (e.g., task ID for update/complete), or refers to a non-existent task ID. Ask clarifying questions in the response field.
- 'respond': For general conversation, greetings, or if no specific task action is identified. Generate a suitable response.

**Important:**
- If adding, ensure 'task_description' is populated.
- If updating, ensure 'task_id' and 'task_description' are populated.
- If completing, ensure 'task_id' is populated.
- If listing, generate the list in the 'response' field.
- If clarifying, explain what's needed in the 'response' field.
- If responding, generate the conversational reply in the 'response' field.
- Only select ONE action. If unsure, choose 'clarify' or 'respond'.
- If the user mentions a task ID that doesn't exist for update/complete, choose 'clarify'.

Respond using the TaskAction tool.
"""

def analyze_request(state: AgentState):
    print("\n--- ANALYZING REQUEST ---")
    history = state.get('conversation_history', [])
    user_input = state['user_input']
    tasks = state.get('tasks', [])

    # Format history (simple last 5 messages for context) and tasks
    recent_history = history[-5:]
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    formatted_tasks = json.dumps([task.dict() for task in tasks], indent=2) if tasks else "No tasks yet."

    prompt = system_prompt.format(
        tasks=formatted_tasks,
        history=formatted_history,
        user_input=user_input
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."), # Keep it simple
        HumanMessage(content=prompt) # Put the detailed instructions here
    ]

    # Bind the tool to the LLM
    llm_with_tool = llm.with_structured_output(TaskAction)

    try:
        # print(f"DEBUG: Sending prompt to LLM:\n{prompt}") # Optional debug
        ai_response: TaskAction = llm_with_tool.invoke(messages)
        print(f"LLM Analysis Result: action='{ai_response.action}', id={ai_response.task_id}, desc='{ai_response.task_description}', response='{ai_response.response}'")

        # Validate response consistency (basic)
        action = ai_response.action
        if action == 'add' and not ai_response.task_description:
            action = 'clarify'
            ai_response.response = "You asked to add a task, but didn't provide a description. What is the task?"
            print("WARN: LLM chose 'add' but provided no description. Changing action to 'clarify'.")
        elif action == 'update' and (not ai_response.task_id or not ai_response.task_description):
            action = 'clarify'
            ai_response.response = "You asked to update a task, but didn't specify the task ID and/or the new description. Please provide both."
            print("WARN: LLM chose 'update' but missing ID or description. Changing action to 'clarify'.")
        elif action == 'complete' and not ai_response.task_id:
            action = 'clarify'
            ai_response.response = "You asked to complete a task, but didn't specify the task ID. Which task should I mark as done?"
            print("WARN: LLM chose 'complete' but provided no ID. Changing action to 'clarify'.")

        # Update state
        state['next_action'] = action
        state['response'] = ai_response.response # Store LLM response/confirmation
        state['extracted_task_id'] = ai_response.task_id
        state['extracted_task_description'] = ai_response.task_description

    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        state['next_action'] = 'error'
        state['response'] = "Sorry, I encountered an error trying to understand your request. Please try again."
        state['extracted_task_id'] = None
        state['extracted_task_description'] = None

    # Add user input to history *before* returning
    if 'conversation_history' not in state or state['conversation_history'] is None:
        state['conversation_history'] = []
    state['conversation_history'].append({"role": "user", "content": user_input})

    # Add assistant response (even if just analysis) to history for context?
    # Let's add the final response later, after actions are performed.

    return state


# --- Task Manipulation Nodes ---

def add_task(state: AgentState):
    print("--- ADDING TASK ---")
    tasks = state.get('tasks', [])
    description = state.get('extracted_task_description')
    response = state.get('response', "Okay, I've added the task.") # Default confirmation

    if not description:
        print("Error: No description provided for add_task node.")
        state['response'] = "Sorry, something went wrong. I didn't get the task description."
        state['next_action'] = 'error' # Or maybe 'clarify'? Error seems safer.
        return state

    # Find the next available ID
    next_id = max([task.id for task in tasks] + [0]) + 1
    new_task = Task(id=next_id, description=description, status="todo")
    tasks.append(new_task)

    state['tasks'] = tasks
    state['response'] = response # Use LLM confirmation if provided
    # Clear extracted fields after use
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    print(f"Task added: {new_task}")
    return state

def update_task(state: AgentState):
    print("--- UPDATING TASK ---")
    tasks = state.get('tasks', [])
    task_id = state.get('extracted_task_id')
    new_description = state.get('extracted_task_description')
    response = state.get('response', f"Okay, I've updated task {task_id}.") # Default confirmation

    if task_id is None or new_description is None:
        print("Error: Missing task_id or new_description for update_task node.")
        state['response'] = "Sorry, something went wrong. I need both the task ID and the new description to update."
        state['next_action'] = 'error'
        return state

    task_found = False
    for task in tasks:
        if task.id == task_id:
            task.description = new_description
            task_found = True
            state['response'] = response # Use LLM confirmation
            print(f"Task {task_id} updated to: {new_description}")
            break

    if not task_found:
        print(f"Error: Task ID {task_id} not found for update.")
        # LLM should have caught this, but double-check
        state['response'] = f"Sorry, I couldn't find task ID {task_id} to update."
        state['next_action'] = 'clarify' # Ask again

    state['tasks'] = tasks
    # Clear extracted fields after use
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    return state

def complete_task(state: AgentState):
    print("--- COMPLETING TASK ---")
    tasks = state.get('tasks', [])
    task_id = state.get('extracted_task_id')
    response = state.get('response', f"Okay, I've marked task {task_id} as done.") # Default confirmation

    if task_id is None:
        print("Error: Missing task_id for complete_task node.")
        state['response'] = "Sorry, something went wrong. I need the task ID to mark it as complete."
        state['next_action'] = 'error'
        return state

    task_found = False
    for task in tasks:
        if task.id == task_id:
            if task.status == 'done':
                 response = f"Task {task_id} was already marked as done."
                 print(f"Task {task_id} already done.")
            else:
                task.status = 'done'
                print(f"Task {task_id} marked as done.")
            task_found = True
            state['response'] = response # Use LLM confirmation or 'already done' message
            break

    if not task_found:
        print(f"Error: Task ID {task_id} not found for completion.")
        state['response'] = f"Sorry, I couldn't find task ID {task_id} to mark as complete."
        state['next_action'] = 'clarify' # Ask again

    state['tasks'] = tasks
    # Clear extracted fields after use
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    return state

def list_tasks(state: AgentState):
    print("--- LISTING TASKS ---")
    tasks = state.get('tasks', [])
    response = state.get('response') # Use LLM generated list if available

    if not response: # Fallback if LLM didn't generate the list
        if not tasks:
            response = "You have no tasks."
        else:
            response = "Here are your tasks:\n"
            for task in tasks:
                status_marker = "[X]" if task.status == 'done' else "[ ]"
                response += f"{status_marker} ID {task.id}: {task.description}\n"
        state['response'] = response.strip()
        print("Generated task list as LLM didn't provide one.")
    else:
        print("Using LLM-generated task list response.")

    # Clear extracted fields (should be None anyway for list)
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    return state

def prepare_response(state: AgentState):
    """Final node to prepare the response and add it to history."""
    print("--- PREPARING RESPONSE ---")
    response = state.get('response', "I'm not sure how to respond to that.") # Fallback
    state['response'] = response # Ensure it's set

    # Add assistant response to history
    if 'conversation_history' not in state or state['conversation_history'] is None:
        state['conversation_history'] = [] # Should exist, but safety check
    state['conversation_history'].append({"role": "assistant", "content": response})

    print(f"Final response: {response}")
    # Save tasks after potential modifications
    save_tasks(state.get('tasks', []))
    return state


print("Task Agent Nodes Defined.")

# --- Build Graph ---
print("Building graph...")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze", analyze_request)
workflow.add_node("add", add_task)
workflow.add_node("update", update_task)
workflow.add_node("complete", complete_task)
workflow.add_node("list", list_tasks)
workflow.add_node("finalize", prepare_response) # Node that prepares final response and saves

# Set entry point
workflow.set_entry_point("analyze")

# Define conditional edges from analysis
def decide_next_node(state: AgentState):
    print(f"--- DECISION: Next Action is '{state['next_action']}' ---")
    return state['next_action']

workflow.add_conditional_edges(
    "analyze",
    decide_next_node,
    {
        "add": "add",
        "update": "update",
        "complete": "complete",
        "list": "list",
        "clarify": "finalize", # Go directly to response if clarifying
        "respond": "finalize", # Go directly to response for general chat
        "error": "finalize",   # Go directly to response on error
    }
)

# Define edges from action nodes to final response node
workflow.add_edge("add", "finalize")
workflow.add_edge("update", "finalize")
workflow.add_edge("complete", "finalize")
workflow.add_edge("list", "finalize")

# Final node leads to end
workflow.add_edge("finalize", END)

# Compile the graph
# Using SqliteSaver for persistence (optional but good practice)
# memory = SqliteSaver.from_conn_string(":memory:") # In-memory persistence
memory = SqliteSaver.from_conn_string("tasks_agent_memory.sqlite") # File-based persistence

app = workflow.compile(checkpointer=memory)
print("Graph built and compiled.")

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("\nWelcome to the Task Management Agent!")
    print("Type 'quit' or 'exit' to stop.")

    # Unique ID for the conversation thread
    # Using a fixed ID for simplicity in this example.
    # For multiple users/sessions, generate unique IDs (e.g., using uuid)
    config = {"configurable": {"thread_id": "user-task-session-1"}}

    # Load initial state (tasks)
    current_tasks = load_tasks()
    current_history = [] # History will be managed by the checkpointer

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Assistant: Goodbye!")
            break

        if not user_input.strip():
            continue

        # Prepare input for the graph
        inputs = {
            "user_input": user_input,
            "tasks": current_tasks, # Pass the current task list
            # History is managed by the checkpointer based on thread_id
            # "conversation_history": current_history # Don't need to pass explicitly if using checkpointer correctly
        }

        try:
            # Invoke the graph
            # The checkpointer automatically loads/saves history for the config['configurable']['thread_id']
            final_state = app.invoke(inputs, config=config)

            # Extract response and update state for the next loop iteration
            assistant_response = final_state.get('response', "Sorry, I didn't get a response.")
            current_tasks = final_state.get('tasks', []) # Update tasks from the final state
            # current_history = final_state.get('conversation_history', []) # History updated via checkpointer

            print(f"Assistant: {assistant_response}")

        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
            # Optionally, decide if you want to reset state or try to recover
            print("Please try again.")

    print("\nTask agent session ended.")