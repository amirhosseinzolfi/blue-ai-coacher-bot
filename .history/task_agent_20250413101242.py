import json
import os
from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator  # Added for Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph_checkpoint.sqlite import SqliteSaver  # Using SqliteSaver for checkpointing

# --- Configuration ---
TASKS_FILE = "tasks.json"
CHECKPOINT_DB = "checkpoints.sqlite"  # Added for checkpointer

# --- Initialize LLM ---
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"  # Replace with your actual API key or load from env
)

# --- Pydantic Models and State Types ---
class Task(BaseModel):
    id: int = Field(description="Unique identifier for the task")
    description: str = Field(description="Detailed description of the task")
    status: str = Field(description="Current status of the task (e.g., 'todo', 'done')", default="todo")

class TaskAction(BaseModel):
    action: str = Field(description="The action to perform: 'add', 'update', 'complete', 'list', 'clarify', 'respond'")
    task_id: Optional[int] = Field(None, description="The ID of the task to update or complete. Required for 'update' and 'complete'.")
    task_description: Optional[str] = Field(None, description="The description of the task. Required for 'add', potentially used for 'update'.")
    response: str = Field(description="A direct response to the user if no task action is needed, clarification is required, or confirming an action.")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # Manages history
    tasks: List[Task]
    next_action: str  # What the LLM decided the next step should be
    response: str  # The final response to the user
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
            if not isinstance(tasks_data, list):
                print(f"Warning: '{TASKS_FILE}' does not contain a list. Starting fresh.")
                return []
            valid_tasks = []
            for i, task_data in enumerate(tasks_data):
                if isinstance(task_data, dict) and 'id' in task_data and 'description' in task_data:
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
    except Exception as e:
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
Analyze the latest user message in the context of the conversation history and the current list of tasks.
Determine the appropriate action and extract necessary information.

Current Tasks:
{tasks}

Conversation History (most recent messages first):
{history}

Latest User Request: {user_input}

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
    current_tasks = load_tasks()
    state['tasks'] = current_tasks

    messages = state['messages']
    if not messages or not isinstance(messages[-1], HumanMessage):
        print("Error: Last message is not a HumanMessage.")
        state['next_action'] = 'error'
        state['response'] = "Sorry, there was an issue processing your request."
        return state

    user_input = messages[-1].content
    history_messages = messages[:-1]
    formatted_history = "\n".join(
        [f"{type(msg).__name__}: {msg.content}" for msg in history_messages[-5:]]
    )
    formatted_tasks = json.dumps([task.dict() for task in current_tasks], indent=2) if current_tasks else "No tasks yet."

    prompt = system_prompt.format(
        tasks=formatted_tasks,
        history=formatted_history,
        user_input=user_input
    )

    llm_messages = [
        SystemMessage(content=prompt),
    ]

    llm_with_tool = llm.with_structured_output(TaskAction)

    try:
        ai_response: TaskAction = llm_with_tool.invoke(llm_messages)
        print(f"LLM Analysis Result: action='{ai_response.action}', id={ai_response.task_id}, desc='{ai_response.task_description}', response='{ai_response.response}'")

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

        state['next_action'] = action
        state['response'] = ai_response.response
        state['extracted_task_id'] = ai_response.task_id
        state['extracted_task_description'] = ai_response.task_description

    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        state['next_action'] = 'error'
        state['response'] = "Sorry, I encountered an error trying to understand your request. Please try again."
        state['extracted_task_id'] = None
        state['extracted_task_description'] = None

    return state

# --- Task Manipulation Nodes ---
def add_task(state: AgentState):
    print("--- ADDING TASK ---")
    tasks = load_tasks()
    description = state.get('extracted_task_description')
    response = state.get('response', "Okay, I've added the task.")

    if not description:
        state['response'] = "Sorry, something went wrong. I didn't get the task description."
        state['next_action'] = 'error'
        return state

    next_id = max([task.id for task in tasks] + [0]) + 1
    new_task = Task(id=next_id, description=description, status="todo")
    tasks.append(new_task)

    state['tasks'] = tasks
    state['response'] = response
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    print(f"Task added: {new_task}")
    return state

def update_task(state: AgentState):
    print("--- UPDATING TASK ---")
    tasks = load_tasks()
    task_id = state.get('extracted_task_id')
    new_description = state.get('extracted_task_description')
    response = state.get('response', f"Okay, I've updated task {task_id}.")

    if task_id is None or new_description is None:
        state['response'] = "Sorry, something went wrong. I need both the task ID and the new description to update."
        state['next_action'] = 'error'
        return state

    task_found = False
    for task in tasks:
        if task.id == task_id:
            task.description = new_description
            task_found = True
            state['response'] = response
            print(f"Task {task_id} updated to: {new_description}")
            break

    if not task_found:
        state['response'] = f"Sorry, I couldn't find task ID {task_id} to update."
        state['next_action'] = 'clarify'

    state['tasks'] = tasks
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    return state

def complete_task(state: AgentState):
    print("--- COMPLETING TASK ---")
    tasks = load_tasks()
    task_id = state.get('extracted_task_id')
    response = state.get('response', f"Okay, I've marked task {task_id} as done.")

    if task_id is None:
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
            state['response'] = response
            break

    if not task_found:
        state['response'] = f"Sorry, I couldn't find task ID {task_id} to mark as complete."
        state['next_action'] = 'clarify'

    state['tasks'] = tasks
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    return state

def list_tasks(state: AgentState):
    print("--- LISTING TASKS ---")
    tasks = load_tasks()
    response = state.get('response')

    if not response:
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

    state['tasks'] = tasks
    state['extracted_task_id'] = None
    state['extracted_task_description'] = None
    return state

def prepare_response(state: AgentState):
    print("--- PREPARING RESPONSE ---")
    response_content = state.get('response', "I'm not sure how to respond to that.")
    state['response'] = response_content

    ai_message = AIMessage(content=response_content)
    state['messages'] = state.get('messages', []) + [ai_message]

    print(f"Final response: {response_content}")
    save_tasks(state.get('tasks', []))
    return state

print("Task Agent Nodes Defined.")

# --- Build Graph ---
print("Building graph...")
workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_request)
workflow.add_node("add", add_task)
workflow.add_node("update", update_task)
workflow.add_node("complete", complete_task)
workflow.add_node("list", list_tasks)
workflow.add_node("finalize", prepare_response)

workflow.set_entry_point("analyze")

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
        "clarify": "finalize",
        "respond": "finalize",
        "error": "finalize",
    }
)

workflow.add_edge("add", "finalize")
workflow.add_edge("update", "finalize")
workflow.add_edge("complete", "finalize")
workflow.add_edge("list", "finalize")
workflow.add_edge("finalize", END)

# --- Compile and Run ---

# Use a 'with' block as SqliteSaver.from_conn_string returns a context manager
with SqliteSaver.from_conn_string(CHECKPOINT_DB) as memory:
    print(f"Initialized SQLite checkpointer ({CHECKPOINT_DB}).")
    app = workflow.compile(checkpointer=memory)
    print("Graph built and compiled.")

    # --- Main Interaction Loop (now inside the 'with' block) ---
    if __name__ == "__main__":
        print("\nWelcome to the Task Management Agent!")
        print("Type 'quit' or 'exit' to stop.")

        thread_id = input("Enter a unique ID for this conversation (e.g., user123): ")
        config = {"configurable": {"thread_id": thread_id}}
        print(f"Using thread ID: {thread_id}")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Assistant: Goodbye!")
                break

            if not user_input.strip():
                continue

            inputs = {"messages": [HumanMessage(content=user_input)]}

            try:
                # Use the 'app' compiled within the 'with' block
                final_state_stream = app.stream(inputs, config=config)
                assistant_response = "Processing..."
                final_state_value = None

                for event in final_state_stream:
                    # print(f"DEBUG Event: {event}") # Optional debug
                    # Check if the event dictionary contains the END key
                    if isinstance(event, dict) and END in event:
                        final_state_value = event[END]


                if final_state_value and final_state_value.get('messages'):
                    last_message = final_state_value['messages'][-1]
                    if isinstance(last_message, AIMessage):
                        assistant_response = last_message.content
                    else:
                        # Fallback if the last message isn't AIMessage but response exists
                        assistant_response = final_state_value.get('response', "Sorry, I didn't get a valid response.")
                elif final_state_value:
                     # Fallback if messages list is missing or empty but state exists
                     assistant_response = final_state_value.get('response', "Sorry, something went wrong during processing.")
                else:
                    # Fallback if no final state value was captured
                    assistant_response = "Sorry, the process didn't complete as expected."


                print(f"Assistant: {assistant_response}")

            except Exception as e:
                print(f"\nAn error occurred during processing: {e}")
                import traceback
                traceback.print_exc()
                print("Please try again.")

        print("\nTask agent session ended.")

# Ensure the script exits cleanly if not run as main (though it likely always is)
# This part is outside the 'with' block