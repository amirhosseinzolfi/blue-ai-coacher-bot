import json
import os
from typing import List, TypedDict, Annotated, Sequence
from operator import itemgetter
import sys

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import threading
import time
import requests
import datetime
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- G4F API Server Setup ---
G4F_PORT = 16200
G4F_BASE_URL = f"http://localhost:{G4F_PORT}/v1"

try:
    from g4f.api import run_api
except ImportError:
    logger.error("g4f.api module not found. Please install with: pip install g4f")
    run_api = None

def start_g4f_api_server_thread():
    """Target function to run the g4f API server."""
    if run_api:
        try:
            logger.info(f"Starting G4F API server on 0.0.0.0:{G4F_PORT}...")
            # Note: run_api might block, hence it runs in a thread.
            # Adjust bind address if needed, 0.0.0.0 makes it accessible externally within the container/network
            run_api(bind=f"0.0.0.0:{G4F_PORT}")
        except Exception as e:
            logger.error(f"Failed to start G4F API server: {e}")
    else:
        logger.warning("G4F API server not started due to missing 'g4f' module.")

def wait_for_api_server(url, timeout=45):
    """Waits for the API server to become available."""
    start_time = datetime.datetime.now()
    logger.info(f"Waiting for API server at {url} (timeout: {timeout}s)...")
    check_url = f"{url}/models" # Use the /models endpoint for a simple check
    while True:
        try:
            # Use a short timeout for each check attempt
            response = requests.get(check_url, timeout=2)
            if response.ok:
                logger.info("API server is available.")
                return True
            else:
                logger.warning(f"API server check failed with status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"API server not yet available at {url}...")
        except requests.exceptions.Timeout:
             logger.debug(f"API server check timed out at {url}...")
        except Exception as e:
            logger.error(f"Error checking API server: {e}") # Log other potential errors

        if (datetime.datetime.now() - start_time).seconds > timeout:
            logger.error(f"API server did not become available after {timeout} seconds.")
            return False
        time.sleep(2) # Wait before retrying

# --- Constants ---
TASKS_FILE = "tasks.json"

# --- Initialize LLM ---
# Now points to the local G4F server started by this script.
logger.info(f"Configuring LLM to use base URL: {G4F_BASE_URL}")
llm = ChatOpenAI(
    base_url=G4F_BASE_URL,
    model_name="gemma-7b-it", # Default or specify a model supported by your G4F setup
    temperature=0.5,
    api_key="nokey" # API key often not needed or dummy for local servers
)

# --- Task Structure ---
class Task(TypedDict):
    id: int
    description: str
    status: str # e.g., "pending", "done"

# --- Agent State ---
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    tasks: List[Task]
    next_task_id: int

# --- Task Persistence ---
def load_tasks() -> List[Task]:
    """Loads tasks from the JSON file."""
    if not os.path.exists(TASKS_FILE):
        return []
    try:
        with open(TASKS_FILE, 'r') as f:
            tasks_data = json.load(f)
            # Ensure tasks have the required fields, default if necessary
            validated_tasks = []
            for task in tasks_data:
                validated_tasks.append({
                    "id": task.get("id", -1), # Assign default if missing, handle later
                    "description": task.get("description", "No description"),
                    "status": task.get("status", "pending")
                })
            # Filter out tasks with invalid IDs from initial load if any
            validated_tasks = [t for t in validated_tasks if t["id"] != -1]
            return validated_tasks
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Warning: Could not load or parse {TASKS_FILE}. Starting with empty task list.")
        return []

def save_tasks(tasks: List[Task]):
    """Saves tasks to the JSON file."""
    with open(TASKS_FILE, 'w') as f:
        json.dump(tasks, f, indent=4)

def get_next_task_id(tasks: List[Task]) -> int:
    """Calculates the next available task ID."""
    if not tasks:
        return 1
    # Ensure all task IDs are integers before finding the max
    valid_ids = [task['id'] for task in tasks if isinstance(task.get('id'), int)]
    if not valid_ids:
        return 1
    return max(valid_ids) + 1

# --- Initialize tasks.json if it doesn't exist or is empty ---
if not os.path.exists(TASKS_FILE) or os.path.getsize(TASKS_FILE) == 0:
     print(f"Initializing {TASKS_FILE}...")
     save_tasks([])

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# --- LLM Instruction Model ---
class TaskUpdateInstruction(BaseModel):
    """Instructions for updating the task list based on user request."""
    operation: str = Field(description="The operation to perform: 'add', 'update', 'done', 'refine', or 'none'.")
    task_id: int | None = Field(description="The ID of the task to update/mark done/refine. Required for 'update', 'done', 'refine'.", default=None)
    description: str | None = Field(description="The new description for 'add' or 'update'/'refine' operations.", default=None)
    response_message: str = Field(description="A message to show the user confirming the action or asking for clarification.")

# --- Graph Nodes ---

def call_llm_task_analyzer(state: AgentState):
    """Analyzes conversation and tasks to determine the next action."""
    print("--- Calling LLM Task Analyzer ---")
    messages = state['messages']
    tasks = state['tasks']

    # Prepare the prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a task management assistant. Your goal is to analyze the user's latest message in the context of the conversation history and the current task list.
Current Tasks:
{tasks_json}

Based on the latest user message, determine if they want to 'add' a new task, 'update' an existing task's description, mark a task as 'done', 'refine' (clarify/modify) an existing task, or if it's just conversation ('none').
Respond ONLY with the JSON structure defined by the TaskUpdateInstruction tool.
If adding, provide a description.
If updating, done, or refining, provide the task_id and relevant details (new description for update/refine).
If no task action is needed, set operation to 'none'.
Always provide a user-facing 'response_message' confirming the action or continuing the conversation."""),
        ("human", "Conversation History:\n{history}\n\nLatest User Message: {input}")
    ])

    # Format history
    history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[:-1]]) # Exclude the latest user message
    latest_input = messages[-1].content if messages else ""
    tasks_json_str = json.dumps(tasks, indent=2) if tasks else "[]"

    # Chain with structured output
    structured_llm = llm.with_structured_output(TaskUpdateInstruction)
    chain = prompt_template | structured_llm

    try:
        instruction: TaskUpdateInstruction = chain.invoke({
            "tasks_json": tasks_json_str,
            "history": history,
            "input": latest_input
        })
        print(f"LLM Instruction: {instruction}")
        # Add the AI's intended action/response as a message for context
        # We add the *parsed* instruction, not the raw LLM response
        ai_message_content = f"Instruction: {instruction.operation}"
        if instruction.task_id:
             ai_message_content += f", Task ID: {instruction.task_id}"
        if instruction.description:
             ai_message_content += f", Desc: {instruction.description}"
        # Also include the user-facing message
        ai_message_content += f"\nResponse: {instruction.response_message}"

        return {"messages": [AIMessage(content=ai_message_content)], "task_update_instruction": instruction}
    except Exception as e:
        print(f"Error calling LLM or parsing response: {e}")
        # Provide a fallback response
        fallback_instruction = TaskUpdateInstruction(
            operation="none",
            response_message="Sorry, I encountered an error processing your request. Can you please rephrase?"
        )
        return {"messages": [AIMessage(content=fallback_instruction.response_message)], "task_update_instruction": fallback_instruction}


def update_tasks_node(state: AgentState):
    """Updates the tasks list based on the LLM instruction."""
    print("--- Updating Tasks ---")
    instruction: TaskUpdateInstruction = state.get("task_update_instruction")
    tasks = list(state['tasks']) # Make a mutable copy
    next_task_id = state['next_task_id']
    updated = False

    if not instruction or instruction.operation == 'none':
        print("No task operation requested.")
        # The response message is already in the messages list from the previous node
        return {"tasks": tasks, "next_task_id": next_task_id}

    op = instruction.operation
    task_id = instruction.task_id
    description = instruction.description

    if op == 'add' and description:
        new_task = {"id": next_task_id, "description": description, "status": "pending"}
        tasks.append(new_task)
        next_task_id += 1
        updated = True
        print(f"Added Task: {new_task}")
    elif op in ['update', 'refine'] and task_id is not None and description:
        for task in tasks:
            if task['id'] == task_id:
                task['description'] = description
                task['status'] = "pending" # Ensure status is pending after update/refine
                updated = True
                print(f"Updated Task {task_id}: {task}")
                break
    elif op == 'done' and task_id is not None:
        for task in tasks:
            if task['id'] == task_id:
                task['status'] = 'done'
                updated = True
                print(f"Marked Task {task_id} as Done: {task}")
                break

    if updated:
        save_tasks(tasks)
        print("Tasks saved.")

    # The response message is already added in the call_llm node
    return {"tasks": tasks, "next_task_id": next_task_id}


# --- Define Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("analyze", call_llm_task_analyzer)
workflow.add_node("update", update_tasks_node)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "update")
workflow.add_edge("update", END)

# Use memory saver for checkpointing conversation state
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

print("Agent graph compiled.")

# --- Main Interaction Loop ---
if __name__ == "__main__":
    logger.info("--- Initializing Task Management Agent ---")

    # Start G4F API Server in a background thread
    if run_api:
        api_thread = threading.Thread(target=start_g4f_api_server_thread, daemon=True, name="G4F-API-Thread")
        api_thread.start()

        # Wait for the server to be ready
        if not wait_for_api_server(G4F_BASE_URL):
            logger.error("Exiting: G4F API server failed to start.")
            sys.exit(1) # Exit if server isn't available
    else:
        logger.error("Exiting: G4F module not found. Cannot start API server.")
        sys.exit(1) # Exit if g4f is not installed

    # --- Agent Setup (proceed only if API server is ready) ---
    logger.info("Compiling LangGraph application...")
    # Use memory saver for checkpointing conversation state
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    logger.info("Agent graph compiled.")

    print("\n--- Starting Task Management Agent Interaction ---")
    print(f"Using G4F API server at {G4F_BASE_URL}")
    print("Enter 'quit' to exit.")

    # Unique conversation ID (can be made more robust)
    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Load initial state for this conversation thread
    initial_tasks = load_tasks()
    initial_next_id = get_next_task_id(initial_tasks)
    logger.info(f"Loaded {len(initial_tasks)} tasks. Next ID: {initial_next_id}")

    # Prime the state if needed (optional, depends if you want initial tasks in first graph run)
    # You might want to ensure the initial state is correctly set in the memory checkpointer
    # Check if state exists, if not, put the initial state
    # if not memory.get(config):
    #    logger.info("Priming initial state in memory.")
    #    memory.put(config, {"messages": [], "tasks": initial_tasks, "next_task_id": initial_next_id})


    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError: # Handle Ctrl+D or unexpected end of input
             print("\nExiting due to EOF.")
             break
        if user_input.lower() == 'quit':
            break

        # Get current state for the thread
        # It's generally better to pass the necessary parts of the state explicitly
        # rather than fetching and re-injecting the whole state object.
        # However, the current structure relies on the checkpointer managing this.
        # Let's fetch the latest known tasks/next_id to pass into the stream,
        # as the stream input merges with the checkpointed state.
        current_checkpoint = app.get_state(config)
        # Use loaded tasks/id as fallback if no checkpoint exists yet
        tasks_for_input = current_checkpoint.values.get("tasks", initial_tasks) if current_checkpoint else initial_tasks
        next_id_for_input = current_checkpoint.values.get("next_task_id", initial_next_id) if current_checkpoint else initial_next_id

        # Append user message and run the graph
        # The input here should ideally just be the new message,
        # LangGraph + checkpointer handles merging with the existing state.
        # inputs = {"messages": [HumanMessage(content=user_input)]}
        # Let's stick to the previous explicit passing for now, though it might be redundant with MemorySaver
        inputs = {"messages": [HumanMessage(content=user_input)], "tasks": tasks_for_input, "next_task_id": next_id_for_input}


        final_state_update = None
        try:
            logger.info("Streaming graph execution...")
            # The graph execution updates the state via the checkpointer
            for event in app.stream(inputs, config=config, stream_mode="values"):
                 # Keep the last state update which contains the final response message
                 logger.debug(f"Graph Event: {event}") # Log events for debugging
                 final_state_update = event
            logger.info("Graph execution finished.")

        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            print("\nAgent: Sorry, an internal error occurred while processing your request.")
            # Optionally display current tasks even after error
            # continue # Skip displaying response/tasks if error is severe

        # Extract the latest AI response message to display
        if final_state_update and "messages" in final_state_update and final_state_update["messages"]:
             # Get the last message, assuming it's the AI's response
             ai_response_message = final_state_update["messages"][-1]
             if isinstance(ai_response_message, AIMessage):
                 ai_response_content = ai_response_message.content
                 # Extract just the user-facing part from the structured content
                 response_prefix = "Response: "
                 if response_prefix in ai_response_content:
                     print(f"\nAgent: {ai_response_content.split(response_prefix, 1)[1]}")
                 else:
                      # Fallback if the format isn't as expected
                      logger.warning("AI response format unexpected, printing full content.")
                      print(f"\nAgent: {ai_response_content}")
             else:
                 logger.warning(f"Last message was not an AIMessage: {type(ai_response_message)}")
                 print("\nAgent: (Received non-AI message)")

        else:
             # This might happen if the graph ends without producing a new message in the final step
             logger.warning("No new message found in final state update.")
             print("\nAgent: (No response generated or state update issue)")

        # Display current tasks after update
        updated_state = app.get_state(config) # Get the very latest state after run
        print("\nCurrent Tasks:")
        if updated_state and updated_state.values.get("tasks"):
            tasks_to_show = updated_state.values["tasks"]
            if tasks_to_show:
                for task in tasks_to_show:
                    # Ensure task is a dict before accessing keys
                    if isinstance(task, dict):
                         print(f"  - ID: {task.get('id', 'N/A')}, Desc: {task.get('description', 'N/A')}, Status: {task.get('status', 'N/A')}")
                    else:
                         logger.warning(f"Found non-dict item in tasks list: {task}")
                         print(f"  - Invalid task entry: {task}")

            else:
                print("  (No tasks)")
        else:
             # If state is somehow lost, reload from file as fallback display
             logger.warning("Could not retrieve updated state from checkpointer, reloading from file.")
             print("(Reloaded from file):")
             reloaded_tasks = load_tasks()
             if reloaded_tasks:
                  for task in reloaded_tasks:
                       if isinstance(task, dict):
                            print(f"  - ID: {task.get('id', 'N/A')}, Desc: {task.get('description', 'N/A')}, Status: {task.get('status', 'N/A')}")
                       else:
                            print(f"  - Invalid task entry: {task}")

             else:
                  print("  (No tasks)")


    print("\n--- Exiting Agent ---")