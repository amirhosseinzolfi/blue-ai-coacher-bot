from __future__ import annotations
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
    model_name="gemini-2.0-flash", # Changed back as requested
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
    task_update_instruction: TaskUpdateInstruction | None

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
    logger.info("--- Calling LLM Task Analyzer ---")
    messages = state['messages']
    tasks = state['tasks']
    instruction = None # Initialize instruction

    # Prepare the prompt - More explicit JSON instruction
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a task management assistant. Your goal is to analyze the user's latest message in the context of the conversation history and the current task list.
Current Tasks:
{tasks_json}

Based on the latest user message, determine if they want to 'add' a new task, 'update' an existing task's description, mark a task as 'done', 'refine' (clarify/modify) an existing task, or if it's just conversation ('none').

**CRITICAL:** Respond ONLY with a valid JSON object matching the following Pydantic schema. Do NOT include any other text, explanations, or markdown formatting before or after the JSON object.

Schema:
```json
{{
  "operation": "add | update | done | refine | none",
  "task_id": int | null,
  "description": str | null,
  "response_message": str
}}
```

Details:
- 'operation': The action ('add', 'update', 'done', 'refine', 'none').
- 'task_id': Required for 'update', 'done', 'refine'. Null otherwise.
- 'description': Required for 'add'. New description for 'update'/'refine'. Null otherwise.
- 'response_message': A user-facing message confirming the action or continuing conversation.

Example for adding:
```json
{{
  "operation": "add",
  "task_id": null,
  "description": "Buy groceries",
  "response_message": "Okay, I've added 'Buy groceries' to your task list."
}}
```
Example for marking done:
```json
{{
  "operation": "done",
  "task_id": 1,
  "description": null,
  "response_message": "Great! I've marked task 1 as done."
}}
```
Example for no action:
```json
{{
  "operation": "none",
  "task_id": null,
  "description": null,
  "response_message": "Okay, let me know if there's anything else!"
}}
```
"""),
        ("human", "Conversation History:\n{history}\n\nLatest User Message: {input}")
    ])

    # Format history
    history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[:-1]]) # Exclude the latest user message
    latest_input = messages[-1].content if messages else ""
    tasks_json_str = json.dumps(tasks, indent=2) if tasks else "[]"

    # Chain for raw text output
    chain = prompt_template | llm

    try:
        logger.info("Invoking LLM chain...")
        response = chain.invoke({
            "tasks_json": tasks_json_str,
            "history": history,
            "input": latest_input
        })
        raw_response_content = response.content
        logger.info(f"LLM Raw Response:\n{raw_response_content}")

        # Attempt to parse the raw response as JSON into the Pydantic model
        try:
            # Clean potential markdown code fences
            if raw_response_content.strip().startswith("```json"):
                 raw_response_content = raw_response_content.strip()[7:-3].strip()
            elif raw_response_content.strip().startswith("```"):
                 raw_response_content = raw_response_content.strip()[3:-3].strip()

            instruction = TaskUpdateInstruction.parse_raw(raw_response_content)
            logger.info(f"Parsed LLM Instruction: {instruction}")

        except Exception as parse_error:
            logger.error(f"Failed to parse LLM response into TaskUpdateInstruction: {parse_error}")
            logger.error(f"Raw response was: {raw_response_content}")
            # Fallback if parsing fails
            instruction = TaskUpdateInstruction(
                operation="none",
                response_message="Sorry, I couldn't understand the task instruction from the response. Could you try again?"
            )

    except Exception as llm_error:
        logger.error(f"Error calling LLM: {llm_error}", exc_info=True)
        # Fallback if LLM call fails
        instruction = TaskUpdateInstruction(
            operation="none",
            response_message="Sorry, I encountered an error communicating with the language model. Please try again later."
        )

    # Ensure instruction is never None here
    if instruction is None:
         logger.error("Instruction became None unexpectedly, using fallback.")
         instruction = TaskUpdateInstruction(
             operation="none",
             response_message="Sorry, an unexpected internal error occurred. Please rephrase."
         )


    # Add the AI's intended action/response as a message for context
    ai_message_content = f"Instruction: {instruction.operation}"
    if instruction.task_id is not None: # Check for None explicitly
         ai_message_content += f", Task ID: {instruction.task_id}"
    if instruction.description:
         ai_message_content += f", Desc: {instruction.description}"
    # Also include the user-facing message
    ai_message_content += f"\nResponse: {instruction.response_message}"

    # Return the parsed (or fallback) instruction and the AI message
    return {"messages": [AIMessage(content=ai_message_content)], "task_update_instruction": instruction}


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

    logger.info(f"Processing operation: {op}")  # Add logging
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
                task['status'] = "pending"  # Ensure status is pending after update/refine
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
    # Updated config: use a flat dictionary with required key
    config = {"thread_id": str(uuid.uuid4())}

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