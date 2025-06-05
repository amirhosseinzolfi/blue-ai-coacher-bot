import asyncio
import getpass
import os
import operator
import json
from typing import Annotated, TypedDict, Literal, List, Optional, dict

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import RemoveMessage # For pruning history

# --- MongoDB Persistence Imports ---
from motor.motor_asyncio import AsyncIOMotorClient
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

# --- Configuration ---

# 1. API Keys & Environment Setup
def _set_env(var: str):
    """Helper function to set environment variables if not already set."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var}: ")

# Set LangSmith keys for tracing (optional but recommended)
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "LangGraph Task Manager with History" # Optional project name

# 2. LLM Configuration
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)
print("ChatOpenAI LLM Initialized.")

# 3. MongoDB Configuration
# --- IMPORTANT: Replace with your MongoDB connection string ---
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "langgraph_task_manager_db"
COLLECTION_NAME = "task_agent_checkpoints"

# 4. Conversation Management Configuration
MAX_MESSAGES_BEFORE_SUMMARY = 5 # Number of messages before triggering summarization
MESSAGES_TO_KEEP_AFTER_SUMMARY = 3 # Number of recent messages to retain after summarizing

# --- Define the state for the Langgraph agent ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tasks: Optional[List[dict]]
    summary: Optional[str]

# --- Define the function to load tasks from JSON ---
TASK_FILE = "tasks.json"

async def load_tasks(config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    if thread_id == "default":
        if os.path.exists(TASK_FILE):
            with open(TASK_FILE, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        else:
            return []
    else:
        # When using MongoDB, tasks should ideally be part of the state
        # For simplicity in this example, we'll still use the file for initial load if no history
        checkpointer = config.get("checkpointer")
        if checkpointer:
            state = await checkpointer.aget(config)
            if state and "tasks" in state:
                return state["tasks"]
        if os.path.exists(TASK_FILE):
            with open(TASK_FILE, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

# --- Define the function to save tasks to JSON ---
async def save_tasks(state: AgentState, config: RunnableConfig):
    tasks = state.get("tasks")
    if tasks:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        if thread_id == "default":
            with open(TASK_FILE, "w") as f:
                json.dump(tasks, f, indent=4)
        else:
            # When using MongoDB, tasks are part of the state and will be saved by the checkpointer
            pass
    return state

# --- Define the function to process user input and decide on the next action ---
async def route(state: AgentState):
    messages = state["messages"]
    latest_message = messages[-1]
    current_summary = state.get("summary")

    prompt = f"""You are a task management expert. Analyze the latest user message and the conversation history to determine the best next action. You can:

1.  Add a new task.
2.  Check a task as done.
3.  Refine an existing task.
4.  List all current tasks.
5.  Respond directly to the user.

Here's the conversation history:
{messages}

{'Here is a summary of the previous conversation: ' + current_summary if current_summary else ''}

What is the user trying to do? Respond with a JSON object containing the "action" and any necessary "input".

Example responses:
{{
  "action": "add_task",
  "input": "Buy groceries"
}}

{{
  "action": "check_task",
  "input": "Book doctor appointment"
}}

{{
  "action": "refine_task",
  "input": {{
    "old_task": "Write report",
    "new_task": "Draft the initial version of the quarterly report"
  }}
}}

{{
  "action": "list_tasks"
}}

{{
  "action": "respond",
  "response": "Okay, I understand."
}}

Current time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    response = await llm.ainvoke(prompt)
    try:
        action_data = json.loads(response.content)
        return {"action": action_data.get("action"), "input": action_data.get("input"), "messages": [AIMessage(content=response.content)]}
    except json.JSONDecodeError:
        return {"action": "respond", "response": "Sorry, I couldn't understand your request. Please try again.", "messages": [AIMessage(content="Sorry, I couldn't understand your request. Please try again.")]}

# --- Define the functions for each action ---

async def add_task(state: AgentState):
    action_input = state.get("input")
    if action_input:
        task_description = action_input
        tasks = state.get("tasks", [])
        tasks.append({"description": task_description, "status": "pending"})
        return {"tasks": tasks, "messages": [AIMessage(content=f"Added task: {task_description}")]}
    return {"messages": [AIMessage(content="Please provide the description of the task to add.")]}

async def check_task(state: AgentState):
    action_input = state.get("input")
    if action_input:
        task_description = action_input
        tasks = state.get("tasks", [])
        found = False
        for task in tasks:
            if task["description"] == task_description:
                task["status"] = "done"
                found = True
                break
        if found:
            return {"tasks": tasks, "messages": [AIMessage(content=f"Marked task '{task_description}' as done.")]}
        else:
            return {"messages": [AIMessage(content=f"Task '{task_description}' not found.")]}
    return {"messages": [AIMessage(content="Please provide the description of the task to mark as done.")]}

async def refine_task(state: AgentState):
    action_input = state.get("input")
    if isinstance(action_input, dict) and "old_task" in action_input and "new_task" in action_input:
        old_task_description = action_input["old_task"]
        new_task_description = action_input["new_task"]
        tasks = state.get("tasks", [])
        found = False
        for task in tasks:
            if task["description"] == old_task_description:
                task["description"] = new_task_description
                found = True
                break
        if found:
            return {"tasks": tasks, "messages": [AIMessage(content=f"Refined task '{old_task_description}' to '{new_task_description}'.")]}
        else:
            return {"messages": [AIMessage(content=f"Task '{old_task_description}' not found.")]}
    return {"messages": [AIMessage(content="Please provide the old and new task descriptions for refinement.")]}

async def list_tasks(state: AgentState):
    tasks = state.get("tasks", [])
    if not tasks:
        response = "No tasks found."
    else:
        response = "Current tasks:\n"
        for i, task in enumerate(tasks):
            response += f"{i+1}. {task['description']} - Status: {task['status']}\n"
    return {"messages": [AIMessage(content=response)]}

async def respond(state: AgentState):
    response = state.get("response")
    if response:
        return {"messages": [AIMessage(content=response)]}
    return {}

async def summarize_conversation(state: AgentState):
    """Summarizes the conversation history."""
    print(f"--- Summarizing Conversation (>{MAX_MESSAGES_BEFORE_SUMMARY} messages) ---")
    current_messages = state["messages"]
    current_summary = state.get("summary")

    # Prepare messages for the summarization call
    summary_prompt_messages = []
    if current_summary:
        summary_prompt_messages.append(SystemMessage(content=f"Existing summary:\n{current_summary}"))
    summary_prompt_messages.extend(current_messages)
    summary_prompt_messages.append(HumanMessage(content="Summarize the conversation above, concisely capturing key points and user/assistant intents."))

    try:
        summary_response = await llm.ainvoke(summary_prompt_messages)
        new_summary = summary_response.content
        print(f"--- New Summary: {new_summary} ---")
        # Prune messages: Keep last N + the summary message
        messages_to_keep = current_messages[-MESSAGES_TO_KEEP_AFTER_SUMMARY:]
        return {"summary": new_summary, "messages": messages_to_keep}
    except Exception as e:
        print(f"Error during summarization: {e}")
        return {"summary": current_summary}

def should_summarize(state: AgentState):
    return len(state.get("messages", [])) > MAX_MESSAGES_BEFORE_SUMMARY

# --- Define the Langgraph graph ---
builder = StateGraph(AgentState)

# Add the nodes
builder.add_node("load_tasks", load_tasks)
builder.add_node("route", route)
builder.add_node("add_task", add_task)
builder.add_node("check_task", check_task)
builder.add_node("refine_task", refine_task)
builder.add_node("list_tasks", list_tasks)
builder.add_node("respond", respond)
builder.add_node("summarize", summarize_conversation)
builder.add_node("save_tasks", save_tasks)

# Define the edges
builder.set_entry_point("load_tasks")

builder.add_edge("load_tasks", "route")

builder.add_conditional_edges(
    "route",
    should_summarize,
    {
        True: "summarize",
        False: "handle_action"
    }
)

builder.add_conditional_edges(
    "handle_action",
    lambda x: x.get("action"),
    {
        "add_task": "add_task",
        "check_task": "check_task",
        "refine_task": "refine_task",
        "list_tasks": "list_tasks",
        "respond": "respond",
    },
)

builder.add_edge("add_task", "save_tasks")
builder.add_edge("check_task", "save_tasks")
builder.add_edge("refine_task", "save_tasks")
builder.add_edge("list_tasks", "save_tasks")
builder.add_edge("respond", END)
builder.add_edge("summarize", "route") # After summarizing, go back to routing
builder.add_edge("save_tasks", "route")

# Compile the graph
task_management_agent = builder.compile()

# --- Function to run the agent in a terminal with MongoDB persistence ---
async def run_agent():
    print("Welcome to the Task Management Agent with History!")

    # Initialize MongoDB client
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    checkpointer = AsyncMongoDBSaver(client=client, database=DB_NAME, collection=COLLECTION_NAME)

    thread_id = "task-manager-terminal-chat" # Consistent ID for the conversation

    async def invoke_agent(user_input: str, current_messages: List[BaseMessage] = None):
        messages = current_messages if current_messages else []
        messages.append(HumanMessage(content=user_input))
        config = RunnableConfig(configurable={"thread_id": thread_id}, checkpointer=checkpointer)
        result = await task_management_agent.ainvoke({"messages": messages}, config)
        return result

    # Load initial state (including tasks if available)
    config_load = RunnableConfig(configurable={"thread_id": thread_id}, checkpointer=checkpointer)
    initial_state = await checkpointer.aget(config_load)
    conversation_history = initial_state.get("messages", []) if initial_state else []
    tasks = initial_state.get("tasks", []) if initial_state else await load_tasks(config_load) # Load from file if no history

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = await invoke_agent(user_input, conversation_history)

        if "messages" in result:
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    print(f"Agent: {message.content}")
                    conversation_history.append(message)

        # Update tasks in the conversation history if they were modified
        if "tasks" in result:
            tasks = result["tasks"]

        # Optionally print the current state for debugging
        # current_state = await checkpointer.aget({"configurable": {"thread_id": thread_id}})
        # print("Current State:", current_state)

    # Close MongoDB connection
    client.close()

if __name__ == "__main__":
    import datetime
    asyncio.run(run_agent())