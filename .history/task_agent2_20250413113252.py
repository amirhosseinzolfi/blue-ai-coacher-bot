import json
import os
from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation, ToolOutput
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime

# --- Initialize LLM ---
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

# --- Define the state for the Langgraph agent ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    tasks: Optional[List[Dict]]

# --- Define the tools the agent can use ---

# Define the schema for adding a task
class AddTaskArgs(BaseModel):
    task_description: str = Field(description="The description of the task to add.")

class AddTaskTool(BaseTool):
    name = "add_task"
    description = "Adds a new task to the task list."
    args_schema: type[AddTaskArgs] = AddTaskArgs

    def _run(self, task_description: str) -> str:
        return f"Successfully requested to add task: {task_description}"

    async def _arun(self, task_description: str) -> str:
        raise NotImplementedError("This tool does not support asynchronous execution.")

# Define the schema for checking a task as done
class CheckTaskArgs(BaseModel):
    task_description: str = Field(description="The exact description of the task to mark as done.")

class CheckTaskTool(BaseTool):
    name = "check_task"
    description = "Marks an existing task as done in the task list."
    args_schema: type[CheckTaskArgs] = CheckTaskArgs

    def _run(self, task_description: str) -> str:
        return f"Successfully requested to check task: {task_description} as done."

    async def _arun(self, task_description: str) -> str:
        raise NotImplementedError("This tool does not support asynchronous execution.")

# Define the schema for refining a task
class RefineTaskArgs(BaseModel):
    old_task_description: str = Field(description="The exact description of the task to refine.")
    new_task_description: str = Field(description="The new description for the task.")

class RefineTaskTool(BaseTool):
    name = "refine_task"
    description = "Refines an existing task in the task list with a new description."
    args_schema: type[RefineTaskArgs] = RefineTaskArgs

    def _run(self, old_task_description: str, new_task_description: str) -> str:
        return f"Successfully requested to refine task: '{old_task_description}' to '{new_task_description}'."

    async def _arun(self, old_task_description: str, new_task_description: str) -> str:
        raise NotImplementedError("This tool does not support asynchronous execution.")

# Define the schema for listing all tasks
class ListTasksArgs(BaseModel):
    pass

class ListTasksTool(BaseTool):
    name = "list_tasks"
    description = "Lists all the current tasks in the task list."
    args_schema: type[ListTasksArgs] = ListTasksArgs

    def _run(self) -> str:
        return "Successfully requested to list all tasks."

    async def _arun(self) -> str:
        raise NotImplementedError("This tool does not support asynchronous execution.")

# --- Define the function to load tasks from JSON ---
TASK_FILE = "tasks.json"

def load_tasks():
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    else:
        return []

# --- Define the function to save tasks to JSON ---
def save_tasks(tasks: List[Dict]):
    with open(TASK_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

# --- Define the function to process user input and decide on the next action ---
def route(state):
    messages = state["messages"]
    latest_message = messages[-1]

    prompt = f"""You are a task management expert. Analyze the latest user message and the conversation history to determine the best next action. You can use the following tools:

1.  `add_task`: Use this to add a new task to the list. The input should be a JSON object with the key "task_description".
2.  `check_task`: Use this to mark an existing task as done. The input should be a JSON object with the key "task_description" and the exact description of the task to mark as done.
3.  `refine_task`: Use this to modify an existing task. The input should be a JSON object with keys "old_task_description" (the exact description of the task to refine) and "new_task_description" (the new description).
4.  `list_tasks`: Use this to list all the current tasks. This tool does not require any input.

Based on the user's request, decide which tool to use. If the user is just chatting or the request is unclear, you can respond directly without using a tool by setting the "action" to "respond".

Here's the conversation history:
{messages}

What is the next action? Respond with a JSON object containing the "action" and any necessary "input" for the tool. If the action is "respond", include the "response" in the JSON object.

Example responses:
{{
  "action": "add_task",
  "input": {{
    "task_description": "Buy groceries"
  }}
}}

{{
  "action": "check_task",
  "input": {{
    "task_description": "Book doctor appointment"
  }}
}}

{{
  "action": "refine_task",
  "input": {{
    "old_task_description": "Write report",
    "new_task_description": "Draft the initial version of the quarterly report"
  }}
}}

{{
  "action": "list_tasks"
}}

{{
  "action": "respond",
  "response": "Okay, I understand."
}}

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    response = llm.invoke(prompt)
    try:
        action_data = json.loads(response.content)
        return action_data
    except json.JSONDecodeError:
        return {"action": "respond", "response": "Sorry, I couldn't understand your request. Please try again."}

# --- Define the functions for each tool ---

def add_task(state):
    tool_input = state["tool_input"]
    task_description = tool_input["task_description"]
    tasks = state.get("tasks", load_tasks())
    tasks.append({"description": task_description, "status": "pending"})
    save_tasks(tasks)
    return {"messages": [HumanMessage(content=f"Added task: {task_description}")]}

def check_task(state):
    tool_input = state["tool_input"]
    task_description = tool_input["task_description"]
    tasks = state.get("tasks", load_tasks())
    found = False
    for task in tasks:
        if task["description"] == task_description:
            task["status"] = "done"
            found = True
            break
    save_tasks(tasks)
    if found:
        return {"messages": [HumanMessage(content=f"Marked task '{task_description}' as done.")]}
    else:
        return {"messages": [HumanMessage(content=f"Task '{task_description}' not found.")]}

def refine_task(state):
    tool_input = state["tool_input"]
    old_task_description = tool_input["old_task_description"]
    new_task_description = tool_input["new_task_description"]
    tasks = state.get("tasks", load_tasks())
    found = False
    for task in tasks:
        if task["description"] == old_task_description:
            task["description"] = new_task_description
            found = True
            break
    save_tasks(tasks)
    if found:
        return {"messages": [HumanMessage(content=f"Refined task '{old_task_description}' to '{new_task_description}'.")]}
    else:
        return {"messages": [HumanMessage(content=f"Task '{old_task_description}' not found.")]}

def list_tasks(state):
    tasks = state.get("tasks", load_tasks())
    if not tasks:
        response = "No tasks found."
    else:
        response = "Current tasks:\n"
        for i, task in enumerate(tasks):
            response += f"{i+1}. {task['description']} - Status: {task['status']}\n"
    return {"messages": [HumanMessage(content=response)]}

def respond(state):
    response = state["action_output"]["response"]
    return {"messages": [HumanMessage(content=response)]}

# --- Define the Langgraph graph ---
builder = StateGraph(AgentState)

# Add the routing node
builder.add_node("route", route)

# Add the tool nodes
builder.add_node("add_task", add_task)
builder.add_node("check_task", check_task)
builder.add_node("refine_task", refine_task)
builder.add_node("list_tasks", list_tasks)

# Add the response node
builder.add_node("respond", respond)

# Define the edges
builder.set_entry_point("route")

builder.add_conditional_edges(
    "route",
    lambda x: x.get("action", "respond"),
    {
        "add_task": "add_task",
        "check_task": "check_task",
        "refine_task": "refine_task",
        "list_tasks": "list_tasks",
        "respond": "respond",
    },
)

builder.add_edge("add_task", "route")
builder.add_edge("check_task", "route")
builder.add_edge("refine_task", "route")
builder.add_edge("list_tasks", "route")
builder.add_edge("respond", END)

# Compile the graph
task_management_agent = builder.compile()

# --- Function to run the agent in a terminal ---
def run_agent():
    print("Welcome to the Task Management Agent!")
    conversation_history = []

    # Load initial tasks
    initial_tasks = load_tasks()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        conversation_history.append(HumanMessage(content=user_input))

        result = task_management_agent.invoke({"messages": conversation_history, "tasks": initial_tasks})

        if "messages" in result:
            for message in result["messages"]:
                if isinstance(message, HumanMessage):
                    print(f"Agent: {message.content}")
                    conversation_history.append(message)

        # Update initial tasks after each interaction
        initial_tasks = load_tasks()

if __name__ == "__main__":
    run_agent()