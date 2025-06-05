import json
from typing import List, Dict, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import os

# --- Initialize LLM ---
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)


# --- Define Tools ---
def get_current_tasks(task_file="tasks.json") -> List[Dict[str, str]]:
    """
    Retrieves the current tasks from the tasks.json file.  Handles file not found and JSON decode errors.

    Returns:
        List[Dict[str, str]]: A list of task dictionaries.  Returns an empty list if the file doesn't exist
                            or if there's an error decoding the JSON.
    """
    try:
        with open(task_file, "r") as f:
            tasks = json.load(f)
            if not isinstance(tasks, list):
                print(f"Warning: {task_file} did not contain a list.  Returning an empty list.")
                return []
            return tasks
    except FileNotFoundError:
        print(f"Info: {task_file} not found.  Returning an empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {task_file} contained invalid JSON. Returning an empty list.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Returning an empty list.")
        return []



def update_tasks(tasks: List[Dict[str, str]], task_file="tasks.json") -> None:
    """
    Updates the tasks.json file with the provided list of tasks.
    Handles potential file writing errors.

    Args:
        tasks (List[Dict[str, str]]): The list of task dictionaries to write to the file.
    """
    try:
        with open(task_file, "w") as f:
            json.dump(tasks, f, indent=4)
    except Exception as e:
        print(f"Error writing to {task_file}: {e}")



def add_task(new_task: str, task_file="tasks.json") -> str:
    """Adds a new task to the tasks.json file.

    Args:
        new_task (str): The task to add.
    """
    tasks = get_current_tasks(task_file)
    tasks.append({"task": new_task, "status": "pending"})
    update_tasks(tasks, task_file)
    return f"Added task: {new_task}"



def complete_task(task_number: int, task_file="tasks.json") -> str:
    """Completes a task given its number.

       Numbering starts from 1. Handles invalid task numbers and file errors.
    """
    tasks = get_current_tasks(task_file)
    if not tasks:
        return "No tasks found."

    if 1 <= task_number <= len(tasks):
        tasks[task_number - 1]["status"] = "completed"
        update_tasks(tasks, task_file)
        return f"Completed task: {tasks[task_number - 1]['task']}"
    else:
        return f"Error: Task number {task_number} is invalid.  There are only {len(tasks)} tasks."



def refine_tasks(user_input: str, conversation_history: List[Dict[str, str]], task_file="tasks.json") -> str:
    """Refines the existing tasks based on user input and conversation history.

    Args:
        user_input (str): The latest user input.
        conversation_history (List[Dict[str, str]]): The conversation history.
    """
    tasks = get_current_tasks(task_file)
    # Construct messages for the LLM.  Include system message, conversation history, and current tasks.
    messages: List[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant that manages a list of tasks.  The current tasks are: " + str(tasks) + ".  Use this information to help refine the tasks."),
    ]
    for message in conversation_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        else:
            messages.append(AIMessage(content=message["content"]))
    messages.append(HumanMessage(content=user_input)) # Add the current user input

    # Get the LLM's response.  Instruct it to return a JSON list of tasks.
    response = llm.invoke(messages).content
    try:
        refined_tasks = json.loads(response)
        if not isinstance(refined_tasks, list):
            return "Refinement failed: LLM did not return a list of tasks."
        update_tasks(refined_tasks, task_file)
        return "Tasks refined."
    except json.JSONDecodeError:
        return "Refinement failed: LLM returned invalid JSON."
    except Exception as e:
        return f"Refinement failed: {e}"



def list_tasks(task_file="tasks.json") -> str:
    """Lists all tasks with their statuses.
       Handles the case where there are no tasks.
    """
    tasks = get_current_tasks(task_file)
    if not tasks:
        return "No tasks found."
    else:
        task_list = "\n"
        for i, task in enumerate(tasks):
            task_list += f"{i+1}. {task['task']} - Status: {task['status']}\n"
        return task_list


# --- Define Tools List ---
tools = [
    Tool(
        name="add_task",
        func=add_task,
        description="Adds a new task to the list.",
    ),
    Tool(
        name="complete_task",
        func=complete_task,
        description="Completes a task given its number.",
    ),
    Tool(
        name="refine_tasks",
        func=refine_tasks,
        description="Refines the list of tasks based on user input and conversation history.",
    ),
    Tool(
        name="list_tasks",
        func=list_tasks,
        description="Lists all tasks.",
    ),
    Tool(
        name="get_current_tasks",
        func=get_current_tasks,
        description="Gets the current tasks."
    )
]


# --- Define Agent State ---
class AgentState(TypedDict):
    """
    Represents the state of the agent.  Includes the current tasks,
    the conversation history, and the current user input.
    """
    tasks: List[Dict[str, str]]
    conversation_history: List[Dict[str, str]]
    user_input: str



# --- Define Agent Logic ---
def run_agent(state: AgentState) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """
    The core logic of the agent.  It decides what action to take
    based on the user input and the conversation history.
    """
    print(f"Current state: {state}")  # Print the current state for debugging
    user_input = state["user_input"]
    conversation_history = state["conversation_history"]
    tasks = state["tasks"]

    # Construct messages for the LLM.  Include system message, conversation history, and current tasks.
    messages: List[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant that manages a list of tasks.  You can add tasks, complete tasks, refine tasks, and list tasks.  The current tasks are: " + str(tasks) + ".  Use this information to help determine what action to take."),
    ]
    for message in conversation_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        else:
            messages.append(AIMessage(content=message["content"]))
    messages.append(HumanMessage(content=user_input)) # Add the current user input

    # Get the LLM's response.  Instruct it to choose a tool or respond directly.
    response = llm.invoke(messages).content
    print(f"LLM response: {response}") # Print LLM response

    # Parse the response.  Look for a tool invocation, otherwise treat it as a direct response to the user.
    if "Action:" in response and "Action Input:" in response:
        action = response.split("Action:")[1].split("Action Input:")[0].strip()
        action_input = response.split("Action Input:")[1].strip()

        # Execute the chosen tool.
        for tool in tools:
            if tool.name == action:
                print(f"Calling tool: {action} with input: {action_input}")
                tool_output = tool.func(action_input)
                print(f"Tool output: {tool_output}")
                return {"action": action, "action_input": action_input, "tool_output": tool_output, "tasks": get_current_tasks()}  # Include tasks in the state
        else:
            return {"response": "I don't know how to perform that action.", "tasks": get_current_tasks()}
    else:
        return {"response": response, "tasks": get_current_tasks()}  # Include tasks in the state



# --- Define the LangGraph Graph ---
def create_agent_graph() -> StateGraph:
    """
    Creates the LangGraph graph for the agent.
    """
    # Define the agent state
    state_graph = StateGraph(AgentState)

    # Define the nodes in the graph
    state_graph.add_node("agent", run_agent)

    # Define the edges in the graph
    state_graph.add_edge("agent", END)

    # Set the entry point
    state_graph.set_entry_point("agent")

    return state_graph



if __name__ == "__main__":
    # --- Initialize the graph and agent ---
    agent_graph = create_agent_graph()
    agent = agent_graph.compile()

    # --- Initialize conversation history and tasks ---
    conversation_history: List[Dict[str, str]] = []
    tasks = get_current_tasks() # Load any existing tasks

    # --- Main loop for interacting with the agent ---
    print("Welcome to the Task Management Agent!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # --- Send input to the agent ---
        result = agent.invoke({"user_input": user_input, "conversation_history": conversation_history, "tasks": tasks})

        # --- Print the agent's response ---
        if "response" in result:
            print("Agent:", result["response"])
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "agent", "content": result["response"]})
        elif "tool_output" in result:
            print("Agent:", result["tool_output"])
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "agent", "content": result["tool_output"]})
        tasks = result.get("tasks", tasks) # update the tasks

        # Print current tasks
        print("Current Tasks:")
        print(list_tasks())
