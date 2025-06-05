from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import json
import uuid

# --- Initialize LLM ---
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

# --- Define State ---
class State(TypedDict):
    conversation_history: list
    tasks: list
    configurable: dict  # Changed back from settings to configurable

# --- Task Management Functions ---
def add_task(state: State, task: str):
    state['tasks'].append({"task": task, "done": False})

def check_task_done(state: State, task_index: int):
    if 0 <= task_index < len(state['tasks']):
        state['tasks'][task_index]['done'] = True

def refine_tasks(state: State):
    # Logic to refine tasks based on user input
    pass

# --- Define Nodes ---
def handle_user_input(state: State):
    user_input = input("You: ")
    state['conversation_history'].append(user_input)
    
    # Process user input with LLM
    response = llm({"messages": [{"role": "user", "content": user_input}]})
    
    # Analyze response to manage tasks
    if "add task" in user_input.lower():
        task = user_input.split("add task")[-1].strip()
        add_task(state, task)
        print(f"Task added: {task}")
    elif "check task" in user_input.lower():
        task_index = int(user_input.split("check task")[-1].strip())
        check_task_done(state, task_index)
        print(f"Task {task_index} marked as done.")
    elif "refine tasks" in user_input.lower():
        refine_tasks(state)
        print("Tasks refined.")
    
    return response

# --- Build the Graph ---
builder = StateGraph(State)
builder.add_node("handle_user_input", handle_user_input)
builder.add_edge(START, "handle_user_input")
builder.add_edge("handle_user_input", END)

# --- Set Up Memory ---
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- Initialize State ---
initial_state = {
    "conversation_history": [],
    "tasks": [],
    "configurable": {  # Changed from settings to configurable
        "thread_id": str(uuid.uuid4()),  # Unique thread ID
        "checkpoint_ns": "blue_business",  # Non-empty namespace value
        "checkpoint_id": str(uuid.uuid4())  # Unique checkpoint ID
    }
}

# --- Run the Agent ---
while True:
    for event in graph.stream(initial_state, {}, stream_mode="values"):
        if "messages" in event:
            print("Agent:", event["messages"][-1]["content"])
