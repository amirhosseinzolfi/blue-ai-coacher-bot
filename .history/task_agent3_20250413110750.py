from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.func import task, entrypoint
from langchain_openai import ChatOpenAI
import json

# Define the state structure
class State(TypedDict):
    conversation_history: list
    tasks: list

# Initialize the LLM
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

# Initialize the state
initial_state: State = {
    "conversation_history": [],
    "tasks": []
}

# Function to add a task
def add_task(state: State, task: str) -> State:
    state['tasks'].append({"task": task, "done": False})
    return state

# Function to check a task as done
def check_task_done(state: State, task_index: int) -> State:
    if 0 <= task_index < len(state['tasks']):
        state['tasks'][task_index]['done'] = True
    return state

# Function to refine tasks
def refine_tasks(state: State, new_tasks: list) -> State:
    state['tasks'] = new_tasks
    return state

# Function to process user input
def process_user_input(user_input: str, state: State) -> tuple[State, str]:
    # Update conversation history
    state['conversation_history'].append(user_input)

    # Use the LLM to analyze the user input and generate tasks
    response = llm({"messages": [{"role": "user", "content": user_input}]})
    
    # Here we would parse the response to determine if it includes tasks
    # For simplicity, let's assume the response contains a task suggestion
    task_suggestion = response['content']  # This should be parsed appropriately

    # Add the suggested task
    state = add_task(state, task_suggestion)

    # Return the updated state and response
    return state, f"Task added: {task_suggestion}"

# Create the StateGraph
builder = StateGraph(State)
builder.add_node("process_input", process_user_input)
builder.add_edge(START, "process_input")
builder.add_edge("process_input", END)

# Compile the graph
graph = builder.compile()

# Define the entrypoint for the agent
@entrypoint
def run_agent():
    state = initial_state
    print("Welcome to the Task Manager Agent! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        state, response = process_user_input(user_input, state)
        print("Agent:", response)
        print("Current Tasks:", json.dumps(state['tasks'], indent=2))

# Run the agent
if __name__ == "__main__":
    run_agent()
