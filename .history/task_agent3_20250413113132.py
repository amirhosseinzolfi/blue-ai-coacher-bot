from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, RemoveMessage
import json
import uuid
import operator
from typing import List, Optional, Annotated

# --- Initialize LLM ---
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

# --- Define State ---
class State(TypedDict):
    conversation_history: Annotated[List[BaseMessage], operator.add]
    tasks: list
    summary: Optional[str]

# --- Task Management Functions ---
def add_task(state: State, task: str):
    state['tasks'].append({"task": task, "done": False})

def check_task_done(state: State, task_index: int):
    if 0 <= task_index < len(state['tasks']):
        state['tasks'][task_index]['done'] = True

def refine_tasks(state: State):
    # Logic to refine tasks based on user input
    pass

# --- Message Management Functions ---
def summarize_conversation(state: State):
    """Summarizes conversation and prunes old messages."""
    print("--- Summarizing Conversation ---")
    messages = state["conversation_history"]
    current_summary = state.get("summary")

    # Create summarization prompt
    summary_messages = []
    if current_summary:
        summary_messages.append(SystemMessage(content=f"Previous summary:\n{current_summary}"))
    summary_messages.extend(messages)
    summary_messages.append(HumanMessage(content="Summarize the conversation above, including any tasks discussed."))

    # Get new summary using proper invocation
    summary_response = llm.invoke(summary_messages)
    new_summary = summary_response.content

    # Keep only last 2 messages
    messages_to_keep = 2
    messages_to_remove = [RemoveMessage(id=m.id) for m in messages[:-messages_to_keep]]

    return {"summary": new_summary, "conversation_history": messages_to_remove}

# --- Define Nodes ---
def handle_user_input(state: State):
    """Process user input with context from summary."""
    user_input = input("You: ")
    
    # Prepare messages for LLM
    messages_to_send = []
    if state.get("summary"):
        messages_to_send.append(SystemMessage(content=f"Previous context:\n{state['summary']}"))
    messages_to_send.extend(state["conversation_history"])
    messages_to_send.append(HumanMessage(content=user_input))

    # Process with LLM using proper invocation
    response = llm.invoke(messages_to_send)
    
    # Handle tasks
    if "add task" in user_input.lower():
        task = user_input.split("add task")[-1].strip()
        add_task(state, task)
        print(f"Task added: {task}")
    elif "check task" in user_input.lower():
        task_index = int(user_input.split("check task")[-1].strip())
        check_task_done(state, task_index)
        print(f"Task {task_index} marked as done.")
    
    return {"conversation_history": [HumanMessage(content=user_input), response]}

def should_continue_or_summarize(state: State) -> Literal["summarize", "__end__"]:
    """Decide whether to summarize or end turn."""
    if len(state["conversation_history"]) > 6:  # Threshold for summarization
        return "summarize"
    return END

# --- Build the Graph ---
builder = StateGraph(State)
builder.add_node("agent", handle_user_input)
builder.add_node("summarize", summarize_conversation)

# Define edges
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue_or_summarize,
    {
        "summarize": "summarize",
        END: END
    }
)
builder.add_edge("summarize", END)

# --- Set Up Memory ---
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- Initialize State ---
initial_state = {
    "conversation_history": [],
    "tasks": [],
    "summary": None
}

# --- Run the Agent ---
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "checkpoint_ns": "blue_business",
        "checkpoint_id": str(uuid.uuid4())
    }
}

while True:
    for event in graph.stream(initial_state, config, stream_mode="values"):
        if "messages" in event:
            print("Agent:", event["messages"][-1].content)
