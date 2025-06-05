from typing import Literal, Union
import os
import getpass
from pydantic import BaseModel

from pymongo import MongoClient

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver
from langmem import create_memory_manager

# --- Define Memory Structure ---
class Triple(BaseModel):
    """Store all facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# --- Initialize Environment & LLM ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

llm = ChatOpenAI(
    base_url="http://localhost:15209/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = llm

# --- Define Memory Manager ---
memory_manager = create_memory_manager(
    model="gemini-2.0-flash",  # Using the same model as main conversation
    schemas=[Triple],
    instructions="Extract user preferences and any other useful information as triples.",
    enable_inserts=True,
    enable_deletes=True,
)

# --- Define Persistent MongoDB Checkpointer ---
MONGODB_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
mongodb_client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(mongodb_client)

# --- Define State and Functions ---
class State(MessagesState):
    summary: str
    memories: list = []  # Store extracted memories

def extract_memories(messages):
    """Extract memories from messages using memory manager."""
    return memory_manager.invoke({"messages": messages})

def call_model(state: State):
    summary = state.get("summary", "")
    
    # Extract memories from new messages
    new_memories = extract_memories(state["messages"])
    state["memories"].extend(new_memories)
    
    # Include summary and relevant memories in system message
    if summary:
        context = f"Summary of conversation earlier: {summary}\n\nRelevant facts:"
        for memory in state["memories"][-3:]:  # Include last 3 memories
            if isinstance(memory.content, Triple):
                context += f"\n- {memory.content.subject} {memory.content.predicate} {memory.content.object}"
        messages = [SystemMessage(content=context)] + state["messages"]
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: State) -> Union[Literal["summarize_conversation"], Literal[END]]:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# --- Build LangGraph ---
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue, ["summarize_conversation", END])
workflow.add_edge("summarize_conversation", END)

app = workflow.compile(checkpointer=checkpointer)

# --- Terminal interaction ---
def print_update_terminal(update):
    for k, v in update.items():
        if "messages" in v:
            for m in v["messages"]:
                if isinstance(m, AIMessage):
                    print(f"\nAI: {m.content}")
                elif isinstance(m, HumanMessage):
                    pass
        if "summary" in v:
            print("\nSummary:")
            print(v["summary"])

config = {"configurable": {"thread_id": "terminal_chat"}}

print("Welcome to the LangGraph Terminal Chatbot!")
print("Type 'exit' or 'quit' to end the conversation.")

try:
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break

        input_message = HumanMessage(content=user_input)
        print("AI is thinking...")

        for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
            print_update_terminal(event)
finally:
    # Ensure the MongoDB client is closed after use.
    mongodb_client.close()
