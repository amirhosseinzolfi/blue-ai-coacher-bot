from typing import Literal, Union
import os
import getpass

from pymongo import MongoClient

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver

from pydantic import BaseModel

# --- Define Triple Model used for Memory Extraction ---
class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    context: str | None = None

# --- Initialize Environment & LLM ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("ANTHROPIC_API_KEY")

llm = ChatOpenAI(
    base_url="http://localhost:15209/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = llm

# --- Define Persistent MongoDB Checkpointer ---
MONGODB_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
mongodb_client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(mongodb_client)

# --- Set up Semantic Memory Manager via LangMem ---
from langmem import create_memory_manager
semantic_manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Triple],
    instructions="Extract structured facts as subject-predicate-object triples from the conversation.",
    enable_inserts=True,
    enable_deletes=True,
)

# --- Define State and Functions ---
class State(MessagesState):
    summary: str

def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def extract_semantic_memory(state: State):
    # Prepare messages for LangMem extraction (convert messages to dict format)
    msgs = []
    for m in state["messages"]:
        if hasattr(m, "content"):
            # Use the role name in lowercase
            role = m.__class__.__name__.replace("Message", "").lower()  
            msgs.append({"role": role, "content": m.content})
    # Invoke the semantic memory manager
    semantic_result = semantic_manager.invoke({"messages": msgs})
    # Return results under the key "semantic_memories"
    return {"semantic_memories": semantic_result}

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
workflow.add_node("extract_semantic", extract_semantic_memory)
workflow.add_node("summarize_conversation", summarize_conversation)
# Rewire workflow: go from conversation -> semantic extraction -> conditionally to summarization (or end)
workflow.add_edge(START, "conversation")
workflow.add_edge("conversation", "extract_semantic")
workflow.add_conditional_edges("extract_semantic", should_continue, ["summarize_conversation", END])
workflow.add_edge("summarize_conversation", END)
# Reinitialize semantic memory manager to use our own models instead of Anthropic
semantic_manager = create_memory_manager(
    "openai:gpt-3.5-turbo",
    schemas=[Triple],
    instructions="Extract structured facts as subject-predicate-object triples from the conversation.",
    enable_inserts=True,
    enable_deletes=True,
)
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
        if "semantic_memories" in v:
            print("\nExtracted Semantic Memories:")
            for mem in v["semantic_memories"]:
                # Each memory's content is an instance of Triple (or similar)
                print(mem)

config = {"configurable": {"thread_id": "terminal_chat"}}

print("Welcome to the LangGraph Terminal Chatbot with Semantic Memory!")
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
