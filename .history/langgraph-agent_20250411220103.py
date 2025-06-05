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
from langmem import create_memory_manager            # Added for semantic memory extraction
from pydantic import BaseModel                        # Added for Triple definition


# --- Initialize Environment & LLM ---
def _set_env(var: str, default: str | None = None):
    if not os.environ.get(var):
        if default:
            os.environ[var] = default
        else:
            os.environ[var] = getpass.getpass(f"{var}: ")

# Set OpenAI key only
_set_env("OPENAI_API_KEY", "123")  # Set default OpenAI key

llm = ChatOpenAI(
    base_url="http://localhost:15209/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

memory_llm = ChatOpenAI(
    base_url="http://localhost:15209/v1",
    model_name="gpt-4o",
    temperature=0.5,
    api_key="324"
)


embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = llm

# --- Define Persistent MongoDB Checkpointer ---
MONGODB_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
mongodb_client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(mongodb_client)

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

# --- (New) Semantic Memory Setup ---
ENABLE_SEMANTIC_MEMORY = True

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    context: str | None = None

def extract_semantic_memory(messages):
    try:
        input_messages = []
        for m in messages:
            role = "human" if m.__class__.__name__ == "HumanMessage" else "ai" if m.__class__.__name__ == "AIMessage" else "system"
            input_messages.append({"role": role, "content": m.content})

        # Use memory_llm instead of regular llm
        semantic_manager = create_memory_manager(
            memory_llm,  # Use dedicated memory model
            schemas=[Triple],
            instructions="""Extract key facts as subject-predicate-object triples. 
            For example: 'Alice manages Bob' becomes {subject: 'Alice', predicate: 'manages', object: 'Bob'}""",
            enable_inserts=True,
            enable_deletes=True,
        )
        
        memories = semantic_manager.invoke({"messages": input_messages})
        print("\n[Semantic Memory Extraction using GPT-4]:")
        for mem in memories:
            print(mem)
    except Exception as e:
        print(f"\n[Semantic Memory Error]: {str(e)}")
        print("Continuing conversation without memory extraction...")

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

# Initialize a conversation history list for semantic memory extraction
conversation_history = []  

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
        # Append user message to history
        conversation_history.append(input_message)
        print("AI is thinking...")

        for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
            print_update_terminal(event)
            # Append any new AI messages from update to history
            for m in event.get("messages", []):
                if isinstance(m, (HumanMessage, AIMessage)):
                    conversation_history.append(m)
        if ENABLE_SEMANTIC_MEMORY:
            extract_semantic_memory(conversation_history)
finally:
    # Ensure the MongoDB client is closed after use.
    mongodb_client.close()
