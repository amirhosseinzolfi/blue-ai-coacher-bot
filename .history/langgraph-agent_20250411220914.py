# 1. Install necessary libraries
# pip install langchain langchain_openai langchain_ollama langgraph langmem pymongo motor langgraph-checkpoint-mongodb pydantic

import os
import sys # Import sys for exit
from typing import List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore # Using InMemory for LangMem store as per guide example structure

from langmem import create_memory_store_manager
from pydantic import BaseModel, Field

from pymongo import MongoClient
# from motor.motor_asyncio import AsyncIOMotorClient # Import if using async checkpointer

# --- Configuration ---
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017/" # Replace with your MongoDB connection string
LLM_BASE_URL = "http://localhost:15209/v1" # User specified LLM endpoint
LLM_MODEL_NAME = "gpt-4o"
LLM_API_KEY = "324" # User specified API key (handle securely in production)
EMBEDDING_MODEL = "nomic-embed-text"
MAX_MESSAGES_BEFORE_SUMMARY = 6 # Number of messages before triggering summarization
MESSAGES_TO_KEEP_AFTER_SUMMARY = 2 # Number of recent messages to keep after summarizing
MESSAGES_TO_FILTER_FOR_LLM = 10 # Max recent messages to send to LLM (includes summary)

# --- 2. Initialize LLM & Embeddings ---
print("🚀 Initializing LLM and Embeddings...")
try:
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model_name=LLM_MODEL_NAME,
        temperature=0.5,
        api_key=LLM_API_KEY
    )
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print("✅ LLM and Embeddings Initialized Successfully.")
except Exception as e:
    print(f"❌ Error initializing LLM or Embeddings: {e}")
    print("Please ensure the Ollama server and the OpenAI-compatible endpoint are running and accessible.")
    sys.exit(1) # Exit if core components fail

# Use the same LLM for summarization
summarizer_llm = llm

# --- 3. Define State for the Graph ---
class ChatState(MessagesState):
    """Extends MessagesState to include conversation summary."""
    summary: Optional[str] = Field(default=None, description="Running summary of the conversation")

# --- 4. Setup MongoDB Checkpointer for Persistence ---
print("📦 Setting up MongoDB Checkpointer...")
mongo_client = None # Initialize client to None
mongodb_checkpointer = None # Initialize checkpointer to None
try:
    mongo_client = MongoClient(MONGODB_CONNECTION_STRING, serverSelectionTimeoutMS=5000) # Add timeout
    # Test MongoDB connection
    mongo_client.admin.command('ping')
    mongodb_checkpointer = MongoDBSaver(mongo_client)
    print("✅ MongoDB Checkpointer Configured.")
except Exception as e:
    print(f"❌ Error connecting to MongoDB or setting up checkpointer: {e}")
    print(f"Connection String: {MONGODB_CONNECTION_STRING}")
    print("Please ensure MongoDB is running and accessible.")
    print("⚠️ Warning: Proceeding without MongoDB persistence.")
    if mongo_client: # Close client if connection failed after opening
        mongo_client.close()
    mongo_client = None # Ensure client is None if setup failed


# --- 5. Setup Semantic Memory Extraction (LangMem) ---
print("🧠 Configuring Semantic Memory (LangMem)...")
langmem_store = InMemoryStore() # Using InMemoryStore for LangMem
langmem_manager = None # Initialize manager to None
try:
    langmem_manager = create_memory_store_manager(
        llm=llm,
        namespace=("chat", "{thread_id}", "semantic_facts"),
        schemas=[Triple],
        instructions="Extract user preferences, relationships, key facts, and events as structured triples. Focus on information that provides long-term context about the user or conversation topics.",
        enable_inserts=True,
        enable_deletes=True,
        store=langmem_store
    )
    print("✅ Semantic Memory Manager Configured (using InMemoryStore).")
except Exception as e:
    print(f"❌ Error configuring LangMem manager: {e}")
    print("⚠️ Warning: Proceeding without Semantic Memory Extraction.")

# Define the Triple schema here as it's used in LangMem setup
class Triple(BaseModel):
    """Store facts, preferences, relationships as subject-predicate-object triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# --- 6. Define Graph Nodes ---

# --- 6.a. Filtering Logic ---
def filter_messages(messages: List[BaseMessage], n: int = MESSAGES_TO_FILTER_FOR_LLM) -> List[BaseMessage]:
    """Keeps only the most recent 'n' messages."""
    return messages[-n:]

# --- 6.b. Agent Node (Calls LLM with Filtering and Summary) ---
def call_llm_node(state: ChatState):
    print("📞 Calling LLM...")
    messages_to_send = []
    if state.get("summary"):
        print("  Adding summary to context...")
        messages_to_send.append(SystemMessage(content=f"Summary of prior conversation:\n{state['summary']}"))

    current_messages = state["messages"]
    # Filter core messages, leave space for summary if present
    filtered_core_messages = filter_messages(current_messages, MESSAGES_TO_FILTER_FOR_LLM - (1 if state.get("summary") else 0))
    messages_to_send.extend(filtered_core_messages)

    print(f"  Sending {len(messages_to_send)} messages to LLM (after filtering/summary).")

    try:
        response = llm.invoke(messages_to_send)
        print("  LLM Response received.")
        if not isinstance(response, AIMessage):
             response = AIMessage(content=str(response.content))
        return {"messages": [response]}
    except Exception as e:
        print(f"❌ Error during LLM invocation: {e}")
        error_message = AIMessage(content=f"Sorry, I encountered an error processing your request.")
        return {"messages": [error_message]}


# --- 6.c. Summarization Node ---
def summarize_node(state: ChatState):
    print("⏳ Summarizing conversation...")
    current_summary = state.get("summary")
    messages_to_summarize = state["messages"]

    prompt_parts = []
    if current_summary:
        prompt_parts.append(f"Current summary:\n{current_summary}\n")
    prompt_parts.append("Please summarize the following conversation messages, incorporating the existing summary if provided. Focus on key decisions, facts, and unanswered questions:")
    prompt_text = "\n".join(prompt_parts)

    summarization_messages = [SystemMessage(content=prompt_text)] + messages_to_summarize

    try:
        new_summary_content = summarizer_llm.invoke(summarization_messages).content
        print("  Summarization complete.")

        messages_to_remove_count = len(messages_to_summarize) - MESSAGES_TO_KEEP_AFTER_SUMMARY
        ids_to_remove = [m.id for m in messages_to_summarize[:messages_to_remove_count]]
        delete_instructions = [RemoveMessage(id=msg_id) for msg_id in ids_to_remove]
        print(f"  Generated {len(delete_instructions)} RemoveMessage instructions.")

        return {"summary": new_summary_content, "messages": delete_instructions}
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return {}


# --- 6.d. Semantic Memory Update Node ---
def update_semantic_memory_node(state: ChatState, config: dict):
    if not langmem_manager:
        # print("⚠️ Skipping semantic memory update (manager not configured).") # Reduce noise
        return {}

    print("🧠 Updating semantic memory...")
    try:
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            print("⚠️ Warning: thread_id not found in config. Cannot update semantic memory.")
            return {}

        messages_for_extraction = state["messages"][-2:] # Last user msg + last AI response

        langmem_manager.invoke({"messages": messages_for_extraction}, config=config)
        print(f"  Semantic memory processed for thread: {thread_id}")
    except Exception as e:
        print(f"❌ Error updating semantic memory: {e}")
    return {}


# --- 7. Define Conditional Logic ---

def should_summarize(state: ChatState) -> Literal["summarize", "update_memory"]:
    """Decide whether to summarize based on message count."""
    message_count = len(state["messages"])
    # print(f"💬 Checking message count: {message_count} (Threshold: {MAX_MESSAGES_BEFORE_SUMMARY})") # Reduce noise
    if message_count > MAX_MESSAGES_BEFORE_SUMMARY:
        print("  Decision: Summarize")
        return "summarize"
    else:
        # print("  Decision: Continue (Update Memory)") # Reduce noise
        return "update_memory"

# --- 8. Build the Graph ---
print("🏗️ Building the LangGraph graph...")
workflow = StateGraph(ChatState)

workflow.add_node("agent", call_llm_node)
workflow.add_node("summarize", summarize_node)
# Pass config to the semantic memory node when adding it
workflow.add_node("update_memory", update_semantic_memory_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_summarize,
    {"summarize": "summarize", "update_memory": "update_memory"}
)
workflow.add_edge("summarize", "update_memory")
workflow.add_edge("update_memory", END)

# --- 9. Compile the Graph with Checkpointer ---
print("⚙️ Compiling the graph...")
if mongodb_checkpointer:
    app = workflow.compile(checkpointer=mongodb_checkpointer)
    print("✅ Graph compiled with MongoDB persistence.")
else:
    app = workflow.compile()
    print("⚠️ Graph compiled without persistence (MongoDB connection failed).")


# --- 10. Interactive Terminal Chat Loop ---
if __name__ == "__main__":
    print("\n--- LangGraph Chatbot Terminal ---")
    print("Enter 'quit' or 'exit' to end the chat.")

    # Use a fixed or dynamically generated thread_id
    conversation_thread_id = "terminal_chat_session_001"
    config = {"configurable": {"thread_id": conversation_thread_id}}

    print(f"\n➡️ Starting/Resuming conversation with thread_id: {conversation_thread_id}")
    if mongodb_checkpointer:
        print("(Persistence enabled via MongoDB)")
    else:
        print("(Persistence disabled)")

    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("🤖 AI: Goodbye!")
                break

            if not user_input.strip(): # Handle empty input
                continue

            # Invoke the graph
            print("...") # Indicate processing
            response = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

            # Get the latest AI message
            ai_message = response["messages"][-1]
            if isinstance(ai_message, AIMessage):
                print(f"🤖 AI: {ai_message.content}")
            else:
                # Handle cases where the last message might not be AI (e.g., error state)
                print(f"🤖 AI: (Received unexpected message type: {type(ai_message)})")


        except KeyboardInterrupt:
            print("\n🤖 AI: Goodbye! (Interrupted by user)")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            # Depending on the error, you might want to break or continue
            # break

    # --- Cleanup ---
    if mongo_client:
        print("\nClosing MongoDB client connection.")
        mongo_client.close()

    print("\n--- Chat Session Ended ---")