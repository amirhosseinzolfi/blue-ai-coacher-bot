# 1. Install necessary libraries
# pip install langchain langchain_openai langchain_ollama langgraph langmem pymongo motor langgraph-checkpoint-mongodb pydantic

import os
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
    # Test LLM connection (optional but recommended)
    # llm.invoke("Hello!")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    # Test Embeddings (optional)
    # embeddings.embed_query("Test embedding")

    print("✅ LLM and Embeddings Initialized Successfully.")
except Exception as e:
    print(f"❌ Error initializing LLM or Embeddings: {e}")
    print("Please ensure the Ollama server and the OpenAI-compatible endpoint are running and accessible.")
    exit() # Exit if core components fail

# Use the same LLM for summarization
summarizer_llm = llm

# --- 3. Define State for the Graph ---
class ChatState(MessagesState):
    """Extends MessagesState to include conversation summary."""
    summary: Optional[str] = Field(default=None, description="Running summary of the conversation")

# --- 4. Setup MongoDB Checkpointer for Persistence ---
print("📦 Setting up MongoDB Checkpointer...")
try:
    # Using MongoClient for potentially longer-running applications
    mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
    # Test MongoDB connection
    mongo_client.admin.command('ping')
    mongodb_checkpointer = MongoDBSaver(mongo_client)
    print("✅ MongoDB Checkpointer Configured.")
    # Note: Remember to call mongo_client.close() when your application shuts down gracefully.
except Exception as e:
    print(f"❌ Error connecting to MongoDB or setting up checkpointer: {e}")
    print(f"Connection String: {MONGODB_CONNECTION_STRING}")
    print("Please ensure MongoDB is running and accessible.")
    # Decide if you want to exit or continue without persistence
    mongodb_checkpointer = None # Fallback to no persistence if connection fails
    print("⚠️ Warning: Proceeding without MongoDB persistence.")


# --- 5. Setup Semantic Memory Extraction (LangMem) ---
print("🧠 Configuring Semantic Memory (LangMem)...")

# Define the structure for extracted facts (as per guide)
class Triple(BaseModel):
    """Store facts, preferences, relationships as subject-predicate-object triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# Using InMemoryStore for LangMem as per the guide's example structure.
# NOTE: This means semantic memory will NOT persist across application restarts
# unless you implement a persistent LangMem store (e.g., using MongoDB directly
# or a dedicated LangMem MongoDB store if available).
langmem_store = InMemoryStore()

# Configure the memory manager linked to the store
# Using the main 'llm' as the manager's LLM, as suggested in the guide
try:
    langmem_manager = create_memory_store_manager(
        llm=llm, # Use the configured LLM
        namespace=("chat", "{thread_id}", "semantic_facts"), # Dynamic namespace per conversation thread
        schemas=[Triple],
        instructions="Extract user preferences, relationships, key facts, and events as structured triples. Focus on information that provides long-term context about the user or conversation topics.",
        enable_inserts=True,
        enable_deletes=True, # Allow updates/deletions
        store=langmem_store # Link the manager to the InMemoryStore
    )
    print("✅ Semantic Memory Manager Configured (using InMemoryStore).")
except Exception as e:
    print(f"❌ Error configuring LangMem manager: {e}")
    langmem_manager = None # Disable semantic memory if setup fails
    print("⚠️ Warning: Proceeding without Semantic Memory Extraction.")


# --- 6. Define Graph Nodes ---

# --- 6.a. Filtering Logic ---
def filter_messages(messages: List[BaseMessage], n: int = MESSAGES_TO_FILTER_FOR_LLM) -> List[BaseMessage]:
    """Keeps only the most recent 'n' messages."""
    return messages[-n:]

# --- 6.b. Agent Node (Calls LLM with Filtering and Summary) ---
def call_llm_node(state: ChatState):
    print("📞 Calling LLM...")
    messages_to_send = []
    # Prepend summary if it exists
    if state.get("summary"):
        print("  Adding summary to context...")
        messages_to_send.append(SystemMessage(content=f"Summary of prior conversation:\n{state['summary']}"))

    # Get current messages from state
    current_messages = state["messages"]

    # Apply filtering to the messages *before* adding summary (or decide if summary counts towards limit)
    # Here, we filter the core messages and then add the summary
    filtered_core_messages = filter_messages(current_messages, MESSAGES_TO_FILTER_FOR_LLM - (1 if state.get("summary") else 0))
    messages_to_send.extend(filtered_core_messages)

    print(f"  Sending {len(messages_to_send)} messages to LLM (after filtering/summary).")

    # Invoke the LLM
    try:
        response = llm.invoke(messages_to_send)
        print("  LLM Response received.")
        # Ensure response is AIMessage
        if not isinstance(response, AIMessage):
             response = AIMessage(content=str(response.content)) # Adapt if necessary based on LLM output type
        return {"messages": [response]}
    except Exception as e:
        print(f"❌ Error during LLM invocation: {e}")
        # Return an error message or handle appropriately
        error_message = AIMessage(content=f"Sorry, I encountered an error: {e}")
        return {"messages": [error_message]}


# --- 6.c. Summarization Node ---
def summarize_node(state: ChatState):
    print("⏳ Summarizing conversation...")
    current_summary = state.get("summary")
    messages_to_summarize = state["messages"] # Use all messages currently in state for summary context

    # Create the prompt for the summarizer LLM
    prompt_parts = []
    if current_summary:
        prompt_parts.append(f"Current summary:\n{current_summary}\n")
    prompt_parts.append("Please summarize the following conversation messages, incorporating the existing summary if provided. Focus on key decisions, facts, and unanswered questions:")
    prompt_text = "\n".join(prompt_parts)

    # Prepare messages for the summarizer LLM
    summarization_messages = [SystemMessage(content=prompt_text)] + messages_to_summarize

    try:
        new_summary_content = summarizer_llm.invoke(summarization_messages).content
        print("  Summarization complete.")

        # Create instructions to remove older messages, keeping the last few
        messages_to_remove_count = len(messages_to_summarize) - MESSAGES_TO_KEEP_AFTER_SUMMARY
        ids_to_remove = [m.id for m in messages_to_summarize[:messages_to_remove_count]]
        delete_instructions = [RemoveMessage(id=msg_id) for msg_id in ids_to_remove]
        print(f"  Generated {len(delete_instructions)} RemoveMessage instructions.")

        return {"summary": new_summary_content, "messages": delete_instructions}
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        # If summarization fails, maybe just return the current state? Or log error.
        # Returning empty dict means state remains unchanged for this node's outputs.
        return {}


# --- 6.d. Semantic Memory Update Node ---
def update_semantic_memory_node(state: ChatState, config: dict):
    if not langmem_manager:
        print("⚠️ Skipping semantic memory update (manager not configured).")
        return {} # No update if manager failed to init

    print("🧠 Updating semantic memory...")
    try:
        # Extract thread_id from config - essential for namespacing
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            print("⚠️ Warning: thread_id not found in config. Cannot update semantic memory.")
            return {}

        # Get the latest messages (e.g., the last user message and AI response)
        # Adjust how many messages you send based on what's needed for context
        messages_for_extraction = state["messages"][-2:] # Example: last user msg + last AI response

        # Invoke the LangMem manager
        # Pass the config so it can resolve the namespace placeholders like {thread_id}
        langmem_manager.invoke({"messages": messages_for_extraction}, config=config)
        print(f"  Semantic memory processed for thread: {thread_id}")
        # Note: Since using InMemoryStore, these memories are only held for the current session thread.
        # For persistence across restarts, a persistent LangMem store is needed.
    except Exception as e:
        print(f"❌ Error updating semantic memory: {e}")
        # Decide how to handle errors, e.g., log and continue.
    return {} # This node modifies the external LangMem store, not the graph state directly


# --- 7. Define Conditional Logic ---

def should_summarize(state: ChatState) -> Literal["summarize", "update_memory"]:
    """Decide whether to summarize based on message count."""
    message_count = len(state["messages"])
    print(f"💬 Checking message count: {message_count} (Threshold: {MAX_MESSAGES_BEFORE_SUMMARY})")
    if message_count > MAX_MESSAGES_BEFORE_SUMMARY:
        print("  Decision: Summarize")
        return "summarize"
    else:
        print("  Decision: Continue (Update Memory)")
        return "update_memory"

# --- 8. Build the Graph ---
print("🏗️ Building the LangGraph graph...")
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("agent", call_llm_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("update_memory", update_semantic_memory_node) # Pass config to this node

# Define edges
workflow.add_edge(START, "agent")

# Conditional edge after agent call
workflow.add_conditional_edges(
    "agent",
    should_summarize,
    {
        "summarize": "summarize",
        "update_memory": "update_memory", # Go directly to memory update if no summary needed
    }
)

# After summarization, update memory
workflow.add_edge("summarize", "update_memory")

# After updating memory, end the turn
workflow.add_edge("update_memory", END)

# --- 9. Compile the Graph with Checkpointer ---
print("⚙️ Compiling the graph...")
# Compile with the MongoDB checkpointer if it was initialized successfully
if mongodb_checkpointer:
    app = workflow.compile(checkpointer=mongodb_checkpointer)
    print("✅ Graph compiled with MongoDB persistence.")
else:
    # Compile without a checkpointer if MongoDB failed
    app = workflow.compile()
    print("⚠️ Graph compiled without persistence (MongoDB connection failed).")


# --- 10. Example Usage ---
if __name__ == "__main__":
    print("\n--- Starting Chatbot Example ---")

    # Use unique thread_id for each conversation
    conversation_thread_id = "example_conversation_123"
    config = {"configurable": {"thread_id": conversation_thread_id}}

    print(f"\n➡️ Starting/Resuming conversation with thread_id: {conversation_thread_id}")

    # Function to interact with the chatbot
    def chat(user_input):
        print(f"\n👤 User: {user_input}")
        response = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        ai_response = response["messages"][-1].content
        print(f"🤖 AI: {ai_response}")

    # Example conversation flow
    chat("Hi there! My name is Bob and I'm interested in learning about LangGraph.")
    chat("What are the main components of a LangGraph application?")
    chat("Tell me more about the StateGraph.")
    chat("How does persistence work with checkpointers?")
    chat("Can you remind me what my name is?") # Test short-term memory

    # Add more messages to trigger summarization (adjust MAX_MESSAGES_BEFORE_SUMMARY if needed)
    print("\n--- Adding more messages to potentially trigger summarization ---")
    chat("What was the first thing I asked you about?")
    chat("And what did I say my name was earlier?")
    chat("Okay, now tell me about semantic memory extraction.") # This message might trigger summary
    chat("How does LangMem fit into this?")
    chat("What's the difference between the summarization and semantic memory?")
    chat("Thanks! Can you recall my name again?") # Test memory after potential summary

    # Check the final state (if persistence is enabled)
    if mongodb_checkpointer:
        try:
            final_state = app.get_state(config)
            print("\n--- Final Conversation State (from MongoDB) ---")
            print(f"Summary: {final_state.summary}")
            print(f"Messages ({len(final_state.messages)}):")
            for msg in final_state.messages:
                 # Check if msg has 'pretty_repr' or just use default repr
                 if hasattr(msg, 'pretty_repr'):
                     print(f"  - {msg.pretty_repr()}")
                 else:
                     print(f"  - {repr(msg)}")

            # Note: Semantic memory (LangMem) state is in 'langmem_store' (InMemory)
            # and not directly part of the graph state saved by the checkpointer
            # unless a persistent LangMem store integrated with the checkpointer is used.
            print("\n--- Semantic Memory (InMemoryStore - current session only) ---")
            # This requires inspecting the InMemoryStore directly, which isn't straightforward
            # without adding specific retrieval logic or using LangMem's search tools.
            # For demonstration, we'll just note it's not in the graph state.
            print("(Semantic facts stored in LangMem's InMemoryStore for this session)")

        except Exception as e:
            print(f"\n❌ Error retrieving final state: {e}")

    print("\n--- Chatbot Example Finished ---")

    # --- Cleanup ---
    if mongodb_checkpointer and mongo_client:
        print("Closing MongoDB client connection.")
        mongo_client.close()
