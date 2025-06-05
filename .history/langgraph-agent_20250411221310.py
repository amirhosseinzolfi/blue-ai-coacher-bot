# 1. Install necessary libraries
# pip install langchain langchain_openai langchain_ollama langgraph langmem pymongo motor langgraph-checkpoint-mongodb pydantic

import os
import sys
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
from langgraph.store.memory import InMemoryStore # Using InMemory for LangMem store

from langmem import create_memory_store_manager
from pydantic import BaseModel, Field

from pymongo import MongoClient
# from motor.motor_asyncio import AsyncIOMotorClient # Import if using async checkpointer

# --- Configuration ---
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017/" # Replace with your MongoDB connection string
LLM_BASE_URL = "http://localhost:15209/v1" # User specified LLM endpoint
LLM_API_KEY = "324" # User specified API key (handle securely in production)

MAIN_LLM_MODEL = "gemini-2.0-flash" # Changed as requested
MEMORY_LLM_MODEL = "gpt-4o"         # Separate model for memory extraction
EMBEDDING_MODEL = "nomic-embed-text"

MAX_MESSAGES_BEFORE_SUMMARY = 6 # Number of messages before triggering summarization
MESSAGES_TO_KEEP_AFTER_SUMMARY = 2 # Number of recent messages to keep after summarizing
MESSAGES_TO_FILTER_FOR_LLM = 10 # Max recent messages to send to LLM (includes summary)

# --- 2. Initialize LLMs & Embeddings ---
print("🚀 Initializing LLMs and Embeddings...")
try:
    # Main LLM for chat and summarization
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model_name=MAIN_LLM_MODEL,
        temperature=0.5,
        api_key=LLM_API_KEY
    )
    # Separate LLM for semantic memory extraction
    memory_llm = ChatOpenAI(
        base_url=LLM_BASE_URL, # Assuming same endpoint for both models
        model_name=MEMORY_LLM_MODEL,
        temperature=0.5, # Can adjust temperature for memory extraction if needed
        api_key=LLM_API_KEY
    )
    # Use the main LLM for summarization
    summarizer_llm = llm

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print(f"✅ Main LLM ({MAIN_LLM_MODEL}), Memory LLM ({MEMORY_LLM_MODEL}), and Embeddings Initialized Successfully.")
except Exception as e:
    print(f"❌ Error initializing LLMs or Embeddings: {e}")
    print("Please ensure the Ollama server and the OpenAI-compatible endpoint are running and accessible, and serve the specified models.")
    sys.exit(1) # Exit if core components fail

# --- 3. Define State for the Graph ---
class ChatState(MessagesState):
    """Extends MessagesState to include conversation summary."""
    summary: Optional[str] = Field(default=None, description="Running summary of the conversation")

# --- 4. Define Semantic Memory Schema FIRST ---
class Triple(BaseModel):
    """Store facts, preferences, relationships as subject-predicate-object triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# --- 5. Setup MongoDB Checkpointer for Persistence ---
print("📦 Setting up MongoDB Checkpointer...")
mongo_client = None # Initialize client to None
mongodb_checkpointer = None # Initialize checkpointer to None
try:
    mongo_client = MongoClient(MONGODB_CONNECTION_STRING, serverSelectionTimeoutMS=5000) # Add timeout
    mongo_client.admin.command('ping') # Test connection
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


# --- 6. Setup Semantic Memory Extraction (LangMem) ---
print("🧠 Configuring Semantic Memory (LangMem)...")
langmem_store = InMemoryStore() # Using InMemoryStore for LangMem
langmem_manager = None # Initialize manager to None
try:
    # Use the dedicated memory_llm here
    langmem_manager = create_memory_store_manager(
        llm=memory_llm, # Use the separate LLM for memory tasks
        namespace=("chat", "{thread_id}", "semantic_facts"),
        schemas=[Triple], # Triple class is now defined
        instructions="Extract user preferences, relationships, key facts, and events as structured triples. Focus on information that provides long-term context about the user or conversation topics.",
        enable_inserts=True,
        enable_deletes=True,
        store=langmem_store
    )
    print("✅ Semantic Memory Manager Configured (using InMemoryStore and dedicated Memory LLM).")
except Exception as e:
    print(f"❌ Error configuring LangMem manager: {e}")
    print("⚠️ Warning: Proceeding without Semantic Memory Extraction.")


# --- 7. Define Graph Nodes ---

# --- 7.a. Filtering Logic ---
def filter_messages(messages: List[BaseMessage], n: int = MESSAGES_TO_FILTER_FOR_LLM) -> List[BaseMessage]:
    """Keeps only the most recent 'n' messages."""
    return messages[-n:]

# --- 7.b. Agent Node (Calls LLM with Filtering and Summary) ---
def call_llm_node(state: ChatState):
    print("📞 Calling Main LLM...")
    messages_to_send = []
    if state.get("summary"):
        # print("  Adding summary to context...") # Reduce noise
        messages_to_send.append(SystemMessage(content=f"Summary of prior conversation:\n{state['summary']}"))

    current_messages = state["messages"]
    filtered_core_messages = filter_messages(current_messages, MESSAGES_TO_FILTER_FOR_LLM - (1 if state.get("summary") else 0))
    messages_to_send.extend(filtered_core_messages)

    # print(f"  Sending {len(messages_to_send)} messages to LLM (after filtering/summary).") # Reduce noise

    try:
        # Use the main llm instance
        response = llm.invoke(messages_to_send)
        # print("  LLM Response received.") # Reduce noise
        if not isinstance(response, AIMessage):
             response = AIMessage(content=str(response.content))
        return {"messages": [response]}
    except Exception as e:
        print(f"❌ Error during Main LLM invocation: {e}")
        error_message = AIMessage(content=f"Sorry, I encountered an error processing your request.")
        return {"messages": [error_message]}


# --- 7.c. Summarization Node ---
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
        # Use the summarizer_llm (which is the main llm in this setup)
        new_summary_content = summarizer_llm.invoke(summarization_messages).content
        print("  Summarization complete.")

        messages_to_remove_count = len(messages_to_summarize) - MESSAGES_TO_KEEP_AFTER_SUMMARY
        ids_to_remove = [m.id for m in messages_to_summarize[:messages_to_remove_count]]
        delete_instructions = [RemoveMessage(id=msg_id) for msg_id in ids_to_remove]
        # print(f"  Generated {len(delete_instructions)} RemoveMessage instructions.") # Reduce noise

        return {"summary": new_summary_content, "messages": delete_instructions}
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return {}


# --- 7.d. Semantic Memory Update Node ---
def update_semantic_memory_node(state: ChatState, config: dict):
    if not langmem_manager:
        return {} # Skip if manager not configured

    print("🧠 Updating semantic memory...")
    try:
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            print("⚠️ Warning: thread_id not found in config. Cannot update semantic memory.")
            return {}

        # Use last user message and AI response for context
        messages_for_extraction = state["messages"][-2:]

        # Invoke the LangMem manager (which uses memory_llm internally)
        langmem_manager.invoke({"messages": messages_for_extraction}, config=config)
        print(f"  Semantic memory processed for thread: {thread_id}")
    except Exception as e:
        print(f"❌ Error updating semantic memory: {e}")
    return {}


# --- 8. Define Conditional Logic ---

def should_summarize(state: ChatState) -> Literal["summarize", "update_memory"]:
    """Decide whether to summarize based on message count."""
    message_count = len(state["messages"])
    if message_count > MAX_MESSAGES_BEFORE_SUMMARY:
        # print("  Decision: Summarize") # Reduce noise
        return "summarize"
    else:
        # print("  Decision: Continue (Update Memory)") # Reduce noise
        return "update_memory"

# --- 9. Build the Graph ---
print("🏗️ Building the LangGraph graph...")
workflow = StateGraph(ChatState)

workflow.add_node("agent", call_llm_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("update_memory", update_semantic_memory_node) # Config passed implicitly by compile

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_summarize,
    {"summarize": "summarize", "update_memory": "update_memory"}
)
workflow.add_edge("summarize", "update_memory")
workflow.add_edge("update_memory", END)

# --- 10. Compile the Graph with Checkpointer ---
print("⚙️ Compiling the graph...")
if mongodb_checkpointer:
    app = workflow.compile(checkpointer=mongodb_checkpointer)
    print("✅ Graph compiled with MongoDB persistence.")
else:
    app = workflow.compile()
    print("⚠️ Graph compiled without persistence (MongoDB connection failed).")


# --- 11. Interactive Terminal Chat Loop ---
if __name__ == "__main__":
    print("\n--- LangGraph Chatbot Terminal ---")
    print(f"Using Main LLM: {MAIN_LLM_MODEL}, Memory LLM: {MEMORY_LLM_MODEL}")
    print("Enter 'quit' or 'exit' to end the chat.")

    # Use a fixed or dynamically generated thread_id
    conversation_thread_id = "terminal_chat_session_002" # Changed ID slightly
    config = {"configurable": {"thread_id": conversation_thread_id}}

    print(f"\n➡️ Starting/Resuming conversation with thread_id: {conversation_thread_id}")
    if mongodb_checkpointer:
        print("(Persistence enabled via MongoDB)")
    else:
        print("(Persistence disabled)")
    if langmem_manager:
        print("(Semantic Memory enabled - In-Memory)")
    else:
        print("(Semantic Memory disabled)")


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
                print(f"🤖 AI: (Received unexpected message type: {type(ai_message)})")


        except KeyboardInterrupt:
            print("\n🤖 AI: Goodbye! (Interrupted by user)")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred during chat loop: {e}")
            # Consider adding more robust error handling or logging here
            # break # Optionally break the loop on error

    # --- Cleanup ---
    if mongo_client:
        print("\nClosing MongoDB client connection.")
        mongo_client.close()

    print("\n--- Chat Session Ended ---")