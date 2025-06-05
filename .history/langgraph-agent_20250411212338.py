import asyncio
import getpass
import os
import operator
from typing import Annotated, TypedDict, Literal, List, Optional

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END # Use END constant
from langgraph.prebuilt import ToolNode, create_react_agent # Use create_react_agent for simplicity as per guide
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver # Example alternative
from langgraph.checkpoint.memory import MemorySaver # For fallback

# --- MongoDB Persistence Imports ---
from motor.motor_asyncio import AsyncIOMotorClient
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langchain_core.runnables import RunnableConfig

# --- Configuration ---

# 1. API Keys & Environment Setup
def _set_env(var: str):
    """Helper function to set environment variables if not already set."""
    if not os.environ.get(var):
        # Try to get from environment first for non-interactive setup
        env_val = os.environ.get(var)
        if env_val:
            print(f"Using {var} from environment.")
        else:
            # Fallback to prompt if not in environment
            os.environ[var] = getpass.getpass(f"Enter your {var}: ")
    # else:
    #     print(f"{var} is already set.") # Optional confirmation

# Set OpenAI Key (required by ChatOpenAI and create_react_agent)
_set_env("OPENAI_API_KEY")

# Set LangSmith keys for tracing (optional but recommended)
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "LangGraph MongoDB ReAct Chatbot" # Optional project name

# 2. LLM Configuration (Using model specified in the MongoDB guide)
# NOTE: If you strictly need the 'gpt-4o' model from your *first* request,
#       you'd replace 'gpt-4o-mini' here and provide the base_url/api_key
#       as shown previously. However, the MongoDB guide uses standard OpenAI.
llm = ChatOpenAI(
    model_name="gpt-4o-mini", # Model used in the MongoDB guide example
    temperature=0
    # If using a custom endpoint like in the initial request:
    # base_url="http://185.110.190.167:15203/v1",
    # api_key="324" # Manage securely
)
print(f"ChatOpenAI LLM Initialized (Model: {llm.model_name}).")

# 3. MongoDB Configuration
# --- IMPORTANT: Replace with your MongoDB connection string ---
MONGODB_URI = "mongodb://localhost:27017/"
# Example Atlas URI: MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster-url>/?retryWrites=true&w=majority"
DB_NAME = "langgraph_react_agent_db"
COLLECTION_NAME = "checkpoints_async_client"

# --- Tool Definition (as per MongoDB guide) ---
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information for New York City (nyc) or San Francisco (sf)."""
    print(f"--- Tool Called: get_weather(city='{city}') ---")
    if city == "nyc":
        return "It might be cloudy in NYC."
    elif city == "sf":
        return "It's likely sunny in SF."
    else:
        # Although Literal restricts input, good practice to handle unexpected cases
        return f"Weather information for {city} is not available with this tool."
        # raise AssertionError("Unknown city specified to get_weather tool.")


tools = [get_weather]

# --- Main Execution Logic (Async) ---

async def run_chat():
    """Sets up persistence and runs the chat interaction loop."""
    print("\n--- Initializing MongoDB Checkpointer ---")
    mongodb_client_async = None
    checkpointer = None # Default to in-memory if Mongo fails

    try:
        # Use Motor for async MongoDB connection
        print(f"Attempting to connect to MongoDB at: {MONGODB_URI}")
        mongodb_client_async = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000) # Add timeout
        # The ismaster command is cheap and does not require auth.
        await mongodb_client_async.admin.command('ismaster')
        print("Async MongoDB Client Connected Successfully.")

        # Initialize the MongoDB checkpointer using the async client
        checkpointer = AsyncMongoDBSaver(
            client=mongodb_client_async,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME
        )
        print(f"MongoDB Checkpointer configured for db='{DB_NAME}', collection='{COLLECTION_NAME}'.")

    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! MongoDB Connection/Setup Error: {e} !!!")
        print(f"!!! Falling back to In-Memory Checkpointer (state will NOT be persistent across runs) !!!")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        checkpointer = MemorySaver() # Fallback

    # --- Compile the Prebuilt ReAct Agent with the Checkpointer ---
    # The checkpointer handles saving/loading state based on thread_id
    # create_react_agent handles the underlying graph structure (StateGraph, nodes, edges)
    try:
        print("Compiling ReAct agent with checkpointer...")
        app = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
        print("--- LangGraph ReAct Agent Compiled ---")
    except Exception as e:
        print(f"FATAL: Error compiling LangGraph agent: {e}")
        if mongodb_client_async:
            mongodb_client_async.close()
        return # Exit if compilation fails

    # --- Interaction Loop ---
    thread_id = input("Enter a Thread ID (e.g., 'user-session-1') to load/start a conversation: ")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- Starting/Resuming Conversation (Thread ID: {thread_id}) ---")
    print("--- Using Model: gpt-4o-mini ---")
    print("--- Enter 'quit', 'exit', or 'q' to end the chat. ---")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Ending chat. Goodbye!")
            break
        if not user_input.strip():
            continue

        print("\n--- Processing... ---")
        try:
            # Prepare input in the format expected by create_react_agent's state
            input_data = {"messages": [HumanMessage(content=user_input)]}

            # Stream the response chunks
            async for chunk in app.astream(input_data, config=config):
                 # create_react_agent stream yields dicts with the 'messages' key
                ai_message = chunk.get("messages", [])[-1]
                if isinstance(ai_message, AIMessage):
                    if ai_message.content:
                         print(f"AI: {ai_message.content}", end="", flush=True)
                    # You could also print tool calls here if desired
                    # if ai_message.tool_calls:
                    #    print(f"\n   (Tool Call: {ai_message.tool_calls})")
            print() # Newline after streaming finishes for one turn

            # # Optional: Retrieve and print the final state after the turn for debugging
            # try:
            #     final_state_tuple = await checkpointer.aget_tuple(config)
            #     if final_state_tuple:
            #         print("\n--- Current State Snapshot (from Checkpointer) ---")
            #         # print(final_state_tuple.checkpoint) # Print raw checkpoint data
            #         messages = final_state_tuple.checkpoint.get('channel_values', {}).get('messages', [])
            #         print(f"Messages in history: {len(messages)}")
            #         if messages:
            #             messages[-1].pretty_print()
            #     else:
            #          print("\n--- No state found in checkpointer for this thread_id ---")
            #     print("-------------------------------------------")
            # except Exception as get_state_err:
            #     print(f"\nError retrieving state from checkpointer: {get_state_err}")


        except Exception as e:
            print(f"\n--- Error during streaming/invocation: {e} ---")
            # Attempt to show current state even if stream failed mid-way
            # try:
            #     current_state_tuple = await checkpointer.aget_tuple(config)
            #     if current_state_tuple:
            #         print("Last known state before error:")
            #         messages = current_state_tuple.checkpoint.get('channel_values', {}).get('messages', [])
            #         print(f"  Messages Count: {len(messages)}")
            #         if messages: messages[-1].pretty_print()
            # except Exception as se:
            #     print(f"Could not retrieve state after error: {se}")

    # --- Cleanup ---
    if mongodb_client_async:
        mongodb_client_async.close()
        print("\nAsync MongoDB Client Closed.")

# --- Run the Application ---
if __name__ == "__main__":
    try:
        print("Starting Async Chat Application...")
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nChat interrupted by user.")
    except Exception as main_err:
        print(f"\nAn unexpected error occurred: {main_err}")