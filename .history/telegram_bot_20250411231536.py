#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
telegram_bot.py – Integrates LangChain with MongoDB for conversation
history management and LLM responses using a dedicated LangGraph agent.
Provides functions for generating reports, summarizing business info,
and managing conversation state via external graph.

Note: Refactored to use langgraph_agent.py for core graph logic.
"""

# ... (Keep all existing Standard Library, Third-Party, and Local Imports) ...
import os
import datetime
import logging
import atexit
import asyncio
import time
import json
import re
import threading
import requests
from pymongo import MongoClient

from config import (
    TELEGRAM_BOT_TOKEN,
    GOOGLE_API_KEY, # Assuming needed for other parts, otherwise remove
    OPENAI_API_KEY,
    # chat_session_map, # Removed, session managed via thread_id/db
    # business_info_map, # Removed, managed via db_manager
    # ai_tone_map, # Removed, managed via db_manager
    # business_info_update_pending, # State management should be handled differently if needed
    # business_info_mode,
    # ai_tone_update_pending,
    MONGO_CONNECTION_STRING,
    DATABASE_NAME,
    COLLECTION_NAME, # Should be renamed e.g., LEGACY_CHAT_HISTORY_COLLECTION if needed
    BUSINESS_INFO_COLLECTION,
    LANGGRAPH_CHECKPOINT_COLLECTION # Added to config
)

from utils.rich_logger import (
    setup_logger, display_content, log_function, log_telegram_message,
    log_api_interaction, log_summarization, log_ai_interaction,
    log_user_business_data, log_llm_request, log_agent_execution # Keep log_agent_execution
)
logger = setup_logger(level=logging.INFO, logger_name="TelegramBot") # Specify logger name
logger.info("Initializing Telegram Bot with LangGraph Agent Integration...")

# --- LLM Model Definitions and Instance Setup (Keep this section) ---
# It's generally better to initialize these once and make them accessible
# (e.g., via import or dependency injection) to both telegram_bot.py and langgraph_agent.py
PRIMARY_LLM_MODEL = "gpt-4o"
BUSINESS_LLM_MODEL = "gemini-2.0-flash" # Used for summarize_business_info
USER_REPORT_LLM_MODEL = "gpt-4o" # Used for generate_user_report
SUMMARY_LLM_MODEL = "gemini-2.0-flash" # Used by langgraph_agent

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:15201/v1", # Use config/env vars
    model_name=PRIMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Primary LangChain LLM initialized with model: {PRIMARY_LLM_MODEL}.")

llm_business = ChatOpenAI(
    base_url="http://localhost:15201/v1", # Use config/env vars
    model_name=BUSINESS_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Business Info LangChain LLM initialized with model: {BUSINESS_LLM_MODEL}.")

user_llm = ChatOpenAI(
    base_url="http://localhost:15201/v1", # Use config/env vars
    model_name=USER_REPORT_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"User Report LLM initialized with model: {USER_REPORT_LLM_MODEL}.")

llm_summary = ChatOpenAI(
    base_url="http://localhost:15201/v1", # Use config/env vars
    model_name=SUMMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Summary LLM initialized with model: {SUMMARY_LLM_MODEL}.")

# ... (Keep atexit cleanup) ...
atexit.register(lambda: llm.client.close() if hasattr(llm, "client") and callable(getattr(llm, "close", None)) else None)
atexit.register(lambda: llm_business.client.close() if hasattr(llm_business, "client") and callable(getattr(llm_business.client, "close", None)) else None)
atexit.register(lambda: user_llm.client.close() if hasattr(user_llm, "client") and callable(getattr(user_llm.client, "close", None)) else None)
atexit.register(lambda: llm_summary.client.close() if hasattr(llm_summary, "client") and callable(getattr(llm_summary.client, "close", None)) else None)

# --- Import Prompt Templates and Helper Texts (Keep this) ---
from prompts.prompts import (
    # PROMPT_TEMPLATE_TEXT, # Used inside langgraph_agent
    DAILY_TASK_PROMPT,
    # SUMMARY_PROMPT, # Replaced by SUMMARY_PROMPT_TEXT in langgraph_agent
    DAILY_REPORT_PROMPT,
    INSTA_IDEA_PROMPT,
    IMAGE_ANALYZER_PROMPT,
    BUSINESS_INFO_SUMMARY_PROMPT,
    WELCOME_MESSAGE,
    HELP_TEXT,
    USER_REPORT_PROMPT,
    # SUMMARY_PROMPT_TEXT # Used inside langgraph_agent
)

# ... (Remove old LangChain prompt template building if not used elsewhere) ...
# prompt = ChatPromptTemplate.from_messages([...]) # Remove if only used by old agent

# --- Import New LangGraph Agent ---
try:
    from langgraph_agent import app as langgraph_app, AgentState # Import compiled app and state
    if langgraph_app is None:
        raise ImportError("LangGraph app failed to compile.")
    logger.info("Successfully imported LangGraph agent application.")
except ImportError as e:
    logger.error(f"FATAL: Could not import or compile LangGraph agent from langgraph_agent.py: {e}", exc_info=True)
    langgraph_app = None # Set to None to prevent runtime errors

# --- Type Definitions ---
# Keep if needed by other parts of telegram_bot.py, otherwise remove old AgentState
# from typing import Sequence, Union, Dict, Any, List, Optional
# from typing_extensions import TypedDict
# from langchain.schema import SystemMessage, HumanMessage, AIMessage
# class ToolCall(TypedDict): ... # Remove if not used
# class AgentState(TypedDict): ... # REMOVED - Use AgentState from langgraph_agent

# --- REMOVE OLD LangGraph Workflow and Node Definitions ---
# Remove generate_user_report (keep definition if called elsewhere, e.g., by a command)
# Remove optimize_memory
# Remove MessageCounter class and instance
# Remove agent function
# Remove route_tool function
# Remove workflow = StateGraph(...) definitions and edge definitions

# --- Keep User Report Generation function if it's triggered elsewhere ---
# Global dictionary for user reports - maybe manage via db_manager?
daily_users_report = {} # Keep for now if generate_user_report uses it

# --- User Report Generation (Keep if needed, ensure db saving is correct) ---
def generate_user_report(conversation_text: str, chat_id: str):
    # This function seems disconnected from the main flow in the original code.
    # If it needs to run, decide *when* (e.g., triggered by command, schedule).
    # It uses user_llm.
    try:
        # ... (keep existing logic) ...
        # Ensure db_manager is used correctly for saving
        from db_manager import save_user_info # Use db_manager function
        # The original code saved to daily_users_report dict AND db. Ensure consistency.
        # Maybe save_user_info should update the dict too, or dict is loaded at start.
        # ... (rest of the function) ...
        # Update business_info collection - is this the right place?
        # Consider consolidating user report saving logic.
         # The save_user_info in db_manager might be sufficient? Check its implementation.
        # The original save_user_info saves to 'settings' collection, not 'business_info'.
        # Let's stick to saving in the business_info collection as originally done here.
        try:
            collection = get_business_info_collection() # Use existing helper
            collection.update_one(
                {"chat_id": str(chat_id)}, # Ensure chat_id is string if needed
                {"$set": {"user_report": report_text, "updated_at": datetime.datetime.utcnow()}},
                upsert=True
            )
            logger.info(f"[bold green]User report saved/updated in business_info collection for chat {chat_id}[/bold green]")
            daily_users_report[str(chat_id)] = report_text # Update in-memory cache
        except Exception as db_error:
            logger.error(f"Failed to save user report to business_info collection: {db_error}")

    except Exception as e:
        logger.error(f"Error generating user report for chat {chat_id}: {e}", exc_info=True)


# --- Main Agent Function: run_agent (Modified) ---

# Import LangChain schema for message creation
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

@log_function(logger)
def run_agent(query: Union[str, list], chat_id: Union[str, int], message_id: Optional[int] = None, username: Optional[str] = None):
    """
    Processes a user query using the external LangGraph agent.
    Handles conversation state and history via the checkpointer.
    """
    if langgraph_app is None:
        logger.error("LangGraph application is not available. Cannot process request.")
        return "متأسفم، سیستم پردازش پیام در حال حاضر در دسترس نیست."

    user_id = str(chat_id) # Ensure consistent string ID
    is_multimodal = isinstance(query, list)

    # Use log_agent_execution from utils
    log_agent_execution(logger, user_id, None, username, query, is_multimodal)
    logger.process_start(f"Starting LangGraph agent execution for chat: {user_id}")
    start_time = time.time()

    # --- Session/Thread ID Management ---
    # LangGraph uses thread_id for checkpoints. Use chat_id directly or a derived ID.
    # Using user_id (chat_id as string) directly is simplest if sessions are 1:1 with chats.
    thread_id = user_id
    logger.info(f"🧵 Using Thread ID for LangGraph: {thread_id}")

    # --- Prepare Input for LangGraph ---
    # The graph state includes chat_id and username, passed via config or initial state.
    # The input message needs to be created.
    if is_multimodal:
        # Assume multimodal query is already in LangChain format [{type: "text", text: "..."}, {type: "image_url", ...}]
        input_message = HumanMessage(content=query)
        logger.debug("Created multimodal HumanMessage for graph input.")
    else:
        # Just pass the raw text query. The graph's call_llm node handles context/prompting.
        input_message = HumanMessage(content=query)
        logger.debug("Created text HumanMessage for graph input.")

    # Define the input dictionary for the graph stream
    # We only need to provide the new message(s). Checkpointer loads the rest.
    # Pass chat_id and username in the config for the nodes to access easily.
    inputs = {"messages": [input_message]}

    # Config for checkpointer and potentially passing context to nodes if needed
    # The state definition in langgraph_agent now includes chat_id and username,
    # but how are they set initially or updated?
    # Option 1: Pass them in config and let nodes read from state['chat_id'] etc.
    # Option 2: Update the state explicitly before calling stream (less common with checkpoints)
    # Let's rely on the nodes fetching context using chat_id from the state,
    # assuming the state is correctly loaded/initialized by the checkpointer/graph.
    # We MUST ensure chat_id and username are part of the state saved by the checkpointer.
    # Let's pass them in the configurable part of the input, which can be used to update state fields.
    # state_updates = {"chat_id": user_id, "username": username} # This syntax might be for modifying existing state
    # Check LangGraph docs for how to pass initial/static values into the state via invoke/stream.
    # Often, config is just for the checkpointer thread_id.
    # Let's assume AgentState is correctly loaded by the checkpointer.
    # If it's the *first* run for a thread_id, how does AgentState get chat_id/username?
    # We might need to modify the graph entry point or how state is initialized.
    # Alternative: Pass them in the input dict if the state definition expects them directly.
    # Let's modify AgentState to accept them in the input dict if not present.

    # Revised Input preparation: include chat_id and username directly if AgentState requires them.
    graph_input: AgentState = {
        "messages": [input_message],
        "chat_id": user_id, # Provide chat_id for the state
        "username": username # Provide username for the state
    }

    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Invoking LangGraph app. Config: {config}")
    final_response_content = None
    full_final_state = None

    try:
        # Stream the graph execution
        logger.info(f"[bold blue]=== LangGraph Stream Start (Thread: {thread_id}) ===[/bold blue]")
        for event in langgraph_app.stream(graph_input, config=config, stream_mode="values"):
             # stream_mode="values" gives the full state after each step
             logger.debug(f"Graph stream event: {event}")
             full_final_state = event # Keep the latest full state

             # Extract the latest AIMessage content as the potential response
             if isinstance(event, AgentState) and event['messages']:
                 last_message = event['messages'][-1]
                 if isinstance(last_message, AIMessage):
                     final_response_content = last_message.content
                     logger.info(f"Intermediate AI Message found: '{final_response_content[:50]}...'")

        logger.info(f"[bold blue]=== LangGraph Stream End (Thread: {thread_id}) ===[/bold blue]")

        # Check if we got a response
        if not final_response_content or not final_response_content.strip():
            # If stream finished but no AIMessage found, check the last state again
            if full_final_state and full_final_state['messages']:
                last_message = full_final_state['messages'][-1]
                if isinstance(last_message, AIMessage):
                    final_response_content = last_message.content
                    logger.info("Final AI message extracted from last state.")

            if not final_response_content or not final_response_content.strip():
                 logger.warning("No valid AI response content found after graph execution. Using fallback.")
                 final_response_content = "متأسفم، نتوانستم پاسخ مناسبی بیابم. لطفا دوباره تلاش کنید."

        # --- Refinement (Optional - might be done inside graph already) ---
        # refined_response = refine_ai_response(final_response_content.strip()) # refine_ai_response is now called inside call_llm
        refined_response = final_response_content.strip() # Assume refinement happened in graph node

        elapsed_time = time.time() - start_time
        logger.info(f"✅ LangGraph execution finished in {elapsed_time:.2f}s")
        logger.process_end("LangGraph agent execution successful")
        return refined_response

    except Exception as e:
        logger.error(f"❌ Error during LangGraph execution: {e}", exc_info=True)
        logger.process_end("LangGraph agent execution failed")
        # Provide a user-friendly error message
        return "متأسفم، خطایی در پردازش درخواست شما رخ داد. لطفا بعدا دوباره تلاش کنید."


# --- MongoDB Helper Functions (Keep relevant ones, delegate to db_manager where possible) ---

# Keep get_business_info_collection if used directly (e.g., by generate_user_report)
def get_business_info_collection():
    # Consider moving connection logic fully into db_manager
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    return db[BUSINESS_INFO_COLLECTION]

# Delegate to db_manager for consistency
def get_user_business_info(chat_id: str) -> str:
    from db_manager import get_business_info as db_get_business_info
    return db_get_business_info(str(chat_id)) # Ensure db_manager handles chat_type if needed

def save_user_business_info(chat_id: str, info: str):
    from db_manager import save_business_info as db_save_business_info
    db_save_business_info(str(chat_id), info) # Ensure db_manager handles chat_type if needed

# History retrieval is now handled by the LangGraph checkpointer mostly.
# Keep get_history_for_chat if needed for other purposes (e.g., manual display)
# but it's NOT used by run_agent anymore.
# Remove get_history_count, get_summarized_history_for_session if not used.

# User info retrieval delegated to db_manager
def get_user_info(chat_id: str, date=None):
     from db_manager import get_user_info as db_get_user_info
     # Ensure db_manager's get_user_info matches the expected logic (date handling etc.)
     return db_get_user_info(str(chat_id)) # Pass date if db_manager supports it

# --- Business Info Summarization (Keep if triggered by commands/elsewhere) ---
from langchain.chains.summarize import load_summarize_chain # Keep import if needed
def summarize_business_info(raw_text: str) -> str:
    # This uses llm_business
    # ... (keep existing implementation) ...
    # Ensure logging uses the main logger
    try:
        logger.process_start("Summarizing business info...")
        # Use BUSINESS_INFO_SUMMARY_PROMPT from prompts
        prompt_text = BUSINESS_INFO_SUMMARY_PROMPT.format(raw_text=raw_text)
        start_time = time.time()
        # Assuming llm_business expects list of messages
        response = llm_business.invoke([HumanMessage(content=prompt_text)])
        duration = time.time() - start_time
        result = response.content.strip()
        # Use log_summarization from utils
        log_summarization(logger, raw_text, result, "Business Info")
        logger.process_end(f"Business info summarized in {duration:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error summarizing business info: {e}", exc_info=True)
        logger.process_end("Business info summarization failed")
        return raw_text # Return original text on error

# Keep process_business_info if it does more than strip()
def process_business_info(info_text, chat_id):
    # Currently just strips, might have more logic later
    processed = info_text.strip()
    # Maybe save directly here?
    # save_user_business_info(str(chat_id), processed)
    return processed

# --- Message Saving (Handled by LangGraph Checkpointer) ---
# Remove save_message_to_history function, as checkpointer manages state persistence.

# --- User Report Loading (Keep if generate_user_report uses daily_users_report dict) ---
def load_user_reports():
    # Loads reports into the daily_users_report dictionary
    try:
        collection = get_business_info_collection()
        # Ensure query matches structure (using str(chat_id) if keys are strings)
        reports = collection.find({}, {"chat_id": 1, "user_report": 1})
        count = 0
        loaded_ids = set()
        for report in reports:
            # Handle potential missing fields gracefully
            chat_id = report.get("chat_id")
            user_report = report.get("user_report")
            if chat_id and user_report:
                daily_users_report[str(chat_id)] = user_report
                loaded_ids.add(str(chat_id))
                count += 1
        logger.info(f"[bold green]Loaded {count} user reports from database into memory cache.[/bold green]")
        # Optional: Clean up dict entries for users no longer in DB?
    except Exception as e:
        logger.error(f"Error loading user reports into memory cache: {e}", exc_info=True)

# Load reports at startup
load_user_reports()

# --- Remove Session ID Management (Handled by LangGraph Thread ID) ---
# Remove save_session_id and get_session_id functions. The thread_id (user_id) is used directly.

# --- Exports ---
# Update exports based on what's still relevant and provided by this module
__all__ = [
    # LLM Instances (if needed by other potential modules)
    "llm",
    "llm_business",
    "user_llm",
    "llm_summary",
    # Logger
    "logger",
    # Prompts/Messages (Export only those used by external modules/handlers)
    "help_text_prompt", # Example, keep HELP_TEXT if used directly
    "welcome_message",
    "image_analyzer_prompt",
    "insta_idea_prompt",
    "daily_report_prompt",
    "daily_task_prompt",
    # Core Agent Function
    "run_agent",
    # Business Info Functions (if used by handlers)
    "process_business_info",
    "summarize_business_info",
    "save_user_business_info", # Expose DB function wrapper
    "get_user_business_info",  # Expose DB function wrapper
    # User Report Functions / Data (if used by handlers)
    "generate_user_report",
    "daily_users_report", # Expose the cache if needed directly
    "load_user_reports",
    # DB Manager Functions (Expose wrappers if needed)
    "get_user_info", # Expose DB function wrapper
    # Potentially export db_manager instance if needed directly
    # from db_manager import db_manager
]

# --- (Keep the rest of your Telegram bot handlers, __main__ block, etc.) ---
# Example:
# if __name__ == "__main__":
#     # Initialize Telegram bot, add handlers, start polling/webhook
#     logger.info("Starting Telegram Bot...")
#     # application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
#     # Add handlers that call run_agent, generate_user_report, etc.
#     # application.run_polling()