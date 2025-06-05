#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
telegram_bot.py – Integrates LangChain with MongoDB for conversation
history management and LLM responses using LangGraph summarization pattern.
"""

############################################
# Standard Library Imports
############################################
import os
import datetime
import logging
# import atexit # Keep if needed for LLM client cleanup
import asyncio
import time
import json
import re
# import threading # No longer needed for background summary

############################################
# Third-Party Imports
############################################
import requests
from pymongo import MongoClient
from typing import Sequence, Union, Dict, Any, List, Optional, Literal # Added Literal
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage # Added RemoveMessage

############################################
# Local Imports and Configuration
############################################
from config import (
    # ... existing config imports ...
    MONGO_CONNECTION_STRING, DATABASE_NAME, COLLECTION_NAME, BUSINESS_INFO_COLLECTION
)
from utils.rich_logger import ( setup_logger, ... )
from utils.helpers import refine_ai_response, strip_thinking_tags # Keep these helpers
# ... other necessary utils imports ...

logger = setup_logger(level=logging.INFO)
logger.info("Initializing LangChain integrations with Graph-based Summarization...")

############################################
# LLM Model Definitions and Instance Setup
############################################
PRIMARY_LLM_MODEL = "gpt-4o"
BUSINESS_LLM_MODEL = "gemini-2.0-flash"
USER_REPORT_LLM_MODEL = "gpt-4o"
SUMMARY_LLM_MODEL = "gpt-4o" # Can be same as primary or different

from langchain_openai import ChatOpenAI

# Keep LLM initializations pointing to G4F server
llm = ChatOpenAI(...)
llm_business = ChatOpenAI(...)
user_llm = ChatOpenAI(...)
llm_summary = ChatOpenAI( # LLM specifically for summarization
    base_url="http://localhost:15201/v1",
    model_name=SUMMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Summary LangChain LLM initialized with model: {SUMMARY_LLM_MODEL}.")

# Remove daily_users_report global if generate_user_report saves directly to DB
# daily_users_report = {} # Let's keep it for now if generate_user_report uses it

# ... (atexit cleanup for LLM clients if needed) ...

############################################
# Import Prompt Templates and Helper Texts
############################################
from prompts.prompts import (
    PROMPT_TEMPLATE_TEXT,
    # ... other prompts ...
    SUMMARY_PROMPT_TEXT # Use this or a dedicated prompt for the node
)
# ... (Assign prompts to variables) ...

# Note: LangChain prompt template using MessagesPlaceholder is implicitly handled
# by MessagesState and how we construct messages in the 'agent' node.
# We don't need the explicit ChatPromptTemplate construction here.

############################################
# Type Definitions for AgentState (Updated)
############################################
# MessagesState already includes 'messages'. We add 'summary'.
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    # `messages` attribute is inherited from MessagesState
    summary: Optional[str] # To store the conversation summary
    chat_id: str           # Keep chat_id for context loading
    username: Optional[str] # Keep username for context

############################################
# LangGraph Workflow and Node Definitions (Refactored)
############################################
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver # Keep using MongoDB checkpointer

# --- User Report Generation (Keep as is, likely called elsewhere or via specific command) ---
def generate_user_report(conversation_text: str, chat_id: str):
    # ... (existing logic using user_llm and saving via db_manager) ...
    pass # Placeholder if keeping the function definition

# --- New Summarization Logic (Following the Guide) ---

SUMMARY_THRESHOLD = 10 # Summarize after 10 messages (Human + AI)
MESSAGES_TO_KEEP_AFTER_SUMMARY = 2 # Keep the last N messages after summarizing

def should_continue(state: AgentState) -> Literal["summarize", END]:
    """Return the next node to execute: summarize or end."""
    messages = state['messages']
    # Count only HumanMessage and AIMessage for the threshold
    conversation_messages = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    logger.debug(f"Checking conversation length: {len(conversation_messages)} messages.")
    if len(conversation_messages) > SUMMARY_THRESHOLD:
        logger.info(f"Conversation length ({len(conversation_messages)}) exceeds threshold ({SUMMARY_THRESHOLD}). Routing to summarize.")
        return "summarize"
    else:
        logger.debug("Conversation length within threshold. Routing to END.")
        return END

def summarize_conversation(state: AgentState):
    """Summarizes the conversation history and prepares messages for removal."""
    logger.process_start("Summarizing conversation...")
    current_summary = state.get("summary", "")
    # Get only the user/assistant messages for summarization context
    messages_to_summarize = [m for m in state['messages'] if isinstance(m, (HumanMessage, AIMessage))]
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages_to_summarize])

    # Construct the summary prompt
    if current_summary:
        summary_prompt = (
            f"This is summary of the conversation to date:\n{current_summary}\n\n"
            f"Extend the summary by incorporating the new messages below:\n\n"
            f"{conversation_text}"
        )
    else:
        summary_prompt = (
            f"Create a concise summary of the following conversation:\n\n"
            f"{conversation_text}"
        )
        # Optional: Add SUMMARY_PROMPT_TEXT here for specific instructions if needed
        # summary_prompt += f"\n\n{SUMMARY_PROMPT_TEXT}"

    try:
        start_time = time.time()
        # Invoke the summary LLM
        summary_response = llm_summary.invoke([HumanMessage(content=summary_prompt)])
        new_summary = summary_response.content.strip()
        duration = time.time() - start_time
        logger.info(f"Summarization completed in {duration:.2f}s.")
        log_summarization(logger, conversation_text, new_summary, "Conversation Node")

        # Prepare messages for deletion (all except the last N and non-Human/AI messages)
        messages_to_remove_ids = [
            m.id for m in messages_to_summarize[:-MESSAGES_TO_KEEP_AFTER_SUMMARY]
        ]
        delete_messages = [RemoveMessage(id=msg_id) for msg_id in messages_to_remove_ids]
        logger.info(f"Prepared {len(delete_messages)} messages for removal, keeping last {MESSAGES_TO_KEEP_AFTER_SUMMARY}.")

        logger.process_end("Conversation summarization node finished.")
        # Return the new summary and the list of messages to remove
        return {"summary": new_summary, "messages": delete_messages}

    except Exception as e:
        logger.error(f"Error during summarization node: {e}", exc_info=True)
        logger.process_end("Summarization node failed.")
        # Return empty dict or current state to avoid breaking the graph?
        # Returning empty means summary isn't updated and messages aren't removed on error.
        return {}


############################################
# Agent Function (Main Node - Updated)
############################################
def agent(state: AgentState):
    """Calls the LLM to generate a response, incorporating summary from state."""
    chat_id = state["chat_id"]
    username = state.get("username", "")
    logger.process_start(f"🧠 Processing agent node for chat: {chat_id}")
    # ... (logging username) ...

    # Retrieve context (External info loaded here, summary comes from state)
    ai_tone = ai_tone_map.get(chat_id, "دوستانه")
    business_info = get_user_business_info(chat_id) # Keep loading external info
    messages = state["messages"] # Get messages from state (already potentially trimmed)
    current_summary = state.get("summary") # Get summary from state
    try:
        from utils.date_helpers import get_full_shamsi_date
        current_shamsi_date = get_full_shamsi_date()
    except Exception: current_shamsi_date = "Unknown date"
    user_report = get_user_info(chat_id) # Keep loading external info
    log_user_business_data(logger, chat_id, business_info, user_report)

    # Prepare messages for LLM call
    messages_for_llm = []

    # Add summary as the first system message if it exists
    if current_summary:
        messages_for_llm.append(SystemMessage(content=f"Summary of conversation earlier: {current_summary}"))
        logger.debug("Prepending conversation summary to LLM context.")

    # Add base system prompts (persona, date, business context, tone)
    messages_for_llm.append(SystemMessage(content=prompt_template_text + f"\nCURRENT DATE: {current_shamsi_date}."))
    messages_for_llm.append(SystemMessage(content=f"BUSINESS CONTEXT: {business_info}\nTONE: {ai_tone}"))
    if username:
         messages_for_llm.append(SystemMessage(content=f"The user's name is {username}. Address them directly."))

    # Add the actual conversation messages from the state
    # LangGraph MessagesState ensures these are correctly ordered and potentially trimmed
    messages_for_llm.extend(messages)

    # --- Multimodal Handling (Extract last message for logging/processing) ---
    last_message_content = messages[-1].content if messages else None
    is_multimodal = isinstance(last_message_content, list)
    prompt_input_text = ""
    if is_multimodal:
         for item in last_message_content:
            if item.get("type") == "text": prompt_input_text = item.get("text", "")
         logger.info(f"Processing multimodal input with {len(last_message_content)} components")
    elif isinstance(last_message_content, str):
        prompt_input_text = last_message_content
        logger.info(f"Processing text input: {prompt_input_text[:50]}...")
    # --------------------------------------------------------------------------

    try:
        # Log comprehensive interaction before LLM call
        log_comprehensive_interaction(logger, chat_id, state.get('thread_id', 'N/A'), # Assuming thread_id is available
                                      prompt_template_text, prompt_input_text,
                                      user_report, business_info, ai_tone)
        logger.info("[bold yellow]Invoking LLM...[/bold yellow]")
        start_time = time.time()

        # Invoke the primary LLM
        response = llm.invoke(messages_for_llm)

        end_time = time.time()
        logger.info(f"[bold yellow]LLM invocation completed in {end_time - start_time:.2f} seconds[/bold yellow]")

        ai_message = AIMessage(content=response.content)

        # Post-processing
        if username: ai_message.content = ai_message.content.replace("[نام کاربر]", username)
        ai_message.content = strip_thinking_tags(ai_message.content)
        logger.info("✅ LLM responded successfully")
        log_ai_interaction(logger, prompt_input_text if not is_multimodal else "Multimodal content", response.content, PRIMARY_LLM_MODEL)
        logger.process_end("Agent processing completed successfully")

        # Return dict to update state: just add the new AI message
        # The graph handles adding this to the existing messages list.
        return {"messages": [ai_message]}

    except Exception as e:
        logger.error(f"❌ Error during LLM call: {e}", exc_info=True)
        error_message = AIMessage(content="متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.")
        logger.process_end("Agent processing failed with error")
        # Return error message to be added to state
        return {"messages": [error_message]}

# --- Workflow Initialization (Updated Graph Structure) ---
workflow = StateGraph(AgentState) # Use the updated state definition
logger.info("StateGraph initialized with AgentState structure (incl. summary).")

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("summarize", summarize_conversation)
logger.info("Added 'agent' and 'summarize' nodes to workflow.")

# Define edges
workflow.add_edge(START, "agent") # Entry point remains the agent

# Add conditional edge from agent based on conversation length
workflow.add_conditional_edges(
    "agent", # Starting node
    should_continue, # Function to decide the next path
    {
        "summarize": "summarize", # If should_continue returns "summarize", go to "summarize" node
        END: END                  # If should_continue returns END, go to END
    }
)
logger.info("Added conditional edges from 'agent' to 'summarize' or END.")

# Add edge from summarize node to END
workflow.add_edge("summarize", END)
logger.info("Added edge from 'summarize' to END.")

# --- Compile the graph ---
# Checkpointer remains the same (MongoDBSaver)
# The checkpointer will now handle saving/loading the 'summary' field and
# applying the RemoveMessage updates to the persisted message list.
# memory = MongoDBSaver.from_conn_string(...) # Define checkpointer instance before compiling if needed outside run_agent

############################################
# Main Agent Function: run_agent (Minor Adjustments)
############################################
# from langgraph.checkpoint.mongodb import MongoDBSaver # Import should be here or global
# from langchain.schema import HumanMessage # Ensure import is present

def run_agent(query, chat_id, message_id, username=None):
    """
    Processes a user query using the refactored LangGraph with summarization.
    """
    # ... (logging start) ...
    user_id = str(chat_id)
    is_multimodal = isinstance(query, list) # Check type before formatting

    # Session Management (remains the same)
    session_id = get_session_id(user_id)
    if not session_id:
        session_id = f"{user_id}_{int(datetime.datetime.now().timestamp())}"
        save_session_id(user_id, session_id)
        logger.info(f"Created new session '{session_id}' for chat '{user_id}'.")
    thread_id = session_id
    logger.info(f"🧵 Using thread ID: {thread_id}")

    # Prepare Input Message (remains the same)
    if is_multimodal:
        human_message = HumanMessage(content=query)
    else:
        # Format text query for consistency if needed, or just use raw text
        # The guide doesn't format, let's pass raw query for simplicity
        human_message = HumanMessage(content=query)
        # formatted_query = f"name : {username}\n# user prompt : {query}\n\n" # Optional formatting
        # human_message = HumanMessage(content=formatted_query)

    # Define the input for the graph stream
    # We only need to provide the new message; the checkpointer handles history loading.
    inputs = {"messages": [human_message]}

    # Add chat_id and username to the *initial* config if needed by nodes
    # before the checkpointer loads the full state. For this structure,
    # it's better to load them inside the 'agent' node from the state
    # which the checkpointer provides.
    # However, we need them in the initial state for the *first* run of a thread.
    # Let's check how MongoDBSaver handles initial state creation.
    # It's generally safer to pass necessary IDs via config if nodes need them *before* state loading.
    # But our `agent` node reads `chat_id` and `username` *from the state*.
    # So, we need to ensure the *initial* state passed to `graph.stream` for a *new thread*
    # contains these. LangGraph checkpointers usually merge the input with the loaded state.
    # Let's add them to the initial input payload for safety on the very first message.
    initial_input_payload: AgentState = {
        "messages": [human_message],
        "chat_id": user_id,      # Add chat_id here for first run
        "username": username,    # Add username here for first run
        "summary": None          # Initialize summary as None
    }


    config = {"configurable": {"thread_id": thread_id}}

    try:
        with MongoDBSaver.from_conn_string(
            MONGO_CONNECTION_STRING,
            db_name=DATABASE_NAME,
            collection_name="langgraph_checkpoints" # Ensure this collection exists/is correct
        ) as checkpointer:

            # Compile the graph with the checkpointer
            # It's often better to compile once outside the function if possible,
            # but compiling here ensures the checkpointer is correctly associated.
            graph = workflow.compile(checkpointer=checkpointer)

            logger.info(f"🔄 Running LangGraph workflow for thread {thread_id}")
            final_response = None
            step_count = 0
            start_time_graph = time.time() # Renamed timer

            # Stream the execution
            # The stream gives the output of *each node* as it executes.
            for output in graph.stream(initial_input_payload, config=config, stream_mode="values"):
                # `stream_mode="values"` gives the full state after each step
                step_count += 1
                logger.debug(f"Graph step {step_count} completed. Current state keys: {output.keys()}")

                # The final AI response is generated by the 'agent' node.
                # The state *after* the agent node runs will contain the latest messages.
                # We need to reliably get the *last* AIMessage added.
                if output and "messages" in output and output["messages"]:
                    last_msg = output["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                         # Check if it's not the error message from the agent node itself
                        if "متأسفم، مشکلی در پردازش" not in last_msg.content:
                            final_response = last_msg.content
                            logger.debug(f"Captured potential final response from step {step_count}: {final_response[:50]}...")
                        else:
                            final_response = last_msg.content # Capture error message too
                            logger.warning(f"Captured error message from agent node in step {step_count}")


            elapsed_time = time.time() - start_time_graph
            logger.info(f"Graph execution finished in {elapsed_time:.2f}s")

            # Fallback if no AIMessage was captured
            if not final_response or not final_response.strip():
                # Check the final state directly for the last AI message one more time
                final_state = graph.get_state(config)
                if final_state and final_state.values['messages']:
                     last_msg = final_state.values['messages'][-1]
                     if isinstance(last_msg, AIMessage):
                         final_response = last_msg.content
                         logger.debug("Retrieved final response from get_state.")

            if not final_response or not final_response.strip():
                 final_response = "متأسفم، نتوانستم پاسخ مناسبی بیابم. لطفا دوباره تلاش کنید."
                 logger.warning("No valid response generated, using fallback message.")

            # Refine the final captured response
            refined_response = refine_ai_response(final_response.strip())

            logger.info(f"✅ Generated response in {elapsed_time:.2f}s (Graph runtime)")
            logger.process_end("LangGraph agent execution successful")
            return refined_response

    except Exception as e:
        logger.error(f"❌ Error during LangGraph execution: {e}", exc_info=True)
        logger.process_end("LangGraph agent execution failed")
        return "An error occurred processing your request. Please try again."


############################################
# MongoDB Helper Functions (Cleanup)
############################################

# Keep: get_mongo_collection, get_business_info_collection
# Keep: get_user_business_info, save_user_business_info (used for context/settings)
# Keep: get_history_for_chat (May still be useful for inspection? Or remove if LangGraph handles all history needs)
#       Let's comment it out for now, as LangGraph checkpointer should manage history persistence.
# Keep: get_user_info, save_user_info (if user reports are separate context)
# Keep: summarize_business_info (for business context processing)
# Keep: process_business_info
# Keep: save_message_to_history (Now likely unused - LangGraph handles this via checkpointer. REMOVE)
# Keep: load_user_reports (if needed for initial context)
# Keep: save_session_id, get_session_id (used for thread_id management)

# Remove: background_optimize_memory, MessageCounter (Replaced by graph nodes)
# Remove: save_conversation_summary, get_conversation_summary (Handled by AgentState and checkpointer)

# Adjust save_message_to_history callers if it's removed.
# In bot.py, remove calls to save_message_to_history. LangGraph persists messages automatically.

# Ensure helper functions used by agent node (get_user_business_info, get_user_info) still work.
# These functions might need adjustment if they relied on db_manager functions that are now removed,
# but they seem to use specific collection access or existing db_manager helpers that are kept.

# ... Keep necessary MongoDB helper functions ...

# Example: Removing save_message_to_history related code
# def save_message_to_history(chat_id, role, content):
#     logger.warning("save_message_to_history is deprecated. LangGraph checkpointer handles persistence.")
#     pass # No longer needed


############################################
# Exports (Adjust as needed)
############################################
__all__ = [
    "llm",
    "logger",
    # "prompt", # Implicitly handled
    "help_text_prompt", # Keep if used directly elsewhere
    "welcome_message", # Keep if used directly elsewhere
    # ... other prompts if exported ...
    "process_business_info",
    "summarize_business_info",
    # "get_summarized_history_for_session", # Likely deprecated
    # "get_history_count", # Likely deprecated
    "get_business_info_collection",
    "get_mongo_collection",
    "save_user_business_info",
    "get_user_business_info",
    # "save_message_to_history", # REMOVED
    # "get_history_for_chat", # Likely deprecated
    "get_user_info",
    "save_user_info", # Keep if user report is handled separately
    "run_agent",
    # "daily_users_report", # Keep if loaded and used
    # "load_user_reports", # Keep if needed on startup
    "save_session_id", # Keep for thread management
    "get_session_id", # Keep for thread management
]