#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
telegram_bot.py – Integrates LangChain with MongoDB for conversation
history management and LLM responses. Provides functions for generating
reports, summarizing business info, and managing conversation state.

Note: This file has been refactored for clarity and structure.
All core functionalities remain unchanged.
"""

############################################
# Standard Library Imports
############################################
import os
import datetime
import logging
import atexit
import asyncio
import time
import json
import re
import threading

############################################
# Third-Party Imports
############################################
import requests
from pymongo import MongoClient

############################################
# Local Imports and Configuration
############################################
from config import (
    TELEGRAM_BOT_TOKEN,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    chat_session_map, # Keep for now, might be used elsewhere or can be refactored later if fully unused
    ai_tone_map, # Keep for AI tone state
    ai_tone_update_pending, # Keep for AI tone state
    MONGO_CONNECTION_STRING, # Keep for LangGraph checkpointer
    DATABASE_NAME, # Keep for LangGraph checkpointer
    COLLECTION_NAME # Keep for LangGraph checkpointer
    # BUSINESS_INFO_COLLECTION is now managed within db_manager
)

# Import db_manager functions
import db_manager

from utils.rich_logger import (
    setup_logger, display_content, log_function, log_telegram_message,
    log_api_interaction, log_summarization, log_ai_interaction,
    log_user_business_data, log_llm_request, log_agent_execution
)
logger = setup_logger(level=logging.INFO)
logger.info("Initializing LangChain integrations...")

############################################
# LLM Model Definitions and Instance Setup
############################################
PRIMARY_LLM_MODEL = "gpt-4o"
BUSINESS_LLM_MODEL = "gemini-2.0-flash"
USER_REPORT_LLM_MODEL = "gpt-4o"
SUMMARY_LLM_MODEL = "gpt-4o"  # Change if needed

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=PRIMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Primary LangChain LLM initialized with model: {PRIMARY_LLM_MODEL}.")

llm_business = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=BUSINESS_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Secondary LangChain LLM for business info summarization initialized with model: {BUSINESS_LLM_MODEL}.")

user_llm = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=USER_REPORT_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"User LLM initialized with model: {USER_REPORT_LLM_MODEL}.")

llm_summary = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=SUMMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Summary LLM initialized with model: {SUMMARY_LLM_MODEL}.")

# Global dictionary for user reports
daily_users_report = {}

atexit.register(lambda: llm.client.close() if hasattr(llm, "client") and callable(getattr(llm, "close", None)) else None)
atexit.register(lambda: llm_business.client.close() if hasattr(llm_business, "client") and callable(getattr(llm_business.client, "close", None)) else None)

############################################
# Import Prompt Templates and Helper Texts
############################################
from prompts.prompts import (
    PROMPT_TEMPLATE_TEXT, 
    DAILY_TASK_PROMPT,
    SUMMARY_PROMPT,
    DAILY_REPORT_PROMPT,
    INSTA_IDEA_PROMPT,
    IMAGE_ANALYZER_PROMPT,
    BUSINESS_INFO_SUMMARY_PROMPT,
    WELCOME_MESSAGE,
    HELP_TEXT,
    USER_REPORT_PROMPT,
    SUMMARY_PROMPT_TEXT
)

prompt_template_text = PROMPT_TEMPLATE_TEXT
daily_task_prompt = DAILY_TASK_PROMPT
summary_prompt = SUMMARY_PROMPT
daily_report_prompt = DAILY_REPORT_PROMPT
insta_idea_prompt = INSTA_IDEA_PROMPT
image_analyzer_prompt = IMAGE_ANALYZER_PROMPT
business_info_summary_prompt = BUSINESS_INFO_SUMMARY_PROMPT  
welcome_message = WELCOME_MESSAGE
help_text_prompt = HELP_TEXT

############################################
# Build LangChain Prompt Template
############################################
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template_text),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("User Input: {input}\nCurrent AI Tone: {ai_tone}\nBusiness Context: {business_info}")
])
logger.info("LangChain prompt template created.")

############################################
# Type Definitions for AgentState
############################################
from typing import Sequence, Union, Dict, Any, List, Optional
from typing_extensions import TypedDict
from langchain.schema import SystemMessage, HumanMessage, AIMessage

class ToolCall(TypedDict):
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: str

class AgentState(TypedDict):
    messages: Sequence[Union[SystemMessage, HumanMessage, AIMessage]]
    tool_calls: List[ToolCall]
    requires_tool: bool
    current_tool: Optional[str]
    chat_id: str
    username: Optional[str]

############################################
# LangGraph Workflow and Node Definitions
############################################
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver

llm_instance = llm  # For clarity in later calls

# --- User Report Generation ---
def generate_user_report(conversation_text: str, chat_id: str):
    try:
        user_lines = [line for line in conversation_text.splitlines() if line.startswith("HumanMessage:")]
        user_prompt = "\n".join(user_lines)
        prev_report = daily_users_report.get(chat_id, "")
        combined_context = user_prompt + ("\nPrevious User Report:\n" + prev_report if prev_report else "")
        report_prompt = USER_REPORT_PROMPT.format(conversation_text=combined_context)
        response = user_llm.invoke([HumanMessage(content=report_prompt)])
        report_text = response.content.strip()
        daily_users_report[chat_id] = report_text # Keep in-memory cache if needed

        # Save user report using db_manager
        # Ensure chat_id is integer
        try:
            int_chat_id = int(chat_id)
            success = db_manager.save_business_info(int_chat_id, {"user_report": report_text})
            if success:
                logger.info(f"[bold green]User report saved/updated via db_manager for chat {chat_id}[/bold green]")
            else:
                logger.error(f"Failed to save user report via db_manager for chat {chat_id}")
        except ValueError:
             logger.error(f"Invalid chat_id format for db_manager: {chat_id}")
        except Exception as db_error:
            logger.error(f"Error saving user report via db_manager: {db_error}")

        logger.info(f"[bold blue]User Report for chat {chat_id} generated:[/bold blue] {report_text[:100]}...")
    except Exception as e:
        logger.error(f"Error generating user report for chat {chat_id}: {e}", exc_info=True)

# --- Conversation Summarization Functions ---
def optimize_memory(state: AgentState) -> AgentState:
    THRESHOLD = 10  # Summarize if there are more than 10 human/AI messages
    logger.info(f"Entering optimize_memory for chat: {state['chat_id']}")
    conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    if len(conv_messages) <= THRESHOLD:
        logger.info(f"Conversation length ({len(conv_messages)}) does not exceed threshold ({THRESHOLD}); skipping summarization.")
        return state

    existing_summary = state.get("summary", "")
    if existing_summary:
        summary_prompt_text = f"This is the summary so far: {existing_summary}\nExtend the summary by incorporating the new messages below:"
    else:
        summary_prompt_text = "Create a summary of the conversation above:"
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conv_messages])
    full_prompt = f"{conversation_text}\n\n{summary_prompt_text}"
    
    logger.process_start("Starting conversation summarization")
    try:
        start_time = time.time()
        summary_response = llm_summary.invoke([HumanMessage(content=full_prompt)])
        duration = time.time() - start_time
        new_summary = summary_response.content.strip()
        logger.info(f"Summarization completed in {duration:.2f}s: {new_summary[:100]}...")
        log_summarization(logger, conversation_text, new_summary, "Conversation")
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        logger.process_end("Summarization failed")
        return state

    new_system = SystemMessage(content=f"[CONVERSATION SUMMARY]: {new_summary}")
    new_messages = [new_system] + state["messages"][-2:]
    logger.info(f"Trimmed conversation: kept {len(new_messages)} messages (summary + last 2 messages)")
    logger.process_end("Conversation summarization and trimming completed")
    state["summary"] = new_summary
    return {
        "messages": new_messages,
        "tool_calls": state.get("tool_calls", []),
        "requires_tool": state.get("requires_tool", False),
        "current_tool": state.get("current_tool", None),
        "chat_id": state["chat_id"],
        "username": state.get("username", None),
        "summary": new_summary,
    }

class MessageCounter:
    def __init__(self):
        self._counters = {}
        self._lock = threading.Lock()

    def increment_and_check(self, chat_id: str) -> bool:
        with self._lock:
            self._counters[chat_id] = self._counters.get(chat_id, 0) + 1
            if self._counters[chat_id] >= 5:  # Summarize every 5 messages
                self._counters[chat_id] = 0
                return True
            return False

    def reset(self, chat_id: str):
        with self._lock:
            self._counters[chat_id] = 0

message_counter = MessageCounter()

def background_optimize_memory(state: AgentState):
    """Runs memory optimization in background after 5 messages."""
    chat_id = state['chat_id']
    if not message_counter.increment_and_check(chat_id):
        return state

    def run_summarization():
        try:
            conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
            existing_summary = state.get("summary", "")
            if existing_summary:
                summary_prompt_text = f"This is the summary so far: {existing_summary}\nExtend the summary by incorporating the new messages below:"
            else:
                summary_prompt_text = "Create a summary of the conversation above:"
            conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conv_messages])
            full_prompt = f"{conversation_text}\n\n{summary_prompt_text}"
            
            logger.process_start("Starting background conversation summarization")
            start_time = time.time()
            summary_response = llm_summary.invoke([HumanMessage(content=full_prompt)])
            new_summary = summary_response.content.strip()
            # Save summary using db_manager under business_info
            try:
                int_chat_id = int(chat_id)
                success = db_manager.save_business_info(int_chat_id, {"conversation_summary": new_summary})
                if success:
                    logger.info(f"Background conversation summary saved via db_manager for chat {chat_id}")
                else:
                    logger.error(f"Failed to save background summary via db_manager for chat {chat_id}")
            except ValueError:
                logger.error(f"Invalid chat_id format for db_manager: {chat_id}")
            except Exception as e_save:
                 logger.error(f"Error saving background summary via db_manager: {e_save}")

            duration = time.time() - start_time
            logger.info(f"Background summarization completed in {duration:.2f}s")
            log_summarization(logger, conversation_text, new_summary, "Background Conversation")
        except Exception as e:
            logger.error(f"Background summarization error: {e}", exc_info=True)
            message_counter.reset(chat_id)
    threading.Thread(target=run_summarization, daemon=True).start()
    return state

############################################
# Agent Function
############################################
def agent(state: AgentState):
    """
    Calls the LLM to generate a response from the formatted conversation.
    Handles both text-only and multimodal inputs.
    """
    chat_id = state["chat_id"]
    username = state.get("username", "")
    logger.process_start(f"🧠 Processing agent node for chat: {chat_id}")
    logger.info(f"Username: {username if username else 'Not provided'}")
    
    ai_tone = ai_tone_map.get(chat_id, "دوستانه") # Keep using ai_tone_map for now
    # Load business info using db_manager
    try:
        int_chat_id = int(chat_id)
        business_info_data = db_manager.load_business_info(int_chat_id)
        business_info = business_info_data.get("business_info", "") if business_info_data else ""
        user_report = business_info_data.get("user_report", "") if business_info_data else "" # Also load report here
    except ValueError:
        logger.error(f"Invalid chat_id format for db_manager: {chat_id}")
        business_info = ""
        user_report = ""
    except Exception as e_load:
        logger.error(f"Error loading business info/report via db_manager: {e_load}")
        business_info = ""
        user_report = ""

    messages = state["messages"]
    try:
        from utils.date_helpers import get_full_shamsi_date
        current_shamsi_date = get_full_shamsi_date()
        logger.info(f"📅 Current Shamsi date: {current_shamsi_date}")
    except Exception as e:
        current_shamsi_date = "Unknown date"
        logger.error(f"❌ Error getting Shamsi date: {e}")

    # user_report is now loaded above with business_info
    log_user_business_data(logger, chat_id, business_info, user_report)

    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            formatted_messages.append({"role": "system", "content": msg.content})
    
    is_multimodal = False
    prompt_input = ""
    multimodal_content = None
    if formatted_messages and "content" in formatted_messages[-1]:
        last_content = formatted_messages[-1]["content"]
        if isinstance(last_content, list):
            is_multimodal = True
            multimodal_content = last_content
            for item in last_content:
                if item.get("type") == "text":
                    prompt_input = item.get("text", "")
                    break
            logger.info(f"Processing multimodal input with {len(multimodal_content)} components")
        else:
            prompt_input = last_content
            logger.info(f"Processing text input: {prompt_input[:50]}{'...' if len(prompt_input) > 50 else ''}")
    
    try:
        if is_multimodal:
            logger.info("🖼️ Preparing multimodal message structure")
            messages_for_llm = [
                {"role": "system", "content": prompt_template_text + (f"\n\nADDRESS USER: Always address the user directly as {username}" if username else "") +
                                             f"\n\nCURRENT DATE: Today's date in Iranian Shamsi calendar is {current_shamsi_date}."},
                {"role": "system", "content": f"BUSINESS CONTEXT: {business_info}\nTONE: {ai_tone}"} # Added missing system message for business context/tone
            ]
            for msg in formatted_messages[:-1]:
                messages_for_llm.append(msg)
            messages_for_llm.append({"role": "user", "content": multimodal_content})
            from utils.rich_logger import log_comprehensive_interaction
            # Use the user_report loaded earlier
            # Session ID is not directly relevant here, pass None or empty string
            log_comprehensive_interaction(logger, chat_id, "", prompt_template_text, "Multimodal input", user_report, "N/A", ai_tone)
            logger.debug(f"Sending request to LLM with {len(messages_for_llm)} messages (multimodal)")
            response = llm.invoke(messages_for_llm)
        else:
            # user_info (report) is loaded earlier
            # session_id retrieval needs update to use db_manager
            try:
                # Use the int_chat_id already defined above in the function
                session_id = get_session_id(int_chat_id) # Call updated get_session_id (will be updated later)
                if not session_id:
                     session_id = "N/A" # Fallback if not found
            except NameError: # Handle case where int_chat_id might not be set due to earlier error
                 logger.error(f"Cannot get session_id, int_chat_id not defined for chat: {chat_id}")
                 session_id = "N/A"
            except ValueError: # Should not happen if int_chat_id is used, but keep for safety
                 logger.error(f"Invalid chat_id format for get_session_id: {chat_id}")
                 session_id = "N/A"
            except Exception as e_sess:
                 logger.error(f"Error getting session_id via db_manager: {e_sess}")
                 session_id = "N/A"

            messages_for_llm = [
                {"role": "system", "content": prompt_template_text},
                {"role": "system", "content": f"BUSINESS CONTEXT: {business_info}\n\nCURRENT DATE: Today's date in Iranian Shamsi calendar is {current_shamsi_date}."}
            ]
            for msg in formatted_messages[:-1]:
                messages_for_llm.append(msg)
            if username:
                messages_for_llm.append({
                    "role": "system",
                    "content": f"The user's name is {username}. Address them directly as {username} without using placeholders."
                })
            messages_for_llm.append({"role": "user", "content": f"name : {username}\n# user prompt : {prompt_input}\n\n"})
            from utils.rich_logger import log_comprehensive_interaction
            # Use the user_report loaded earlier and the retrieved session_id
            log_comprehensive_interaction(logger, chat_id, session_id, prompt_template_text, f"name : {username}\n# user prompt : {prompt_input}\n\n", user_report, "N/A", ai_tone)
            logger.debug(f"Sending text-only request to LLM with {len(messages_for_llm)} messages")
            response = llm.invoke(messages_for_llm)
        
        ai_message = AIMessage(content=response.content)
        if username:
            for placeholder in ["[نام کاربر]", "[name]", "[نام]"]:
                ai_message.content = ai_message.content.replace(placeholder, username)
        if state["messages"] and isinstance(state["messages"][-1], AIMessage):
            last_response = state["messages"][-1].content.strip()
            current_response = ai_message.content.strip()
            if last_response == current_response:
                ai_message.content += " (به‌روز شده)"
        from utils.helpers import strip_thinking_tags
        ai_message.content = strip_thinking_tags(ai_message.content)
        if "<think>" in response.content and "</think>" in response.content:
            logger.debug("Detected thinking tags in response, stripping them out")
            logger.debug(f"Original length: {len(response.content)}, New length: {len(ai_message.content)}")
        logger.info("✅ LLM responded successfully")
        log_ai_interaction(logger, prompt_input if not is_multimodal else "Multimodal content with image", response.content, "GPT-4o")
        logger.process_end("Agent processing completed successfully")
        
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [ai_message]
        threading.Thread(target=background_optimize_memory, args=(new_state.copy(),), daemon=True).start()
        
        return new_state
    except Exception as e:
        logger.error(f"❌ Error during LLM call: {e}", exc_info=True)
        error_message = AIMessage(content="متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.")
        logger.process_end("Agent processing failed with error")
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [error_message]
        return new_state

def route_tool(state: AgentState) -> str:
    logger.debug(f"Routing from agent node to optimize_memory for chat: {state['chat_id']}")
    return "optimize_memory"

# --- Workflow Initialization ---
workflow = StateGraph(AgentState)
logger.info("StateGraph initialized with AgentState structure")
workflow.add_node("agent", agent)                 
workflow.add_node("optimize_memory", optimize_memory)
logger.info("Added agent and optimize_memory nodes to workflow")
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_tool, {"optimize_memory": "optimize_memory"})
workflow.add_edge("optimize_memory", END)
logger.info("Workflow graph edges defined")

############################################
# Main Agent Function: run_agent
############################################
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain.schema import HumanMessage

def run_agent(query, chat_id, message_id, username=None):
    """
    Processes a user query using LangGraph with MongoDB checkpointer integration.
    Handles both text and multimodal inputs.
    """
    is_multimodal = isinstance(query, list)
    from utils.rich_logger import log_agent_execution
    log_agent_execution(logger, chat_id, None, username, query, is_multimodal)
    logger.process_start(f"Starting LangGraph agent for chat: {chat_id}")
    start_time = time.time()
    # Add the missing import for refine_ai_response
    from utils.helpers import refine_ai_response

    # Prepare chat_id variations
    user_id = str(chat_id) # String version for thread_id and current session functions
    try:
        int_chat_id = int(chat_id) # Integer version for future db_manager calls
    except ValueError:
        logger.error(f"Invalid chat_id format in run_agent: {chat_id}. Cannot proceed.")
        return "An internal error occurred (invalid chat ID format)."

    # Call existing session functions with string user_id for now
    # These functions (get_session_id, save_session_id) will be updated later to use db_manager and int_chat_id
    session_id = get_session_id(user_id)
    if not session_id:
        session_id = f"{user_id}_{int(datetime.datetime.now().timestamp())}"
        save_session_id(user_id, session_id) # Uses old function expecting string
        logger.info(f"Created new session '{session_id}' for chat '{user_id}'.") # Log with string id

    thread_id = session_id # LangGraph thread_id uses the session_id string
    logger.info(f"🧵 Using thread ID: {thread_id}")
    
    logger.debug("Loading conversation history from MongoDB")
    history_obj = get_history_for_chat(user_id, session_id)
    history_messages = history_obj.messages if history_obj and hasattr(history_obj, 'messages') else []
    logger.info(f"Loaded {len(history_messages)} messages from history")
    
    formatted_query = f"name : {username}\n# user prompt : {query}\n\n"
    if isinstance(query, list):
        from langchain.schema import HumanMessage
        human_message = HumanMessage(content=query)
        logger.debug("Created multimodal HumanMessage")
    else:
        from langchain.schema import HumanMessage
        human_message = HumanMessage(content=formatted_query)
        logger.debug("Created text HumanMessage with formatted prompt")
    
    state_messages = history_messages + [human_message]
    inputs: AgentState = {
        "messages": state_messages,
        "tool_calls": [],
        "requires_tool": False,
        "current_tool": None,
        "chat_id": user_id, # Pass string user_id to state for consistency with LangGraph thread_id
        "username": username
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        with MongoDBSaver.from_conn_string(
            MONGO_CONNECTION_STRING, 
            db_name=DATABASE_NAME, 
            collection_name="langgraph_checkpoints"
        ) as checkpointer:
            graph = workflow.compile(checkpointer=checkpointer)
            logger.info(f"🔄 Running LangGraph workflow for thread {thread_id}")
            final_response = None
            step_count = 0
            for output in graph.stream(inputs, config=config):
                step_count += 1
                logger.debug(f"Processing step {step_count} of graph execution")
                if "optimize_memory" in output and "messages" in output["optimize_memory"]:
                    most_recent_msg = output["optimize_memory"]["messages"][-1]
                    if isinstance(most_recent_msg, AIMessage):
                        final_response = most_recent_msg.content
                        logger.debug("Got response from optimize_memory node")
                elif "agent" in output and "messages" in output["agent"] and final_response is None:
                    most_recent_msg = output["agent"]["messages"][-1]
                    if isinstance(most_recent_msg, AIMessage):
                        final_response = most_recent_msg.content
                        logger.debug("Got response from agent node")
            if not final_response or not final_response.strip():
                final_response = "متأسفم، نتوانستم پاسخ مناسبی بیابم. لطفا دوباره تلاش کنید."
                logger.warning("No valid response generated, using fallback message")
            refined_response = refine_ai_response(final_response.strip())
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Generated response in {elapsed_time:.2f}s")
            logger.process_end("LangGraph agent execution successful")
            return refined_response
    except Exception as e:
        logger.error(f"❌ Error during LangGraph execution: {e}", exc_info=True)
        logger.process_end("LangGraph agent execution failed")
        return "An error occurred processing your request. Please try again."

############################################
# MongoDB Helper Functions for Chat History & Business Info
############################################
def get_mongo_collection():
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    logger.info("Connected to MongoDB (database: '%s').", DATABASE_NAME)
    return db[COLLECTION_NAME] # Keep this for Langchain history

# def get_business_info_collection(): # Remove this function - managed by db_manager
#     client = MongoClient(MONGO_CONNECTION_STRING)
#     db = client[DATABASE_NAME]
#     return db[BUSINESS_INFO_COLLECTION]

def get_user_business_info(chat_id: int | str) -> str:
    """Loads business info using db_manager, accepting int or str chat_id."""
    try:
        int_chat_id = int(chat_id)
    except ValueError:
        logger.error(f"Invalid chat_id format for get_user_business_info: {chat_id}")
        return ""
    info_data = db_manager.load_business_info(int_chat_id)
    return info_data.get("business_info", "") if info_data else ""

def save_user_business_info(chat_id: int | str, info: str):
    """Saves business info using db_manager, accepting int or str chat_id."""
    try:
        int_chat_id = int(chat_id)
    except ValueError:
        logger.error(f"Invalid chat_id format for save_user_business_info: {chat_id}")
        return
    success = db_manager.save_business_info(int_chat_id, {"business_info": info})
    # Logging is handled within db_manager now
    if not success:
         logger.error(f"Failed to save business info via db_manager for chat {int_chat_id}")

def get_history_count(session_id: str):
    collection = get_mongo_collection()
    count = collection.count_documents({"session_id": {"$regex": f"^{session_id}"}})
    logger.debug("Session '%s' has %d messages in history.", session_id, count)
    return count

def get_summarized_history_for_session(session_id: str) -> str:
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
    history_obj = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )
    messages = history_obj.messages
    if not messages:
        return "No messages."
    combined = "\n".join([msg.content if hasattr(msg, "content") else str(msg) for msg in messages])
    try:
        from langchain.chains.summarize import load_summarize_chain
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = summary_chain.run(combined)
    except Exception as e:
        summary = f"Summary failed: {str(e)}"
    return summary # Keep this function as is (summarizes Langchain history)

def get_history_for_chat(telegram_chat_id: str, session_id: str = None):
    """Retrieves Langchain history, using updated session funcs internally."""
    # Convert chat_id to int for session management, but keep string for logging/session id generation
    str_chat_id = str(telegram_chat_id)
    try:
        int_chat_id = int(telegram_chat_id)
    except ValueError:
        logger.error(f"Invalid chat_id format in get_history_for_chat: {str_chat_id}. Cannot manage session.")
        # Proceed without session saving/loading if ID is invalid, Langchain might handle it
        int_chat_id = None # Flag that conversion failed

    if not session_id:
        loaded_session_id = None
        if int_chat_id is not None:
            loaded_session_id = get_session_id(int_chat_id) # Use updated function

        if loaded_session_id:
            session_id = loaded_session_id
        else:
            session_id = f"{str_chat_id}_{int(datetime.datetime.now().timestamp())}"
            if int_chat_id is not None:
                save_session_id(int_chat_id, session_id) # Use updated function
                logger.info(f"Created and saved new session '{session_id}' for chat '{int_chat_id}'.")
            else:
                 logger.warning(f"Created unsaved session '{session_id}' for invalid chat_id '{str_chat_id}'.")

    logger.debug(f"Retrieving Langchain history for session '{session_id}' in chat '{str_chat_id}'.")
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
    history_obj = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )
    try:
        messages = history_obj.messages
        logger.info(f"Loaded {len(messages)} historical messages for session '{session_id}'.")
    except Exception as e:
        logger.warning(f"Could not retrieve history messages for session '{session_id}': {e}")
    return history_obj # Keep returning Langchain history object

def get_user_info(chat_id: int | str, date=None) -> str:
    """Loads user report using db_manager, accepting int or str chat_id. Date is ignored."""
    try:
        int_chat_id = int(chat_id)
    except ValueError:
        logger.error(f"Invalid chat_id format for get_user_info: {chat_id}")
        return ""
    info_data = db_manager.load_business_info(int_chat_id)
    return info_data.get("user_report", "") if info_data else ""

############################################
# Business Info Summarization and Processing
############################################
from langchain.chains.summarize import load_summarize_chain
def summarize_business_info(raw_text: str) -> str:
    try:
        logger.process_start("Summarizing business info...")
        prompt_text = business_info_summary_prompt.format(raw_text=raw_text)
        start_time = time.time()
        response = llm_business([HumanMessage(content=prompt_text)])
        duration = time.time() - start_time
        result = response.content.strip()
        log_summarization(logger, raw_text, result, "Business Info")
        logger.process_end(f"Business info summarized in {duration:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error summarizing business info: {e}")
        return raw_text

def process_business_info(info_text, chat_id):
    return info_text.strip()

def save_message_to_history(chat_id, role, content):
    try:
        from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
        history_obj = get_history_for_chat(chat_id)
        if role == "user":
            from langchain.schema import HumanMessage
            message_obj = HumanMessage(content=content)
        elif role == "assistant":
            from langchain.schema import AIMessage
            message_obj = AIMessage(content=content)
        elif role == "system":
            from langchain.schema import SystemMessage
            message_obj = SystemMessage(content=content)
        else:
            from langchain.schema import HumanMessage
            message_obj = HumanMessage(content=content)
        history_obj.add_message(message_obj)
        if isinstance(content, list):
            from utils.helpers import format_multimodal_input
            truncated = format_multimodal_input(content)
        else:
            truncated = content
        truncated = truncated[:50] + ("..." if len(truncated) > 50 else "")
        logger.info("Saved message to history for chat '%s'. Role: '%s', Content: '%s'", chat_id, role, truncated)
    except Exception as e:
        logger.error("Error saving message to Langchain history for chat '%s': %s", chat_id, e)

# def load_user_reports(): # Comment out - reports loaded on demand
#     try:
#         # Direct access to db_manager collection (use with caution)
#         if db_manager.is_db_connected():
#             reports = db_manager.business_info_coll.find({}, {"chat_id": 1, "user_report": 1})
#             count = 0
#             for report in reports:
#                 if "chat_id" in report and "user_report" in report:
#                     daily_users_report[str(report["chat_id"])] = report["user_report"]
#                     count += 1
#             logger.info(f"[bold green]Loaded {count} user reports from database via db_manager[/bold green]")
#         else:
#             logger.warning("Database not connected, cannot load user reports.")
#     except Exception as e:
#         logger.error(f"Error loading user reports via db_manager: {e}")

# load_user_reports() # Comment out the call

def save_session_id(chat_id: int | str, session_id: str):
    """Saves session ID using db_manager, accepting int or str chat_id."""
    try:
        int_chat_id = int(chat_id)
    except ValueError:
        logger.error(f"Invalid chat_id format for save_session_id: {chat_id}")
        return
    success = db_manager.save_business_info(int_chat_id, {"session_id": session_id})
    if success:
        logger.info(f"Saved session ID '{session_id}' for chat '{int_chat_id}' via db_manager.")
    else:
        logger.error(f"Failed to save session ID via db_manager for chat {int_chat_id}")

def get_session_id(chat_id: int | str) -> str | None:
    """Loads session ID using db_manager, accepting int or str chat_id."""
    try:
        int_chat_id = int(chat_id)
    except ValueError:
        logger.error(f"Invalid chat_id format for get_session_id: {chat_id}")
        return None
    info_data = db_manager.load_business_info(int_chat_id)
    session_id = info_data.get("session_id", None) if info_data else None
    if session_id:
        logger.debug(f"Retrieved session_id '{session_id}' for chat_id '{int_chat_id}'")
    else:
        logger.debug(f"No session_id found for chat_id '{int_chat_id}'")
    return session_id

############################################
# Exports
############################################
__all__ = [
    "llm",
    "logger",
    "prompt",
    "help_text_prompt",
    "welcome_message",
    "image_analyzer_prompt",
    "insta_idea_prompt",
    "daily_report_prompt",
    "daily_task_prompt",
    "process_business_info",
    "summarize_business_info",
    "get_summarized_history_for_session", # Keep (uses Langchain history)
    "get_history_count", # Keep (uses Langchain history)
    # "get_business_info_collection", # Remove
    "get_mongo_collection", # Keep (used by Langchain history/checkpointer)
    "save_user_business_info", # Keep (now uses db_manager)
    "get_user_business_info", # Keep (now uses db_manager)
    "save_message_to_history", # Keep (saves to Langchain history)
    "get_history_for_chat", # Keep (uses Langchain history, calls updated session funcs)
    "get_user_info", # Keep (now uses db_manager)
    "run_agent",
    "daily_users_report", # Keep (in-memory cache, populated by generate_user_report)
    # "load_user_reports", # Remove (commented out)
    "save_session_id", # Expose updated function
    "get_session_id", # Expose updated function
]
