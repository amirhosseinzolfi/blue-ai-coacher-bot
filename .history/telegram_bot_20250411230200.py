# telegram_bot.py
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
    chat_session_map,
    business_info_map,
    ai_tone_map,
    business_info_update_pending,
    business_info_mode,
    ai_tone_update_pending,
    MONGO_CONNECTION_STRING,
    DATABASE_NAME,
    COLLECTION_NAME,
    BUSINESS_INFO_COLLECTION
)

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
SUMMARY_LLM_MODEL = "gemini-2.0-flash"  # Change if needed

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
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

class ToolCall(TypedDict):
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: str

class AgentState(TypedDict):
    messages: Sequence[BaseMessage] # Changed to BaseMessage
    tool_calls: List[ToolCall]
    requires_tool: bool
    current_tool: Optional[str]
    chat_id: str
    username: Optional[str]
    summary: Optional[str] # Added summary to state

############################################
# LangGraph Workflow and Node Definitions - IMPORTED FROM langgraph_agent.py
############################################
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver

# Import the new LangGraph app
from langgraph_agent import app # Import the compiled LangGraph app

llm_instance = llm  # For clarity in later calls

# --- User Report Generation --- (Keep this function as it's application-specific)
def generate_user_report(conversation_text: str, chat_id: str):
    try:
        user_lines = [line for line in conversation_text.splitlines() if line.startswith("HumanMessage:")]
        user_prompt = "\n".join(user_lines)
        prev_report = daily_users_report.get(chat_id, "")
        combined_context = user_prompt + ("\nPrevious User Report:\n" + prev_report if prev_report else "")
        report_prompt = USER_REPORT_PROMPT.format(conversation_text=combined_context)
        response = user_llm.invoke([HumanMessage(content=report_prompt)])
        report_text = response.content.strip()
        daily_users_report[chat_id] = report_text
        from db_manager import save_user_info
        save_user_info(chat_id, report_text)
        logger.info(f"[bold blue]User Report for chat {chat_id} generated:[/bold blue] {report_text[:100]}...")
        try:
            collection = get_business_info_collection()
            collection.update_one(
                {"chat_id": chat_id},
                {"$set": {"user_report": report_text, "updated_at": datetime.datetime.utcnow()}},
                upsert=True
            )
            logger.info(f"[bold green]User report saved to database for chat {chat_id}[/bold green]")
        except Exception as db_error:
            logger.error(f"Failed to save user report to database: {db_error}")
    except Exception as e:
        logger.error(f"Error generating user report for chat {chat_id}: {e}", exc_info=True)

# --- Conversation Summarization Functions --- (Keep optimize_memory as it's still used outside LangGraph potentially, and LangGraph summarization is different)
# def optimize_memory(state: AgentState) -> AgentState: # REPLACED BY LANGGRAPH SUMMARIZATION
#     """
#     Summarize conversation if there are more than 10 human/AI messages.
#     Then keep only a summary and the last 2 messages.
#     """
#     THRESHOLD = 10
#     conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
#     if len(conv_messages) <= THRESHOLD:
#         logger.info(f"Message count ({len(conv_messages)}) below threshold ({THRESHOLD}); skipping summarization")
#         return state
#
#     existing_summary = state.get("summary", "")
#     conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conv_messages])
#     try:
#         full_prompt = summary_prompt.format(conversation=conversation_text, existing_summary=existing_summary)
#     except Exception:
#         full_prompt = f"{conversation_text}\n\n{existing_summary}" if existing_summary else conversation_text
#     logger.process_start("Starting conversation summarization")
#     try:
#         start_time = time.time()
#         summary_response = llm_summary.invoke([HumanMessage(content=full_prompt)])
#         duration = time.time() - start_time
#         new_summary = summary_response.content.strip()
#         logger.info(f"Summarization completed in {duration:.2f}s: {new_summary[:100]}...")
#         log_summarization(logger, conversation_text, new_summary, "Conversation")
#     except Exception as e:
#         logger.error(f"Error during summarization: {e}", exc_info=True)
#         logger.process_end("Summarization failed")
#         return state
#
#     new_system = SystemMessage(content=f"[CONVERSATION SUMMARY]: {new_summary}")
#     new_messages = [new_system] + conv_messages[-2:]
#     logger.info(f"Trimmed conversation: kept {len(new_messages)} messages (summary + last 2 messages)")
#     logger.process_end("Conversation summarization and trimming completed")
#     state["summary"] = new_summary
#     return {
#         "messages": new_messages,
#         "tool_calls": state.get("tool_calls", []),
#         "requires_tool": state.get("requires_tool", False),
#         "current_tool": state.get("current_tool", None),
#         "chat_id": state["chat_id"],
#         "username": state.get("username", None),
#         "summary": new_summary,
#     }

class MessageCounter:
    def __init__(self):
        self._counters = {}
        self._lock = threading.Lock()

    def increment_and_check(self, chat_id: str) -> bool:
        with self._lock:
            self._counters[chat_id] = self._counters.get(chat_id, 0) + 1
            if self._counters[chat_id] >= 10:  # Summarize every 10 messages (adjust if needed to match langgraph condition)
                self._counters[chat_id] = 0
                return True
            return False

    def reset(self, chat_id: str):
        with self._lock:
            self._counters[chat_id] = 0

message_counter = MessageCounter()

############################################
# Agent Function - NO NEED TO DEFINE AGENT FUNCTION HERE, IT'S IN LANGGRAPH NOW
############################################
# def agent(state: AgentState): # REPLACED BY LANGGRAPH AGENT NODE
# ... (Old agent function code)


############################################
# Workflow Initialization - NO NEED TO DEFINE WORKFLOW HERE, IT'S IN LANGGRAPH NOW
############################################
# workflow = StateGraph(AgentState) # REPLACED BY IMPORTED APP
# ... (Old workflow definition)


############################################
# Main Agent Function: run_agent - MODIFIED TO USE NEW LANGGRAPH APP
############################################
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain.schema import HumanMessage

def run_agent(query, chat_id, message_id, username=None):
    """
    Processes a user query using the new LangGraph app from langgraph_agent.py.
    Handles both text and multimodal inputs.
    """
    is_multimodal = isinstance(query, list)
    from utils.rich_logger import log_agent_execution
    log_agent_execution(logger, chat_id, None, username, query, is_multimodal)
    logger.process_start(f"Starting LangGraph agent for chat: {chat_id}")
    start_time = time.time()
    # Add the missing import for refine_ai_response
    from utils.helpers import refine_ai_response
    user_id = str(chat_id)

    session_id = get_session_id(user_id)
    if not session_id:
        session_id = f"{user_id}_{int(datetime.datetime.now().timestamp())}"
        save_session_id(user_id, session_id)
        logger.info(f"Created new session '{session_id}' for chat '{user_id}'.")

    thread_id = session_id
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
    inputs: AgentState = { # Use AgentState
        "messages": state_messages,
        "tool_calls": [],
        "requires_tool": False,
        "current_tool": None,
        "chat_id": user_id,
        "username": username,
        "summary": "" # Initialize summary in the state
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        with MongoDBSaver.from_conn_string(
            MONGO_CONNECTION_STRING,
            db_name=DATABASE_NAME,
            collection_name="langgraph_checkpoints"
        ) as checkpointer:
            # graph = workflow.compile(checkpointer=checkpointer) # USE IMPORTED APP
            graph = app # Use imported compiled app
            logger.info(f"[bold blue]=== Main LangGraph Graph Execution Start for thread {thread_id} ===[/bold blue]")
            final_response = None
            step_count = 0
            for output in graph.stream(inputs, config=config):
                step_count += 1
                logger.info(f"[blue]Step {step_count}: Nodes executed: {list(output.keys())}[/blue]")
                if "summarize_conversation" in output and "messages" in output["summarize_conversation"]: # Node name is now summarize_conversation
                    most_recent_msg = output["summarize_conversation"]["messages"][-1] # Node name is now summarize_conversation
                    if isinstance(most_recent_msg, AIMessage):
                        final_response = most_recent_msg.content
                        logger.debug("Got response from summarize_conversation node") # Node name is now summarize_conversation
                elif "conversation" in output and "messages" in output["conversation"] and final_response is None: # Node name is conversation
                    most_recent_msg = output["conversation"]["messages"][-1] # Node name is conversation
                    if isinstance(most_recent_msg, AIMessage):
                        final_response = most_recent_msg.content
                        logger.debug("Got response from agent node (conversation)") # Node name is conversation
            logger.info(f"[bold blue]=== Main LangGraph Graph Execution Completed ===[/bold blue]")
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
# MongoDB Helper Functions for Chat History & Business Info (KEEP THESE AS THEY ARE)
############################################
def get_mongo_collection():
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    logger.info("Connected to MongoDB (database: '%s').", DATABASE_NAME)
    return db[COLLECTION_NAME]

def get_business_info_collection():
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    return db[BUSINESS_INFO_COLLECTION]

def get_user_business_info(chat_id: str) -> str:
    collection = get_business_info_collection()
    result = collection.find_one({"chat_id": chat_id})
    return result.get("business_info", "") if result else ""

def save_user_business_info(chat_id: str, info: str):
    collection = get_business_info_collection()
    collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"business_info": info, "updated_at": datetime.datetime.utcnow()}},
        upsert=True
    )
    logger.info("Updated business info for chat '%s'", chat_id)

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
    return summary

def get_history_for_chat(telegram_chat_id: str, session_id: str = None):
    if not session_id:
        session_id = get_session_id(telegram_chat_id)
        if not session_id:
            session_id = f"{telegram_chat_id}_{int(datetime.datetime.now().timestamp())}"
            save_session_id(telegram_chat_id, session_id)
            logger.info(f"Created new session '{session_id}' for chat '{telegram_chat_id}'.")
    logger.debug(f"Retrieving history for session '{session_id}' in chat '{telegram_chat_id}'.")
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
    return history_obj

def get_user_info(chat_id, date=None):
    today = date or datetime.datetime.now().strftime("%Y-%m-%d")
    client_db = MongoClient(MONGO_CONNECTION_STRING)
    db = client_db[DATABASE_NAME]
    USER_INFO_COLLECTION = db["user_info"]
    doc = USER_INFO_COLLECTION.find_one({"chat_id": chat_id, "date": today})
    return doc["user_info"] if doc and "user_info" in doc else ""

############################################
# Business Info Summarization and Processing (KEEP THESE AS THEY ARE)
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
        logger.error("Error saving message to history for chat '%s': %s", chat_id, e)

def load_user_reports():
    try:
        collection = get_business_info_collection()
        reports = collection.find({}, {"chat_id": 1, "user_report": 1})
        count = 0
        for report in reports:
            if "chat_id" in report and "user_report" in report:
                daily_users_report[report["chat_id"]] = report["user_report"]
                count += 1
        logger.info(f"[bold green]Loaded {count} user reports from database[/bold green]")
    except Exception as e:
        logger.error(f"Error loading user reports: {e}")

load_user_reports()

def save_session_id(chat_id: str, session_id: str):
    collection = get_business_info_collection()
    collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"session_id": session_id}},
        upsert=True
    )
    logger.info(f"Saved session ID '{session_id}' for chat '{chat_id}'.")

def get_session_id(chat_id: str) -> str:
    collection = get_business_info_collection()
    result = collection.find_one({"chat_id": chat_id})
    return result.get("session_id", None) if result else None

############################################
# Exports (KEEP THESE AS THEY ARE)
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
    "get_summarized_history_for_session",
    "get_history_count",
    "get_business_info_collection",
    "get_mongo_collection",
    "save_user_business_info",
    "get_user_business_info",
    "save_message_to_history",
    "get_history_for_chat",
    "get_user_info",
    "run_agent",
    "daily_users_report",
    "load_user_reports",
]