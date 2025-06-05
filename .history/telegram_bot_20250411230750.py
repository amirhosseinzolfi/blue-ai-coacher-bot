#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
telegram_bot.py – Integrates LangChain with MongoDB for conversation
history management and LLM responses. Provides functions for generating
reports, summarizing business info, and managing conversation state.

Note: This file has been updated to REMOVE the old LangGraph structure.
It now calls into langgraph_agent.py for the conversation flow.
All other functionalities remain largely unchanged.
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

atexit.register(lambda: llm.client.close() if hasattr(llm, "client") else None)
atexit.register(lambda: llm_business.client.close() if hasattr(llm_business, "client") else None)

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
# Summarization & Counters (unchanged)
############################################
def generate_user_report(conversation_text: str, chat_id: str):
    """
    Example function to generate a user report with user_llm.
    """
    try:
        user_lines = [line for line in conversation_text.splitlines() if line.startswith("HumanMessage:")]
        user_prompt = "\n".join(user_lines)
        prev_report = daily_users_report.get(chat_id, "")
        combined_context = user_prompt + ("\nPrevious User Report:\n" + prev_report if prev_report else "")
        report_prompt = USER_REPORT_PROMPT.format(conversation_text=combined_context)
        response = user_llm.invoke([HumanMessage(content=report_prompt)])
        report_text = response.content.strip()
        daily_users_report[chat_id] = report_text

        # Save to DB
        from db_manager import save_user_info
        save_user_info(chat_id, report_text)
        logger.info(f"[bold blue]User Report for chat {chat_id} generated:[/bold blue] {report_text[:100]}...")

        # Also store in business_info
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

class MessageCounter:
    """
    Unchanged, used for custom summarization triggers if needed.
    """
    def __init__(self):
        self._counters = {}
        self._lock = threading.Lock()

    def increment_and_check(self, chat_id: str, threshold=5) -> bool:
        with self._lock:
            self._counters[chat_id] = self._counters.get(chat_id, 0) + 1
            if self._counters[chat_id] >= threshold:
                self._counters[chat_id] = 0
                return True
            return False

    def reset(self, chat_id: str):
        with self._lock:
            self._counters[chat_id] = 0

message_counter = MessageCounter()

############################################
# Business Info Summaries
############################################
from langchain.chains.summarize import load_summarize_chain
def summarize_business_info(raw_text: str) -> str:
    """
    Example business info summarization using llm_business
    """
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

############################################
# DB & Helper Functions (unchanged)
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

def load_user_reports():
    """
    Load any previously saved user_report from DB into daily_users_report.
    """
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
# Import the New LangGraph Agent
############################################
# We import our function from the brand-new file:
from langgraph_agent import run_langgraph_agent
from langchain.schema import HumanMessage

############################################
# run_agent - Integrates with new code
############################################
def run_agent(query, chat_id, message_id, username=None):
    """
    This function calls the new `langgraph_agent.py` workflow via 
    `run_langgraph_agent`. It handles session IDs, DB setup, etc., 
    but delegates the conversation logic to the new LangGraph graph.
    """
    from utils.helpers import refine_ai_response
    from utils.rich_logger import log_agent_execution

    is_multimodal = isinstance(query, list)
    log_agent_execution(logger, chat_id, None, username, query, is_multimodal)
    logger.process_start(f"Starting new LangGraph agent for chat: {chat_id}")
    start_time = time.time()

    # Create or retrieve session ID from DB
    session_id = get_session_id(chat_id)
    if not session_id:
        session_id = f"{chat_id}_{int(datetime.datetime.now().timestamp())}"
        save_session_id(chat_id, session_id)
        logger.info(f"Created new session '{session_id}' for chat '{chat_id}'.")

    # Prepare the user input as a HumanMessage
    if is_multimodal:
        # If your code base uses lists for multimodal content, you can store them 
        # as a single message or parse them further
        human_msg = HumanMessage(content=query)
    else:
        # If text, optionally format with your typical approach:
        formatted_query = f"name : {username}\n# user prompt : {query}\n\n"
        human_msg = HumanMessage(content=formatted_query)

    logger.debug(f"Sending user input to new LangGraph agent. Session ID: {session_id}")

    try:
        # Call into the new graph
        response_text = run_langgraph_agent(human_msg, session_id=session_id)

        # Optionally refine the final text (strip placeholders, remove <think> etc.)
        from utils.helpers import strip_thinking_tags
        refined_text = strip_thinking_tags(response_text)
        refined_text = refine_ai_response(refined_text.strip())

        elapsed_time = time.time() - start_time
        logger.info(f"✅ Generated response in {elapsed_time:.2f}s")
        logger.process_end("LangGraph agent execution successful")
        return refined_text

    except Exception as e:
        logger.error(f"❌ Error during new LangGraph execution: {e}", exc_info=True)
        logger.process_end("LangGraph agent execution failed")
        return "متأسفم، مشکلی در پردازش درخواست شما پیش آمد."

############################################
# Exports (unchanged, but references updated logic)
############################################
__all__ = [
    "llm",
    "logger",
    "prompt_template_text",
    "help_text_prompt",
    "welcome_message",
    "image_analyzer_prompt",
    "insta_idea_prompt",
    "daily_report_prompt",
    "daily_task_prompt",
    "process_business_info",
    "summarize_business_info",
    "get_business_info_collection",
    "get_mongo_collection",
    "save_user_business_info",
    "get_user_business_info",
    "daily_users_report",
    "load_user_reports",
    "run_agent",
]
