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
from typing import List, Dict, Any, Optional, Union
############################################
# Third-Party Imports
############################################
import requests

############################################
# Local Imports and Configuration
############################################
from config import (
    TELEGRAM_BOT_TOKEN,
    ai_tone_map,
    DATABASE_NAME
)

from utils.rich_logger import (
    setup_logger, display_content, log_function, log_telegram_message,
    log_api_interaction, log_summarization, log_ai_interaction,
    log_user_business_data, log_llm_request, log_agent_execution,
    log_comprehensive_interaction, log_langgraph_execution
)

# Import LLM instances from centralized module
from llm_initial import (
    llm,
    llm_business,
    user_llm,
    llm_summary,
    image_analyze_llm,
    PRIMARY_LLM_MODEL
)

logger = setup_logger(level=logging.INFO)
logger.info("Initializing LangChain integrations...")

# Global dictionary for user reports
daily_users_report = {}

############################################
# Import Prompt Templates and Helper Texts
############################################
from prompts.prompts import (
    PROMPT_TEMPLATE_TEXT, 
    DAILY_TASK_PROMPT,
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
daily_report_prompt = DAILY_REPORT_PROMPT
insta_idea_prompt = INSTA_IDEA_PROMPT
image_analyzer_prompt = IMAGE_ANALYZER_PROMPT
business_info_summary_prompt = BUSINESS_INFO_SUMMARY_PROMPT  
welcome_message = WELCOME_MESSAGE
help_text_prompt = HELP_TEXT

############################################
# LangChain Prompt Template
############################################
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template_text),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("User Input: {input}\nCurrent AI Tone: {ai_tone}\nBusiness Context: {business_info}")
])
logger.info("Legacy LangChain prompt template created.")

############################################
# LangGraph Workflow - Imported from graph_definition.py
############################################
from langgraph.checkpoint.sqlite import SqliteSaver
from graph_definition import AgentState, get_compiled_app
logger.info("LangGraph application factory imported from graph_definition.py")

# --- User Report Generation ---
def generate_user_report(conversation_text: str, chat_id: str):
    try:
        user_lines = [line for line in conversation_text.splitlines() if line.startswith("HumanMessage:")]
        user_prompt_text = "\n".join(user_lines)
        prev_report = daily_users_report.get(chat_id, "")
        combined_context = user_prompt_text + ("\nPrevious User Report:\n" + prev_report if prev_report else "")
        
        report_generation_prompt = USER_REPORT_PROMPT.format(conversation_text=combined_context)
        
        logger.info(f"Generating user report for chat {chat_id} with context: {report_generation_prompt[:100]}...")
        response_content_placeholder = "User report content placeholder due to missing LLM call in original function." 
        report_text = response_content_placeholder 
        
        daily_users_report[chat_id] = report_text
        from db_manager import save_user_report
        save_user_report(chat_id, report_text)
        logger.info(f"[bold blue]User Report for chat {chat_id} generated (placeholder):[/bold blue] {report_text[:100]}...")
        try:
            save_user_report(chat_id, report_text)
            logger.info(f"[bold green]User report saved to database for chat {chat_id}[/bold green]")
        except Exception as db_error:
            logger.error(f"Failed to save user report (2nd attempt): {db_error}")
    except Exception as e:
        logger.error(f"Error generating user report for chat {chat_id}: {e}", exc_info=True)

############################################
# Main Agent Function: run_agent
############################################
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage

def load_users_tasks_json() -> str:
    """
    Load user tasks, preferring database but falling back to JSON file if needed
    Returns a JSON string compatible with existing code
    """
    chat_id = None  # We'll need to get the current chat_id from context
    
    try:
        # First try to get from database
        from db_manager import get_all_user_tasks, export_tasks_to_json_format
        
        if chat_id:
            # If we have a specific chat_id, get tasks for that chat
            tasks_dict = {chat_id: export_tasks_to_json_format(chat_id)}
            return json.dumps(tasks_dict, ensure_ascii=False)
        else:
            # Otherwise use JSON file for backward compatibility
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, "database", "users_task.json")
            
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    return json.dumps(json.load(f), ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to load users_task.json: {e}")
                return "{}"
    except Exception as e:
        logger.error(f"Error loading tasks from database: {e}", exc_info=True)
        
        # Fallback to JSON file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "database", "users_task.json")
        
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), ensure_ascii=False)
        except Exception as file_e:
            logger.error(f"Failed to load users_task.json: {file_e}")
            return "{}"

def run_agent(query: Union[str, List[Dict[str, Any]]], chat_id: str, message_id: Any, username: Optional[str] = None) -> str:
    logger.process_start(f"🚀 Starting run_agent for chat: {chat_id}, user: {username}")
    start_time = time.time()

    if isinstance(query, str) and "{users_tasks_json}" in query:
        logger.info("Formatting query with users_tasks_json...")
        query = query.format(users_tasks_json=load_users_tasks_json())

    is_multimodal = isinstance(query, list)
    log_agent_execution(logger, chat_id, None, username, query, is_multimodal)

    from utils.helpers import refine_ai_response
    user_id_str = str(chat_id)

    # Get or create session ID
    current_session_id = get_session_id(user_id_str)
    if not current_session_id:
        current_session_id = f"{user_id_str}_{int(datetime.datetime.now().timestamp())}"
        save_session_id(user_id_str, current_session_id)
        logger.info(f"✨ Created new session '{current_session_id}' for chat '{user_id_str}'.")

    thread_id = current_session_id
    logger.info(f"🧵 Using Thread ID for LangGraph: {thread_id}")

    # Load conversation history
    logger.debug(f"Loading conversation history for chat {user_id_str}, session {thread_id}...")
    history_container = get_history_for_chat(user_id_str, thread_id) 
    history_messages = history_container.messages if history_container else []
    logger.info(f"Loaded {len(history_messages)} messages from history.")

    # Get conversation summary from database
    from db_manager import db_manager as main_db_manager
    conversation_summary_content = main_db_manager.get_conversation_summary(user_id_str)
    
    # Prepare initial messages for the state - filter out any existing summary message
    initial_messages_for_state = []
    for msg in history_messages:
        if not (isinstance(msg, SystemMessage) and "[CONVERSATION SUMMARY]:" in msg.content):
            initial_messages_for_state.append(msg)
    
    # Add current user message
    if is_multimodal:
        human_message = HumanMessage(content=query)
    else:
        human_message = HumanMessage(content=str(query))
    
    # Create state with current messages and summary
    current_state_messages = initial_messages_for_state + [human_message]
    logger.info(f"🔧 Prepared initial state: messages={len(current_state_messages)}, summary={'present' if conversation_summary_content else 'none'}.")
    initial_graph_inputs: AgentState = {
        "messages": current_state_messages,
        "chat_id": user_id_str,
        "session_id": thread_id,
        "username": username,
        "summary": conversation_summary_content if conversation_summary_content else None,
    }

    # Execute graph
    config = {"configurable": {"thread_id": thread_id}}
    final_ai_response_content = None

    try:
        logger.info(f"[graph] Invoking LangGraph app for thread_id: {thread_id}...")
        
        with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
            graph_with_checkpointing = get_compiled_app(checkpointer=checkpointer)
            
            # Replace streaming with a single invoke call
            logger.info("Executing LangGraph in non-streaming mode")
            final_state = graph_with_checkpointing.invoke(initial_graph_inputs, config=config)
            
            # Log the final state results
            logger.info(f"LangGraph execution completed for thread {thread_id}")
            
            # Extract the final AI response from messages
            final_messages = [msg for msg in final_state["messages"] if not isinstance(msg, RemoveMessage)]
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage):
                    if "متأسفم" not in msg.content:
                        final_ai_response_content = msg.content
                        logger.info(f"Retrieved AI response: {final_ai_response_content[:50]}...")
                    break
            
            # Store the final state for reference
            final_graph_state = graph_with_checkpointing.get_state(config)
    
    except Exception as e:
        logger.error(f"❌ Error during LangGraph execution in run_agent: {e}", exc_info=True)
        logger.process_end("LangGraph agent execution failed")
        return "An error occurred processing your request. Please try again."

    # Fallback if no valid response found
    if not final_ai_response_content or "متأسفم" in final_ai_response_content:
        logger.warning("No valid AI response captured from graph execution. Using fallback.")
        final_ai_response_content = "متأسفم، نتوانستم پاسخ مناسبی بیابم. لطفا دوباره تلاش کنید."

    # Process and return response
    refined_response = refine_ai_response(final_ai_response_content.strip())
    elapsed_time = time.time() - start_time
    logger.info(f"✅ run_agent completed in {elapsed_time:.2f}s. Response: {refined_response[:50]}...")
    logger.process_end("LangGraph agent execution successful")
    return refined_response

############################################
# SQLite Helper Functions
############################################
def get_user_business_info(chat_id: str) -> str:
    from db_manager import get_business_info as db_get_business_info
    return db_get_business_info(chat_id)
    
def save_user_business_info(chat_id: str, info: str, chat_type: str = "private"):
    from db_manager import save_business_info as db_save_business_info
    db_save_business_info(chat_id, info, chat_type)

def get_history_for_chat(telegram_chat_id: str, session_id: Optional[str] = None):
    from db_manager import get_chat_messages as db_get_chat_messages, ChatMessage as DBChatMessage
    from db_manager import db_manager as main_db_manager
    if not session_id:
        session_id = main_db_manager.get_session_id(str(telegram_chat_id))

    db_messages: List[DBChatMessage] = db_get_chat_messages(str(telegram_chat_id), session_id)
    
    converted_messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
    for msg in db_messages:
        if msg.role == "system":
            converted_messages.append(SystemMessage(content=msg.content))
        elif msg.role == "user":
            try:
                parsed_content = json.loads(msg.content)
                if isinstance(parsed_content, list):
                    converted_messages.append(HumanMessage(content=parsed_content))
                else:
                    converted_messages.append(HumanMessage(content=msg.content))
            except json.JSONDecodeError:
                converted_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            converted_messages.append(AIMessage(content=msg.content))
    
    class HistoryObject:
        def __init__(self, messages_list):
            self.messages = messages_list
    
    logger.debug(f"Converted {len(converted_messages)} DB messages to LangChain messages for chat {telegram_chat_id}, session {session_id}.")
    return HistoryObject(converted_messages)

def save_message_to_history(chat_id: str, role: str, content: Union[str, List[Dict]], session_id: Optional[str] = None) -> None:
    from db_manager import save_message_to_history as db_save_message_to_history_direct
    from db_manager import db_manager as main_db_manager

    if not session_id:
        session_id = main_db_manager.get_session_id(str(chat_id))

    content_to_save = content
    if isinstance(content, list):
        try:
            content_to_save = json.dumps(content)
        except TypeError as e:
            logger.error(f"Failed to serialize multimodal content to JSON: {e}. Saving as string representation.")
            content_to_save = str(content)

    try:
        db_save_message_to_history_direct(str(chat_id), role, content_to_save, session_id)
        logger.debug(f"Message saved to history. Chat: {chat_id}, Role: {role}, Session: {session_id}, Content: {str(content_to_save)[:50]}...")
    except Exception as e:
        logger.error(f"Failed to save message to history via db_manager: {str(e)}", exc_info=True)

def get_session_id(chat_id: str) -> Optional[str]:
    from db_manager import db_manager as main_db_manager
    return main_db_manager.get_session_id(str(chat_id))

def save_session_id(chat_id: str, session_id: str):
    from db_manager import db_manager as main_db_manager
    main_db_manager.save_session_id(str(chat_id), session_id)
    logger.info(f"Saved session ID '{session_id}' for chat '{chat_id}' via db_manager.")

def get_user_info(chat_id, date=None):
    from db_manager import get_user_info as db_get_user_info_main
    return db_get_user_info_main(str(chat_id), date)

############################################
# Business Info Summarization and Processing
############################################
def summarize_business_info(raw_text: str) -> str:
    try:
        logger.process_start("Summarizing business info...")
        prompt_text_for_biz_summary = business_info_summary_prompt.format(raw_text=raw_text)
        start_time = time.time()
        
        # Direct non-streaming business info summarization
        response = llm_business.invoke([HumanMessage(content=prompt_text_for_biz_summary)])
        
        duration = time.time() - start_time
        result = response.content.strip()
        log_summarization(logger, raw_text, result, "Business Info")
        logger.process_end(f"Business info summarized in {duration:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error summarizing business info: {e}", exc_info=True)
        return raw_text

def process_business_info(info_text, chat_id):
    return info_text.strip()

def load_user_reports():
    try:
        from db_manager import db_manager as main_db_manager
        cursor = main_db_manager.conn.execute("SELECT chat_id, user_report FROM business_info WHERE user_report IS NOT NULL")
        reports = cursor.fetchall()
        count = 0
        for report_row in reports:
            if "chat_id" in report_row.keys() and "user_report" in report_row.keys() and report_row["user_report"]:
                daily_users_report[str(report_row["chat_id"])] = report_row["user_report"]
                count += 1
        logger.info(f"[bold green]Loaded {count} user reports from database into memory.[/bold green]")
    except Exception as e:
        logger.error(f"Error loading user reports from DB: {e}", exc_info=True)

load_user_reports()

def new_chat_session(chat_id: str) -> str:
    from db_manager import start_new_session as db_start_new_session
    new_session_id = db_start_new_session(str(chat_id))
    logger.info(f"🆕 New chat session '{new_session_id}' started for chat '{chat_id}'. Previous conversation history and summary cleared from DB.")
    return new_session_id

############################################
# Exports
############################################
__all__ = [
    "llm",
    "llm_summary",
    "PRIMARY_LLM_MODEL",
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
    "save_user_business_info",
    "get_user_business_info",
    "save_message_to_history",
    "get_history_for_chat",
    "get_user_info",
    "run_agent",
    "daily_users_report",
    "load_user_reports",
    "new_chat_session",
    "AgentState",
    "image_analyze_llm"  # Add image_analyze_llm to exports
]