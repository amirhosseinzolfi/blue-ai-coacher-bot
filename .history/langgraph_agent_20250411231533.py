#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
langgraph_agent.py – Defines the core LangGraph agent structure for conversation
handling and summarization, integrated with the Telegram bot's context.
"""

import logging
from typing import Literal, Optional, Sequence, Union
from typing_extensions import TypedDict

# --- Third-Party Imports ---
from pymongo import MongoClient
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, RemoveMessage

# --- Local Imports and Configuration ---
from config import (
    MONGO_CONNECTION_STRING,
    DATABASE_NAME,
    LANGGRAPH_CHECKPOINT_COLLECTION, # Added to config.py
    TELEGRAM_BOT_TOKEN, # Import other necessary configs if needed elsewhere
    OPENAI_API_KEY,
)
from prompts.prompts import (
    PROMPT_TEMPLATE_TEXT,
    SUMMARY_PROMPT_TEXT, # Renamed from SUMMARY_PROMPT in original telegram_bot.py for clarity
)
from utils.rich_logger import setup_logger, log_function, log_ai_interaction, log_summarization, log_user_business_data
from utils.helpers import strip_thinking_tags, refine_ai_response # Assuming refine_ai_response is needed here or after graph execution
from utils.date_helpers import get_full_shamsi_date
from db_manager import get_user_business_info, get_ai_tone, get_user_info # Import specific functions

# --- Logger Setup ---
logger = setup_logger(level=logging.INFO, logger_name="LangGraphAgent")

# --- LLM Instances (Imported from telegram_bot.py or initialized centrally) ---
# LLMs should be initialized once, ideally passed in or imported from a central setup
# For this example, we'll assume they are imported/accessible globally
# If not, they need to be passed during graph compilation or execution setup
try:
    # Attempt to import pre-initialized LLMs from telegram_bot (requires careful structuring)
    # This might cause circular dependency issues if not handled carefully.
    # A better approach is a dedicated llm_setup.py or passing instances.
    # For simplicity now, let's assume they are accessible.
    from telegram_bot import llm as primary_llm, llm_summary as summary_llm
    logger.info("Imported LLM instances from telegram_bot.")
except ImportError:
    logger.error("Could not import LLM instances from telegram_bot.py. Ensure they are initialized and accessible.")
    # As a fallback, re-initialize (not recommended for production)
    from langchain_openai import ChatOpenAI
    primary_llm = ChatOpenAI(
        base_url="http://localhost:15201/v1", # Use config vars
        model_name="gpt-4o", # Use config vars
        temperature=0.5,
        api_key=OPENAI_API_KEY
    )
    summary_llm = ChatOpenAI(
        base_url="http://localhost:15201/v1", # Use config vars
        model_name="gemini-2.0-flash", # Use config vars
        temperature=0.5,
        api_key=OPENAI_API_KEY
    )
    logger.warning("Re-initialized LLM instances as fallback.")


# --- Checkpointer Setup ---
try:
    checkpointer = MongoDBSaver.from_conn_string(
        MONGO_CONNECTION_STRING,
        db_name=DATABASE_NAME,
        collection_name=LANGGRAPH_CHECKPOINT_COLLECTION # Use dedicated collection
    )
    logger.info(f"LangGraph checkpointer connected to MongoDB: db='{DATABASE_NAME}', collection='{LANGGRAPH_CHECKPOINT_COLLECTION}'")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB checkpointer: {e}", exc_info=True)
    checkpointer = None # Handle graph compilation failure later

# --- State Definition ---
class AgentState(MessagesState):
    """
    Represents the state of the conversation graph.

    Attributes:
        messages: The sequence of messages in the conversation.
        summary: A running summary of the conversation.
        chat_id: The unique identifier for the chat.
        username: The username of the user, if available.
    """
    summary: str = ""
    chat_id: str
    username: Optional[str] = None # Make username optional

# --- Graph Node Functions ---

@log_function(logger)
def call_llm(state: AgentState):
    """
    Generates a response using the primary LLM based on the current conversation state and context.
    """
    logger.process_start("Executing call_llm node")
    chat_id = state["chat_id"]
    username = state.get("username", None) # Get username from state
    messages: Sequence[BaseMessage] = state['messages']
    summary = state.get('summary', "")

    # --- Fetch Context ---
    try:
        ai_tone = get_ai_tone(chat_id) # Uses db_manager -> potentially needs chat_type if used
        business_info = get_user_business_info(chat_id) # Uses db_manager -> potentially needs chat_type if used
        user_report = get_user_info(chat_id) # Uses db_manager
        current_shamsi_date = get_full_shamsi_date()
        log_user_business_data(logger, chat_id, business_info, user_report)
        logger.info(f"Context Fetched - Tone: {ai_tone}, Date: {current_shamsi_date}")
    except Exception as e:
        logger.error(f"Error fetching context for chat {chat_id}: {e}", exc_info=True)
        ai_tone = "دوستانه"
        business_info = "اطلاعات کسب و کار در دسترس نیست."
        user_report = "گزارش کاربر در دسترس نیست."
        current_shamsi_date = "تاریخ نامشخص"

    # --- Prepare Messages for LLM ---
    system_prompt_content = PROMPT_TEMPLATE_TEXT # Base system prompt
    system_prompt_content += f"\n\nCURRENT DATE: Today's date in Iranian Shamsi calendar is {current_shamsi_date}."
    system_prompt_content += f"\n\nBUSINESS CONTEXT: {business_info}"
    system_prompt_content += f"\n\nAI TONE: Please adopt a '{ai_tone}' tone in your response."
    if username:
        system_prompt_content += f"\n\nADDRESS USER: Address the user directly as '{username}' where appropriate."
    if user_report:
         system_prompt_content += f"\n\nUSER REPORT SUMMARY: {user_report}" # Include user report in system prompt

    # Add conversation summary if it exists
    if summary:
        system_prompt_content += f"\n\n[CONVERSATION SUMMARY SO FAR]:\n{summary}"

    # Construct the message list for the LLM
    llm_messages = [SystemMessage(content=system_prompt_content)]

    # Add conversation history (excluding potential previous summaries stored as SystemMessages?)
    # MessagesState stores all messages added, checkpointer loads them.
    llm_messages.extend(messages) # Add all current messages from state

    # Extract last user input for logging
    last_user_input_content = "No user input found"
    if messages and isinstance(messages[-1], HumanMessage):
         last_user_input_content = messages[-1].content
         # Handle multimodal display for logging
         if isinstance(last_user_input_content, list):
             last_user_input_content = "[Multimodal Input]"
         elif isinstance(last_user_input_content, str):
             last_user_input_content = last_user_input_content[:100] + ("..." if len(last_user_input_content) > 100 else "")

    logger.info(f"Calling Primary LLM. Last input: '{last_user_input_content}'")
    logger.debug(f"Messages sent to LLM ({len(llm_messages)}): {llm_messages}")

    # --- Invoke LLM ---
    try:
        response = primary_llm.invoke(llm_messages)
        ai_content = response.content

        # --- Post-processing ---
        if username:
            for placeholder in ["[نام کاربر]", "[name]", "[نام]"]:
                ai_content = ai_content.replace(placeholder, username)

        ai_content = strip_thinking_tags(ai_content)
        # Maybe apply refine_ai_response here or after the graph finishes?
        # Let's apply it here for consistency within the agent's turn.
        ai_content = refine_ai_response(ai_content)

        log_ai_interaction(logger, last_user_input_content, ai_content, primary_llm.model_name)
        logger.process_end("call_llm node execution finished")
        return {"messages": [AIMessage(content=ai_content)]} # Append only the new AI message

    except Exception as e:
        logger.error(f"Error during LLM invocation in call_llm: {e}", exc_info=True)
        logger.process_end("call_llm node failed")
        # Return an error message to the user
        return {"messages": [AIMessage(content="متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.")]}


@log_function(logger)
def summarize_conversation(state: AgentState):
    """
    Summarizes the conversation history using the summary LLM.
    Updates the 'summary' field in the state and keeps a summary message + last N messages.
    """
    logger.process_start("Executing summarize_conversation node")
    messages = state['messages']
    current_summary = state.get("summary", "")

    # Filter out non-Human/AI messages for summarization input (optional, maybe summarize all?)
    conv_messages = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
    if not conv_messages:
         logger.info("No Human/AI messages to summarize.")
         logger.process_end("Summarization skipped (no messages)")
         # Just pass through the state if no messages to summarize
         return {} # Return empty dict to indicate no state change intended by this node

    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conv_messages])

    # Prepare prompt for summarization LLM
    try:
        # Use the dedicated summary prompt template
        summary_llm_prompt = SUMMARY_PROMPT_TEXT.format(
            existing_summary=current_summary,
            conversation_history=conversation_text
        )
    except Exception as e:
         logger.warning(f"Failed to format summary prompt: {e}. Using basic concatenation.")
         summary_llm_prompt = f"Current Summary:\n{current_summary}\n\nNew Conversation:\n{conversation_text}\n\nPlease provide an updated, concise summary."

    logger.info("Calling Summary LLM to update conversation summary.")
    try:
        response = summary_llm.invoke([HumanMessage(content=summary_llm_prompt)])
        new_summary = response.content.strip()
        log_summarization(logger, conversation_text, new_summary, "Conversation")

        # --- Manage Messages ---
        # Keep a system message with the summary and the last N messages
        # Adjust N (e.g., 4 = last 2 user turns) as needed
        N_MESSAGES_TO_KEEP = 4
        summary_system_message = SystemMessage(content=f"[CONVERSATION SUMMARY]:\n{new_summary}", name="ConversationSummary")
        messages_to_keep = [summary_system_message] + messages[-N_MESSAGES_TO_KEEP:]

        # Create RemoveMessage instructions for messages not kept
        # Note: MessagesState doesn't directly support replacing the whole list easily?
        # Checkpointer handles saving the final state. We modify the state dict here.
        # Let's just return the new message list and summary. LangGraph should handle state update.

        logger.info(f"Summarization complete. Kept summary message + last {N_MESSAGES_TO_KEEP} messages.")
        logger.process_end("summarize_conversation node finished")

        # Return the updated summary and the *new* list of messages for the state
        return {
            "summary": new_summary,
            "messages": messages_to_keep
        }

    except Exception as e:
        logger.error(f"Error during summarization LLM call: {e}", exc_info=True)
        logger.process_end("summarize_conversation node failed")
        # If summarization fails, return original state (or just skip summary update)
        # Returning empty dict means no changes from this node.
        return {}


@log_function(logger)
def should_summarize(state: AgentState) -> Literal["summarize_conversation", "__end__"]:
    """
    Determines whether the conversation is long enough to require summarization.
    """
    messages = state['messages']
    # Count only Human/AI messages for threshold decision
    human_ai_messages = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
    # Adjust threshold as needed (e.g., summarize every 5 turns = 10 messages)
    SUMMARY_THRESHOLD = 10
    if len(human_ai_messages) > SUMMARY_THRESHOLD:
        logger.info(f"Message count ({len(human_ai_messages)}) exceeds threshold ({SUMMARY_THRESHOLD}). Routing to summarization.")
        return "summarize_conversation"
    else:
        logger.info(f"Message count ({len(human_ai_messages)}) below threshold. Routing to end.")
        return "__end__" # Use "__end__" for the default END node


# --- Build LangGraph ---
if checkpointer:
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("summarize_conversation", summarize_conversation)

    # Define edges
    workflow.set_entry_point("call_llm") # Start with LLM call

    # After LLM call, decide whether to summarize or end
    workflow.add_conditional_edges(
        "call_llm",
        should_summarize,
        {
            "summarize_conversation": "summarize_conversation",
            "__end__": END
        }
    )

    # After summarization, end the current turn (summary is stored for the *next* turn)
    workflow.add_edge("summarize_conversation", END)

    # Compile the graph with the checkpointer
    try:
        app = workflow.compile(checkpointer=checkpointer)
        logger.info("LangGraph workflow compiled successfully with MongoDB checkpointer.")
    except Exception as e:
        logger.error(f"Failed to compile LangGraph workflow: {e}", exc_info=True)
        app = None
else:
    logger.error("LangGraph checkpointer failed to initialize. Workflow cannot be compiled.")
    app = None

# --- Export the compiled app ---
__all__ = ["app", "AgentState"] # Export the state definition too if needed elsewhere