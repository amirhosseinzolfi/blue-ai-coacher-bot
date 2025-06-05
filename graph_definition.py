import os
import datetime
import logging
import time
import json
import threading
import re
from typing import Sequence, Union, Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

from config import (
    OPENAI_API_KEY,
    ai_tone_map, # Used by call_llm_node
)
from prompts.prompts import (
    PROMPT_TEMPLATE_TEXT,
    SUMMARY_PROMPT_TEXT,
    IMAGE_ANALYZER_PROMPT,
    IMAGE_ANALYZER_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_FALLBACK_PROMPT,
    IMAGE_ANALYSIS_ERROR_MESSAGE,
)

from utils.rich_logger import (
    log_summarization,
    log_user_business_data,
    log_comprehensive_interaction,
    setup_logger,
)
# Ensure db_manager can be imported if its functions are directly called here
# For now, db_manager calls are encapsulated in langgraph_code.py helpers or summarize_conversation_node
from db_manager import db_manager, save_message_to_history as db_save_message_to_history, get_user_info as db_get_user_info, get_business_info as db_get_business_info

logger = setup_logger(level=logging.INFO, logger_name="graph_definition")

############################################
# LLM Instances (Scoped to this graph if specific, or use imported ones)
############################################
# Assuming llm and llm_summary are passed or globally available if not defined here.
# For clarity, let's re-define or ensure they are accessible.
# These are defined in langgraph_code.py and can be imported or passed.
# To keep this file self-contained for graph logic, we might pass them.
# However, the original structure had them as globals in langgraph_code.
# For now, let's assume they are accessible as in the original langgraph_code.py context.

# Placeholder for LLMs that would be initialized in langgraph_code.py and used here
# This avoids re-definition if they are already globally available from langgraph_code.py
# If langgraph_code.py imports this module, it needs to ensure LLMs are set before graph use.
# A cleaner way would be to pass LLM instances to nodes if they become classes, or use a config object.

# For simplicity, let's assume llm and llm_summary are available from langgraph_code's scope
# when this graph is compiled and run.
# If direct import is preferred:
from langgraph_code import llm, llm_summary, PRIMARY_LLM_MODEL, image_analyze_llm


############################################
# AgentState Definition
############################################
class AgentState(TypedDict):
    messages: Sequence[Union[SystemMessage, HumanMessage, AIMessage]]
    chat_id: str
    session_id: str # Added session_id
    username: Optional[str]
    summary: Optional[str] # To store the latest conversation summary

############################################
# Node Definitions
############################################

def call_llm_node(state: AgentState) -> AgentState:
    """
    Calls the primary LLM to generate a response.
    """
    chat_id = state["chat_id"]
    username = state.get("username", "")
    current_messages = list(state["messages"]) # Make a mutable copy
    conversation_summary = state.get("summary", "")

    logger.process_start(f"🧠 Processing call_llm_node for chat: {chat_id}, session: {state['session_id']}")

    ai_tone = ai_tone_map.get(chat_id, "دوستانه") # Get from global config
    business_info = db_get_business_info(chat_id) # Use db_manager function

    # Load today's initial tasks - try database first, fallback to JSON
    try:
        from db_manager import get_user_tasks_for_date
        
        today = datetime.date.today().isoformat()
        today_tasks = get_user_tasks_for_date(chat_id, today)
        
        if today_tasks:
            # We have tasks from the database
            users_tasks_list = [f"{entry.get('user', 'Unknown')}: {entry.get('entry', '')}" for entry in today_tasks]
            users_today_initial_tasks = "\n".join(users_tasks_list)
        else:
            # Fallback to JSON file for backward compatibility
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, "database", "users_task.json")
            with open(db_path, "r", encoding="utf-8") as f:
                tasks_db = json.load(f)
            
            today_tasks_for_chat = tasks_db.get(str(chat_id), {}).get(today, {})
            users_tasks_list = []
            
            # Check if the today_tasks_for_chat is a dictionary (with user objects) or a list (with timestamps)
            if isinstance(today_tasks_for_chat, dict):
                # Old format with user objects
                for user, info in today_tasks_for_chat.items():
                    if isinstance(info, dict) and "to_do" in info:
                        users_tasks_list.append(f"{user}: {', '.join(info.get('to_do', []))}")
            elif isinstance(today_tasks_for_chat, list):
                # New format with timestamp entries
                users_tasks_list = [f"{entry.get('user', 'Unknown')}: {entry.get('entry', '')}" for entry in today_tasks_for_chat]
            
            users_today_initial_tasks = "\n".join(users_tasks_list) if users_tasks_list else "No tasks for today."
    except Exception as e:
        logger.error(f"Error loading tasks: {e}", exc_info=True)
        users_today_initial_tasks = "Error loading tasks."


    try:
        from utils.date_helpers import get_full_shamsi_date
        current_shamsi_date = get_full_shamsi_date()
    except Exception as e:
        current_shamsi_date = "Unknown date"
        logger.error(f"❌ Error getting Shamsi date: {e}")

    user_report = db_get_user_info(chat_id) # From db_manager
    log_user_business_data(logger, chat_id, business_info, user_report)

    # Multimodal input handling (similar to original agent function)
    last_message = current_messages[-1] if current_messages else None
    is_multimodal = False # Initialize is_multimodal
    text_content = "" # Initialize text_content
    image_urls = [] # Initialize image_urls
    multimodal_content = None # Initialize multimodal_content

    if isinstance(last_message, HumanMessage) and isinstance(last_message.content, list):
        is_multimodal = True # Set is_multimodal to True
        multimodal_content = last_message.content # Assign multimodal_content
        text_content = ""
        
        for item in multimodal_content:
            if item.get("type") == "text":
                text_content = item.get("text", "")
            elif item.get("type") == "image_url":
                image_urls.append(item.get("image_url", {}).get("url", ""))
        
        if not text_content and image_urls:
            text_content = "لطفا این تصویر را تحلیل کنید."
            found_text_item = False
            for item in multimodal_content:
                if item.get("type") == "text":
                    item["text"] = text_content
                    found_text_item = True
                    break
            if not found_text_item:
                multimodal_content.insert(0, {"type": "text", "text": text_content})
            current_messages[-1] = HumanMessage(content=multimodal_content) # Update the message in the list

        logger.info(f"Processing multimodal input. Text: '{text_content[:50]}...' Images: {len(image_urls)}")
    elif isinstance(last_message, HumanMessage):
        logger.info(f"Processing text input: {str(last_message.content)[:50]}...")
    else:
        logger.info("Processing non-human last message or empty messages.")

    # System Prompt Construction with Summary Integration
    formatted_system_prompt = PROMPT_TEMPLATE_TEXT.format(
        business_info=business_info,
        user_name=username or "کاربر",
        users_today_initial_tasks=users_today_initial_tasks
    )
    
    system_instruction_content = formatted_system_prompt
    
    # Add additional context
    system_instruction_content += (
        f"\n\nCURRENT DATE: {current_shamsi_date}" +
        "\n\nINPUT FORMAT: User messages are formatted with clear sections (, 👤 user :, 💬 MESSAGE:, ↩️ REPLYING TO:). " +
        "Parse these sections carefully to understand the user's request and context."
    )
    
    # Add conversation summary to system instruction if available - moved to end for better context
    if conversation_summary:
        system_instruction_content += (
            "\n\n## CONVERSATION HISTORY SUMMARY\n"
            f"{conversation_summary}\n\n"
            "Use this summary as context for the current conversation, but prioritize the user's immediate query."
        )
    
    system_message = SystemMessage(content=system_instruction_content)
    
    # current_messages should already be clean of any prior summary SystemMessages
    # due to filtering in run_agent and how summarize_conversation_node returns messages.
    # Thus, no need to filter current_messages here again.
    messages_for_llm = [system_message] + current_messages

    final_user_prompt_content = current_messages[-1].content if current_messages else ""

    logger.info(f"📨 call_llm_node: invoking LLM with {len(messages_for_llm)} messages (system + conversation).")
    try:
        start_time = time.time()
        
        # Direct non-streaming LLM call
        response = llm.invoke(messages_for_llm)
        
        duration = time.time() - start_time
        ai_response_content = response.content
        ai_message = AIMessage(content=ai_response_content)

    except Exception as e:
        logger.error(f"[bold red]❌ Error during LLM call in call_llm_node: {e}[/bold red]", exc_info=True)
        
        # Check if this was a multimodal input - if so, try the specialized image analyzer
        if is_multimodal and image_urls: # Ensure is_multimodal is defined and True
            try:
                logger.info("🖼️ Attempting fallback with specialized image analyzer LLM...")
                
                # Create a comprehensive prompt for the image analyzer using centralized prompts
                image_analyzer_system_prompt = IMAGE_ANALYZER_SYSTEM_PROMPT.format(
                    IMAGE_ANALYZER_PROMPT=IMAGE_ANALYZER_PROMPT,
                    business_info=business_info
                )
                
                # Send the multimodal content to the image analyzer
                # Ensure multimodal_content is defined
                if multimodal_content is None and last_message and isinstance(last_message.content, list):
                    multimodal_content = last_message.content

                image_analysis_messages = [
                    SystemMessage(content=image_analyzer_system_prompt),
                    HumanMessage(content=multimodal_content) # Use the captured multimodal_content
                ]
                
                logger.info(f"Sending multimodal content to image analyzer: text='{text_content[:50]}...' images={len(image_urls)}")
                # Use the imported image_analyze_llm instance
                image_analyzer_response = image_analyze_llm.invoke(image_analysis_messages)
                image_analysis_text = image_analyzer_response.content
                logger.info(f"✅ Image analyzer provided comprehensive analysis: {image_analysis_text[:100]}...")
                
                # Now create a text-only message to send to the main LLM with the analysis
                comprehensive_text_message = IMAGE_ANALYSIS_FALLBACK_PROMPT.format(
                    text_content=text_content,
                    image_analysis_text=image_analysis_text
                )
                
                # Replace multimodal message with comprehensive text message
                text_only_human_message = HumanMessage(content=comprehensive_text_message)
                current_messages[-1] = text_only_human_message
                
                # Create new messages list with the updated content
                messages_for_llm = [system_message] + current_messages
                
                # Try main LLM again with the comprehensive analysis
                logger.info("🔄 Retrying main LLM with comprehensive image analysis...")
                retry_response = llm.invoke(messages_for_llm)
                ai_response_content = retry_response.content
                ai_message = AIMessage(content=ai_response_content)
                logger.info(f"✅ Main LLM responded successfully after image analysis fallback: {ai_response_content[:100]}...")
                
            except Exception as fallback_error:
                logger.error(f"❌ Image analyzer fallback also failed: {fallback_error}", exc_info=True)
                
                # If even the image analyzer fails, provide a helpful error message using centralized prompt
                error_message_content = IMAGE_ANALYSIS_ERROR_MESSAGE.format(
                    text_content=text_content,
                    image_count=len(image_urls)
                )
                
                error_ai_message = AIMessage(content=error_message_content)
                logger.process_end("call_llm_node processing failed with both primary and image analyzer fallback methods")
                updated_messages = current_messages + [error_ai_message]
                return {**state, "messages": updated_messages}
        else:
            # If not multimodal or no fallback available, return error message
            error_message_content = "متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید."
            error_ai_message = AIMessage(content=error_message_content)
            logger.process_end("call_llm_node processing failed with error")
            updated_messages = current_messages + [error_ai_message]
            return {**state, "messages": updated_messages}

    # enriched interaction log
    # Pass current_messages (actual conversation turns) as history for logging
    log_comprehensive_interaction(
        logger,
        chat_id,
        state["session_id"],
        system_instruction_content, # This is the full system prompt including summary
        str(final_user_prompt_content),
        ai_message.content,
        ai_tone,
        PRIMARY_LLM_MODEL,
        history=[m.content if hasattr(m, "content") else str(m) for m in current_messages], # Log actual turns
        summary=state.get("summary", "") # Log the summary string itself for reference
    )

    if username:
        for placeholder in ["[نام کاربر]", "[name]", "[نام]"]:
            ai_message.content = ai_message.content.replace(placeholder, username)
    
    from utils.helpers import strip_thinking_tags
    ai_message.content = strip_thinking_tags(ai_message.content)
    
    logger.info(f"[green]📤 LLM Response: {ai_message.content[:50]}...[/green] (Took {duration:.2f}s)")
    logger.process_end("[bold blue]✅ call_llm_node processing completed successfully[/bold blue]")
    
    updated_messages = current_messages + [ai_message]
    return {**state, "messages": updated_messages}


def summarize_conversation_node(state: AgentState) -> AgentState:
    """
    Summarizes the conversation if needed and trims the history.
    Uses RemoveMessage to explicitly mark messages for removal.
    """
    chat_id = state["chat_id"]
    session_id = state["session_id"]
    current_messages = list(state["messages"])
    logger.process_start(f"🔄 Processing summarize_conversation_node for chat: {chat_id}, session: {session_id}")

    # Extract previous summary from state or first system message
    previous_summary = state.get("summary", "")
    
    # Filter out summary message if present for summarization
    messages_to_summarize = current_messages
    if messages_to_summarize and isinstance(messages_to_summarize[0], SystemMessage) and \
       "[CONVERSATION SUMMARY]:" in messages_to_summarize[0].content:
        messages_to_summarize = messages_to_summarize[1:]

    # Format conversation for summarization
    conversation_text_for_summary = "\n".join([
        f"[{msg.__class__.__name__}]: {msg.content}" for msg in messages_to_summarize
    ])

    # Create summarization prompt
    full_prompt_for_summary = SUMMARY_PROMPT_TEXT.format(
        previous_summary=previous_summary,
        conversation_text=conversation_text_for_summary
    )

    logger.info(f"📝 summarize_conversation_node: {len(messages_to_summarize)} messages to summarize; previous_summary length: {len(previous_summary or '')}.")
    try:
        start_time = time.time()
        
        # Direct non-streaming summarization call
        summary_response = llm_summary.invoke([HumanMessage(content=full_prompt_for_summary)])
        
        duration = time.time() - start_time
        new_summary_content = summary_response.content.strip()
        logger.info(f"Summarization completed in {duration:.2f}s: {new_summary_content[:100]}...")
        log_summarization(logger, conversation_text_for_summary, new_summary_content, "Conversation")

        # Save the summary to database
        db_manager.save_conversation_summary(chat_id, new_summary_content)
        
        # Keep only last user message and AI response
        last_ai_message = None
        last_human_message = None
        
        if len(messages_to_summarize) >= 1 and isinstance(messages_to_summarize[-1], AIMessage):
            last_ai_message = messages_to_summarize[-1]
        if len(messages_to_summarize) >= 2 and isinstance(messages_to_summarize[-2], HumanMessage):
            last_human_message = messages_to_summarize[-2]
        
        # Generate RemoveMessage tokens for all messages except the last Q&A pair
        messages_to_remove = []
        for msg in current_messages:
            if (last_human_message and msg.id == last_human_message.id) or \
               (last_ai_message and msg.id == last_ai_message.id):
                continue
            messages_to_remove.append(RemoveMessage(id=msg.id))
        
        # Update database with trimmed history
        db_manager.clear_chat_history(chat_id, session_id=session_id)
        
        # Save trimmed messages to database
        from langgraph_code import save_message_to_history as lgc_save_message_to_history
        
        preserved_messages = []
        if last_human_message:
            preserved_messages.append(last_human_message)
            lgc_save_message_to_history(chat_id, last_human_message.type, last_human_message.content, session_id=session_id)
        if last_ai_message:
            preserved_messages.append(last_ai_message)
            lgc_save_message_to_history(chat_id, last_ai_message.type, last_ai_message.content, session_id=session_id)
            
        logger.info(f"Trimmed conversation for session {session_id}: kept {len(preserved_messages)} messages (last interaction).")
        logger.process_end(f"✅ summarize_conversation_node completed for session {session_id}.")

        # Return updated state with summary property but not as a message
        return {
            **state, 
            "messages": messages_to_remove + (preserved_messages or []), 
            "summary": new_summary_content
        }

    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        logger.process_end(" summarize_conversation_node failed.")
        return state

############################################
# Conditional Edges
############################################
def should_summarize(state: AgentState) -> str:
    """
    Determines if the conversation is long enough to require summarization.
    """
    messages = state["messages"]
    interaction_messages = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
    interaction_messages_count = len(interaction_messages)
    
    summarization_threshold = 10 # Lowered from 15 to trigger summarization sooner (e.g., after 4 user turns)
    
    if interaction_messages_count >= summarization_threshold:
        logger.info(f"Conversation length ({interaction_messages_count} interactions) meets/exceeds threshold ({summarization_threshold}) for session {state['session_id']}. Routing to summarization.")
        return "summarize_conversation_node"
    else:
        logger.info(f"Conversation length ({interaction_messages_count} interactions) within threshold for session {state['session_id']}. Skipping summarization.")
        return END

############################################
# Graph Construction
############################################
graph_builder = StateGraph(AgentState)

graph_builder.add_node("call_llm_node", call_llm_node)
graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

graph_builder.add_edge(START, "call_llm_node")

graph_builder.add_conditional_edges(
    "call_llm_node",
    should_summarize,
    {
        "summarize_conversation_node": "summarize_conversation_node",
        END: END
    }
)
graph_builder.add_edge("summarize_conversation_node", END)

# Modified to allow checkpointer to be passed during compilation
def get_compiled_app(checkpointer=None):
    return graph_builder.compile(checkpointer=checkpointer)

app = get_compiled_app() # Default compilation without checkpointer

logger.info("✅ LangGraph agent graph compiled successfully.")

# To make `app` importable
__all__ = ['app', 'AgentState', 'get_compiled_app']
