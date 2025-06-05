# langgraph_agent.py
from typing import Literal, Union
import os
import getpass
import datetime
import time
import logging

from pymongo import MongoClient

from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver
from typing import Sequence, Union, Dict, Any, List, Optional
from typing_extensions import TypedDict
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Import necessary modules and configurations from telegram_bot.py and config.py
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
from telegram_bot import (
    llm,
    llm_business,
    user_llm,
    llm_summary,
    prompt_template_text,
    daily_task_prompt,
    summary_prompt,
    daily_report_prompt,
    insta_idea_prompt,
    image_analyzer_prompt,
    business_info_summary_prompt,
    welcome_message,
    help_text_prompt,
    prompt,
    MessageCounter,
    message_counter,
    get_user_business_info,
    get_mongo_collection,
    get_business_info_collection,
    summarize_business_info,
    process_business_info,
    get_user_info
)
from utils.rich_logger import setup_logger
from db_manager import db_manager
from utils.message_counter import MessageCounter, message_counter

logger = setup_logger(__name__, level=logging.INFO)


# --- Define Persistent MongoDB Checkpointer ---
mongodb_client = MongoClient(MONGO_CONNECTION_STRING)
checkpointer = MongoDBSaver(mongodb_client, database_name=DATABASE_NAME)

# --- Type Definitions for AgentState ---
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
    summary: Optional[str]


# --- Node Functions ---
def agent_node(state: AgentState):
    """
    Calls the LLM to generate a response from the formatted conversation.
    Handles both text-only and multimodal inputs. (Replaces original 'agent' function)
    """
    chat_id = state["chat_id"]
    username = state.get("username", "")
    logger.process_start(f"🧠 Processing agent node for chat: {chat_id}")
    logger.info(f"Username: {username if username else 'Not provided'}")

    ai_tone = db_manager.get_ai_tone(chat_id) #ai_tone_map.get(chat_id, "دوستانه") # Use db_manager
    business_info = db_manager.get_business_info(chat_id) #get_user_business_info(chat_id) # Use db_manager
    messages = state["messages"]

    try:
        from utils.date_helpers import get_full_shamsi_date
        current_shamsi_date = get_full_shamsi_date()
        logger.info(f"📅 Current Shamsi date: {current_shamsi_date}")
    except Exception as e:
        current_shamsi_date = "Unknown date"
        logger.error(f"❌ Error getting Shamsi date: {e}")

    user_report = get_user_info(chat_id) #get_user_info(chat_id) # Use db_manager through telegram_bot's get_user_info which uses db_manager
    from utils.rich_logger import log_user_business_data
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
            logger.info("[bold purple]🖼️ Preparing multimodal LLM request[/bold purple]")
            messages_for_llm = [
                {"role": "system", "content": prompt_template_text + (f"\n\nADDRESS USER: Always address the user directly as {username}" if username else "") +
                                             f"\n\nCURRENT DATE: Today's date in Iranian Shamsi calendar is {current_shamsi_date}."},
                {"role": "system", "content": f"BUSINESS CONTEXT: {business_info}\nTONE: {ai_tone}"}
            ]
            for msg in formatted_messages[:-1]:
                messages_for_llm.append(msg)
            messages_for_llm.append({"role": "user", "content": multimodal_content})
            from utils.rich_logger import log_comprehensive_interaction
            log_comprehensive_interaction(logger, chat_id, "", prompt_template_text, "Multimodal input", user_report, "N/A", ai_tone)
            logger.debug(f"Sending request to LLM with {len(messages_for_llm)} messages (multimodal)")
            response = llm.invoke(messages_for_llm)
        else:
            user_info = get_user_info(chat_id) #get_user_info(chat_id) # Use db_manager through telegram_bot's get_user_info which uses db_manager
            session_id = chat_session_map.get(chat_id, "N/A")
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
            log_comprehensive_interaction(logger, chat_id, session_id, prompt_template_text, f"name : {username}\n# user prompt : {prompt_input}\n\n", user_report, "N/A", ai_tone)
            logger.debug(f"Sending text-only request to LLM with {len(messages_for_llm)} messages")

            # Add color-coded logging before LLM invocation
            logger.info("[bold blue]" + "-"*40 + "[/bold blue]")
            logger.info(f"[purple]🤖 Calling {PRIMARY_LLM_MODEL}[/purple]") #PRIMARY_LLM_MODEL from telegram_bot.py implicitly available
            logger.info(f"[blue]👤 User: {username} | Chat: {chat_id} | Session: {session_id}[/blue]")
            logger.info(f"[cyan]📝 Input: {prompt_input[:50]}...[/cyan]")

            start_time = time.time()
            response = llm.invoke(messages_for_llm)
            duration = time.time() - start_time

            logger.info(f"[green]✨ Response generated in {duration:.2f}s[/green]")
            logger.info("[bold blue]" + "-"*40 + "[/bold blue]")

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
            logger.debug("Stripped thinking tags from response")
        logger.info(f"[green]📤 Response: {ai_message.content[:50]}...[/green]")
        from utils.rich_logger import log_ai_interaction
        log_ai_interaction(logger, prompt_input if not is_multimodal else "Multimodal content with image", response.content, PRIMARY_LLM_MODEL) #PRIMARY_LLM_MODEL from telegram_bot.py implicitly available
        logger.process_end("[bold blue]✅ Agent processing completed successfully[/bold blue]")

        new_state = state.copy()
        new_state["messages"] = state["messages"] + [ai_message]
        # Run summarization only every 5 messages (user + AI) # Changed to 5 to match message_counter in telegram_bot.py
        if message_counter.increment_and_check(chat_id):
            new_state = optimize_memory_node(new_state) # Call optimize_memory_node here
        return new_state
    except Exception as e:
        logger.error(f"[bold red]❌ Error during LLM call: {e}[/bold red]")
        error_message = AIMessage(content="متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.")
        logger.process_end("Agent processing failed with error")
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [error_message]
        return new_state


def optimize_memory_node(state: AgentState) -> AgentState:
    """
    Summarize conversation if there are more than 10 human/AI messages.
    Then keep only a summary and the last 2 messages. (Replaces original 'optimize_memory' function)
    """
    THRESHOLD = 10 # Kept as 10, original code used 10, message_counter uses 5 to trigger
    conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    if len(conv_messages) <= THRESHOLD:
        logger.info(f"Message count ({len(conv_messages)}) below threshold ({THRESHOLD}); skipping summarization")
        return state

    existing_summary = state.get("summary", "")
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conv_messages])
    try:
        full_prompt = summary_prompt.format(conversation=conversation_text, existing_summary=existing_summary)
    except Exception:
        full_prompt = f"{conversation_text}\n\n{existing_summary}" if existing_summary else conversation_text
    logger.process_start("Starting conversation summarization")
    try:
        start_time = time.time()
        summary_response = llm_summary.invoke([HumanMessage(content=full_prompt)])
        duration = time.time() - start_time
        new_summary = summary_response.content.strip()
        logger.info(f"Summarization completed in {duration:.2f}s: {new_summary[:100]}...")
        from utils.rich_logger import log_summarization
        log_summarization(logger, conversation_text, new_summary, "Conversation")
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        logger.process_end("Summarization failed")
        return state

    new_system = SystemMessage(content=f"[CONVERSATION SUMMARY]: {new_summary}")
    new_messages = [new_system] + conv_messages[-2:]
    logger.info(f"Trimmed conversation: kept {len(new_messages)} messages (summary + last 2 messages)")
    logger.process_end("Conversation summarization and trimming completed")
    state["summary"] = new_summary # Update summary in the state
    return {
        "messages": new_messages,
        "tool_calls": state.get("tool_calls", []),
        "requires_tool": state.get("requires_tool", False),
        "current_tool": state.get("current_tool", None),
        "chat_id": state["chat_id"],
        "username": state.get("username", None),
        "summary": new_summary, # Carry over summary in state
    }


# --- Build LangGraph ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("optimize_memory", optimize_memory_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", "optimize_memory") # Always optimize memory after agent for now
workflow.add_edge("optimize_memory", END)

app = workflow.compile(checkpointer=checkpointer)
logger.info("LangGraph application compiled successfully in langgraph_agent.py")


if __name__ == "__main__":
    # Example of how to run the graph (for testing purposes, remove from final version)
    from langchain.schema import HumanMessage, AIMessage, SystemMessage

    # Example usage in terminal (optional for testing)
    config = {"configurable": {"thread_id": "terminal_chat"}}
    user_id = "test_user_123"

    async def run_terminal_chat():
        print("Welcome to the LangGraph Terminal Chatbot! (langgraph_agent.py)")
        print("Type 'exit' or 'quit' to end the conversation.")

        history = [] # Simulate history for testing

        try:
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Ending conversation.")
                    break

                input_message = HumanMessage(content=user_input)
                print("AI is thinking...")

                inputs: AgentState = {
                    "messages": history + [input_message],
                    "tool_calls": [],
                    "requires_tool": False,
                    "current_tool": None,
                    "chat_id": user_id,
                    "username": "Test User",
                    "summary": "" # Initialize summary for testing
                }

                for event in app.stream(inputs, config, stream_mode="updates"):
                    for k, v in event.items():
                        if "messages" in v:
                            for m in v["messages"]:
                                if isinstance(m, AIMessage):
                                    print(f"\nAI: {m.content}")
                                    history.append(m) # Add AI response to history
                                elif isinstance(m, HumanMessage):
                                    history.append(m) # Add Human message to history
                        if "summary" in v:
                            print("\nSummary:")
                            print(v["summary"])
                            inputs["summary"] = v["summary"] # Update summary in inputs

        finally:
            mongodb_client.close()

    import asyncio
    asyncio.run(run_terminal_chat())