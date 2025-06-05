from typing import Literal, Union, Dict, List, Optional, TypedDict
import os
import getpass
import logging
from datetime import datetime

from pymongo import MongoClient
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver

from config import (
    MONGO_CONNECTION_STRING,
    DATABASE_NAME,
    OPENAI_API_KEY
)

from utils.rich_logger import setup_logger
logger = setup_logger(level=logging.INFO)

# --- Initialize Environment & LLM ---
def _set_env(var: str, default: str | None = None):
    if not os.environ.get(var):
        if default:
            os.environ[var] = default
        else:
            os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY", OPENAI_API_KEY)

llm = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key=OPENAI_API_KEY
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = llm

# --- Define Persistent MongoDB Checkpointer ---
mongodb_client = MongoClient(MONGO_CONNECTION_STRING)
checkpointer = MongoDBSaver(mongodb_client)

# --- Define State and Functions ---
class State(MessagesState):
    summary: str
    chat_id: str
    username: Optional[str]
    business_info: Optional[str]
    ai_tone: Optional[str]

def call_model(state: State):
    """Process messages through the LLM and return response."""
    summary = state.get("summary", "")
    chat_id = state.get("chat_id", "")
    username = state.get("username", "")
    business_info = state.get("business_info", "")
    ai_tone = state.get("ai_tone", "دوستانه")
    
    system_messages = []
    if summary:
        system_messages.append(SystemMessage(content=f"Previous conversation summary: {summary}"))
    if business_info:
        system_messages.append(SystemMessage(content=f"Business Context: {business_info}"))
    if username:
        system_messages.append(SystemMessage(content=f"Address the user as: {username}"))
    
    system_messages.append(SystemMessage(content=f"Use this tone: {ai_tone}"))
    messages = system_messages + state["messages"]
    
    try:
        logger.info(f"Calling LLM for chat {chat_id}")
        response = model.invoke(messages)
        logger.info(f"LLM response received for chat {chat_id}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in call_model: {str(e)}")
        error_message = AIMessage(content="متأسفم، مشکلی در پردازش درخواست شما پیش آمد.")
        return {"messages": [error_message]}

def should_continue(state: State) -> Union[Literal["summarize_conversation"], Literal[END]]:
    """Decide whether to summarize the conversation or end."""
    messages = state["messages"]
    if len(messages) > 6:  # Adjust threshold as needed
        return "summarize_conversation"
    return END

def summarize_conversation(state: State):
    """Summarize the conversation and update state."""
    summary = state.get("summary", "")
    chat_id = state.get("chat_id", "")
    
    if summary:
        summary_message = (
            f"Previous summary: {summary}\n\n"
            "Please extend this summary with the new conversation above."
        )
    else:
        summary_message = "Please create a concise summary of this conversation:"

    try:
        logger.info(f"Summarizing conversation for chat {chat_id}")
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)
        
        # Keep only the last 2 messages plus the summary
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        
        # Save summary to database
        from db_manager import db_manager
        db_manager.save_conversation_summary(chat_id, response.content)
        
        logger.info(f"Conversation summarized for chat {chat_id}")
        return {
            "summary": response.content,
            "messages": delete_messages
        }
    except Exception as e:
        logger.error(f"Error in summarize_conversation: {str(e)}")
        return state

# --- Build LangGraph ---
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges(
    "conversation",
    should_continue,
    {
        "summarize_conversation": "summarize_conversation",
        END: END
    }
)
workflow.add_edge("summarize_conversation", END)

# --- Compile Graph ---
graph = workflow.compile(checkpointer=checkpointer)

def process_chat_update(
    message_content: Union[str, List],
    chat_id: str,
    username: Optional[str] = None,
    business_info: Optional[str] = None,
    ai_tone: Optional[str] = None
) -> Dict:
    """Process a chat update through the LangGraph."""
    try:
        input_message = HumanMessage(content=message_content)
        
        config = {
            "configurable": {
                "thread_id": f"chat_{chat_id}",
                "chat_id": chat_id,
                "username": username
            }
        }
        
        initial_state = {
            "messages": [input_message],
            "chat_id": chat_id,
            "username": username,
            "business_info": business_info,
            "ai_tone": ai_tone
        }
        
        logger.info(f"Processing chat update for chat {chat_id}")
        result = None
        for event in graph.stream(initial_state, config):
            if "messages" in event.get("conversation", {}):
                messages = event["conversation"]["messages"]
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        result = msg.content
                        
        return {"response": result} if result else {"error": "No response generated"}
        
    except Exception as e:
        logger.error(f"Error in process_chat_update: {str(e)}")
        return {"error": str(e)}

# Cleanup function
def cleanup():
    """Cleanup MongoDB connections."""
    try:
        mongodb_client.close()
        logger.info("MongoDB connections closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

import atexit
atexit.register(cleanup)
