from typing import Literal, Union, Dict, Optional, TypedDict
import os
import logging
from datetime import datetime

from pymongo import MongoClient
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver

from config import (
    MONGO_CONNECTION_STRING,
    DATABASE_NAME,
    OPENAI_API_KEY,
    PRIMARY_LLM_MODEL,
    SUMMARY_LLM_MODEL
)

from utils.rich_logger import setup_logger
logger = setup_logger(level=logging.INFO)

# --- Enhanced State Definition ---
class State(MessagesState):
    summary: str
    chat_id: str
    username: Optional[str]
    business_info: Optional[str]
    ai_tone: Optional[str]
    thread_id: Optional[str]

# --- Initialize LLM Models ---
conversation_model = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=PRIMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)

summary_model = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=SUMMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)

# --- MongoDB Checkpointer Setup ---
mongodb_client = MongoClient(MONGO_CONNECTION_STRING)
checkpointer = MongoDBSaver(mongodb_client)

# --- Enhanced Graph Functions ---
def call_model(state: State):
    """Process messages with context and generate response."""
    chat_id = state.get("chat_id", "")
    username = state.get("username", "")
    business_info = state.get("business_info", "")
    ai_tone = state.get("ai_tone", "دوستانه")
    summary = state.get("summary", "")

    system_messages = []
    if summary:
        system_messages.append(SystemMessage(content=f"Previous conversation summary: {summary}"))
    if business_info:
        system_messages.append(SystemMessage(content=f"Business Context: {business_info}"))
    if username:
        system_messages.append(SystemMessage(content=f"Address user as: {username}"))
    
    system_messages.append(SystemMessage(content=f"Use this tone: {ai_tone}"))
    messages = system_messages + state["messages"]
    
    logger.info(f"Generating response for chat {chat_id}")
    response = conversation_model.invoke(messages)
    
    return {"messages": [response]}

def should_summarize(state: State) -> Union[Literal["summarize_conversation"], Literal[END]]:
    """Decide whether to summarize based on message count."""
    messages = state["messages"]
    if len(messages) > 8:  # Adjust threshold as needed
        return "summarize_conversation"
    return END

def summarize_conversation(state: State):
    """Generate conversation summary and update state."""
    chat_id = state.get("chat_id", "")
    summary = state.get("summary", "")
    
    if summary:
        summary_prompt = (
            f"Previous summary: {summary}\n\n"
            "Please extend this summary with the new conversation content:"
        )
    else:
        summary_prompt = "Create a concise summary of this conversation:"

    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    
    try:
        logger.info(f"Summarizing conversation for chat {chat_id}")
        response = summary_model.invoke(messages)
        
        # Keep last 2 messages plus summary
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        
        # Save summary to database
        from db_manager import db_manager
        db_manager.save_conversation_summary(chat_id, response.content)
        
        return {
            "summary": response.content,
            "messages": delete_messages
        }
    except Exception as e:
        logger.error(f"Summary error for chat {chat_id}: {str(e)}")
        return state

# --- Build Enhanced Graph ---
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_summarize, 
                             {"summarize_conversation": "summarize_conversation", END: END})
workflow.add_edge("summarize_conversation", END)

# --- Compile Graph ---
graph = workflow.compile(checkpointer=checkpointer)

def process_message(
    message_content: Union[str, list],
    chat_id: str,
    username: Optional[str] = None,
    business_info: Optional[str] = None,
    ai_tone: Optional[str] = None,
    thread_id: Optional[str] = None
) -> Dict:
    """Process a message through the LangGraph."""
    try:
        input_message = HumanMessage(content=message_content)
        
        config = {
            "configurable": {
                "thread_id": thread_id or f"chat_{chat_id}",
            }
        }
        
        initial_state = {
            "messages": [input_message],
            "chat_id": chat_id,
            "username": username,
            "business_info": business_info,
            "ai_tone": ai_tone,
            "thread_id": thread_id
        }
        
        logger.info(f"Processing message for chat {chat_id}")
        result = None
        
        for event in graph.stream(initial_state, config):
            if "messages" in event.get("conversation", {}):
                messages = event["conversation"]["messages"]
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        result = msg.content
                        
        return {"response": result} if result else {"error": "No response generated"}
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {"error": str(e)}

# --- Cleanup Function ---
def cleanup():
    """Clean up MongoDB connections."""
    try:
        mongodb_client.close()
        logger.info("MongoDB connections closed")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

import atexit
atexit.register(cleanup)
