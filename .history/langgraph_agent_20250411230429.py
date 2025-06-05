from typing import Literal, Union, Dict, Any, List, Optional
import os
import datetime
import threading
import time
import logging

from pymongo import MongoClient

from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain.schema import BaseMessage

# --- Local Imports and Configuration ---
from config import (
    MONGO_CONNECTION_STRING,
    DATABASE_NAME,
    COLLECTION_NAME,
    OPENAI_API_KEY
)
from prompts.prompts import SUMMARY_PROMPT, PROMPT_TEMPLATE_TEXT
from utils.rich_logger import setup_logger, log_summarization, log_ai_interaction
from langchain_openai import ChatOpenAI

logger = setup_logger(level=logging.INFO)

# --- LLM Model Definitions and Instance Setup ---
PRIMARY_LLM_MODEL = "gpt-4o"
SUMMARY_LLM_MODEL = "gemini-2.0-flash"

llm = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=PRIMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Primary LangChain LLM initialized with model: {PRIMARY_LLM_MODEL}.")

llm_summary = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=SUMMARY_LLM_MODEL,
    temperature=0.5,
    api_key=OPENAI_API_KEY
)
logger.info(f"Summary LLM initialized with model: {SUMMARY_LLM_MODEL}.")

# --- MongoDB Client ---
mongodb_client = MongoClient(MONGO_CONNECTION_STRING)

# --- Define State ---
class AgentState(MessagesState):
    summary: Optional[str] = None
    chat_id: str
    username: Optional[str] = None

# --- Helper Functions ---
def get_mongodb_checkpointer():
    return MongoDBSaver.from_conn_string(
        MONGO_CONNECTION_STRING,
        db_name=DATABASE_NAME,
        collection_name="langgraph_checkpoints"
    )

def get_mongo_collection():
    db = mongodb_client[DATABASE_NAME]
    logger.info("Connected to MongoDB (database: '%s').", DATABASE_NAME)
    return db[COLLECTION_NAME]

# --- Define Graph Nodes ---
def call_model(state: AgentState):
    """
    Calls the LLM to generate a response.
    """
    messages = state["messages"]
    chat_id = state["chat_id"]
    username = state.get("username", "")
    summary = state.get("summary", "")

    # Include summary in the system message if available
    system_message_content = PROMPT_TEMPLATE_TEXT
    if summary:
        system_message_content += f"\nConversation Summary:\n{summary}"

    messages = [SystemMessage(content=system_message_content)] + messages

    logger.info(f"Calling LLM with {len(messages)} messages")
    start_time = time.time()
    response = llm.invoke(messages)
    duration = time.time() - start_time
    logger.info(f"LLM call completed in {duration:.2f}s")

    log_ai_interaction(logger, messages[-1].content, response.content, PRIMARY_LLM_MODEL)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Union[Literal["summarize_conversation"], Literal[END]]:
    """
    Determines whether to continue the conversation or summarize.
    """
    messages = state["messages"]
    if len(messages) > 6:
        logger.info("Message count exceeds threshold, routing to summarization")
        return "summarize_conversation"
    else:
        logger.info("Message count below threshold, ending conversation")
        return END

def summarize_conversation(state: AgentState):
    """
    Summarizes the conversation.
    """
    messages = state["messages"]
    chat_id = state["chat_id"]
    username = state.get("username", "")
    summary = state.get("summary", "")

    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    full_prompt = SUMMARY_PROMPT.format(conversation=conversation_text, existing_summary=summary)

    logger.info("Starting conversation summarization")
    start_time = time.time()
    summary_response = llm_summary.invoke([HumanMessage(content=full_prompt)])
    duration = time.time() - start_time
    new_summary = summary_response.content.strip()
    logger.info(f"Summarization completed in {duration:.2f}s: {new_summary[:100]}...")
    log_summarization(logger, conversation_text, new_summary, "Conversation")

    logger.info("Summarization completed")
    # delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]] # this line cause error
    return {"summary": new_summary, "messages": []}

# --- Build LangGraph ---
def create_graph():
    """
    Creates and compiles the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue, {
        "summarize_conversation": "summarize_conversation",
        "continue": END
    })
    workflow.add_edge("summarize_conversation", END)
    return workflow

def compile_graph():
    """
    Compiles the LangGraph workflow with MongoDB checkpointer.
    """
    checkpointer = get_mongodb_checkpointer()
    workflow = create_graph()
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph workflow compiled successfully.")
    return app

# --- Example Usage ---
if __name__ == "__main__":
    # This is just for demonstration and won't be used in the actual bot
    print("LangGraph agent definition. Use compile_graph() to get the compiled graph.")
    graph = compile_graph()
    print(graph)
