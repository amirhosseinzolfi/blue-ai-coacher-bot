# langgraph_agent.py
# -------------------------------------------------------------------
# Replaces the old langgraph logic and handles the new conversation
# workflow for your Telegram bot integration. 
# Terminal interaction is removed; function-based usage only.

import os
import datetime
from typing import Literal, Union, Optional, Dict, Any, List

from pymongo import MongoClient

# ---- LangChain & LangGraph Imports ----
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver

# ---- Your LLM + Tools Setup ----
# Example: Using openAI-like ChatOpenAI from your environment
# Adjust model_name/base_url/api_key to match your actual config
from langchain_openai import ChatOpenAI

# You can remove or replace embeddings if you don't need them:
# from langchain_ollama import OllamaEmbeddings

# -------------------------------------------------------------------
# ENV / LLM initialization - adjust as needed
# -------------------------------------------------------------------
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "test_db")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# Example LLM setup (replace base_url/model_name with your actual endpoint)
model_name = "gemini-2.0-flash"
llm = ChatOpenAI(
    base_url="http://localhost:15201/v1",
    model_name=model_name,
    temperature=0.5,
    api_key=OPENAI_API_KEY,
)

# Create the MongoDB checkpointer
mongodb_client = MongoClient(MONGO_CONNECTION_STRING)
checkpointer = MongoDBSaver.from_conn_string(
    MONGO_CONNECTION_STRING, 
    db_name=DATABASE_NAME,
    collection_name="langgraph_checkpoints"
)

# -------------------------------------------------------------------
# Define our custom State class
# -------------------------------------------------------------------
class ConversationState(MessagesState):
    """State object for conversation. Tracks a 'summary' as well."""
    summary: Optional[str]

# -------------------------------------------------------------------
# Graph Node 1: call_model
# -------------------------------------------------------------------
def call_model(state: ConversationState) -> Dict[str, Any]:
    """
    Call the primary LLM to generate the next response. 
    Optionally inject conversation summary as system context.
    """
    summary = state.get("summary", "")
    # If we have an existing summary, prepend as system context
    if summary:
        system_message = SystemMessage(content=f"[Conversation Summary So Far]: {summary}")
        messages = [system_message] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)  # returns an AIMessage

    return {
        "messages": [response]
    }

# -------------------------------------------------------------------
# Decide if we should continue or summarize
# -------------------------------------------------------------------
def should_summarize(state: ConversationState) -> Union[Literal["summarize"], Literal[END]]:
    """
    If there are more than 10 total messages, trigger summarization.
    Otherwise, finish the flow.
    """
    if len(state["messages"]) > 10:
        return "summarize"
    return END

# -------------------------------------------------------------------
# Graph Node 2: Summarize conversation
# -------------------------------------------------------------------
def summarize_conversation(state: ConversationState) -> Dict[str, Any]:
    """
    Summarizes the conversation and trims older messages,
    leaving only the last few plus the summary.
    """
    existing_summary = state.get("summary", "")
    prompt_text = (
        "Here is the conversation so far. Please create (or update) a concise summary:\n\n"
    )
    if existing_summary:
        prompt_text += f"Existing summary: {existing_summary}\n\n"
    
    # We ask the user (LLM) to produce a refined summary
    messages = state["messages"] + [HumanMessage(content=prompt_text)]
    response = llm.invoke(messages)  # Summarized text is in response.content

    new_summary = response.content.strip()

    # We can remove older messages from the conversation to reduce memory:
    # Keep the last 2 messages (if you like) or none:
    # For example, let's remove everything but the last 2 messages:
    trimmed_messages = state["messages"][-2:] if len(state["messages"]) >= 2 else state["messages"]

    # Return updated summary and the new trimmed messages
    return {
        "summary": new_summary,
        "messages": trimmed_messages
    }

# -------------------------------------------------------------------
# Build the StateGraph
# -------------------------------------------------------------------
workflow = StateGraph(ConversationState)
workflow.add_node("call_model", call_model)
workflow.add_node("summarize", summarize_conversation)

workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_summarize, ["summarize", END])
workflow.add_edge("summarize", END)

# Compile the graph with our MongoDB saver
app = workflow.compile(checkpointer=checkpointer)

# -------------------------------------------------------------------
# Public function: run_langgraph_agent
# -------------------------------------------------------------------
def run_langgraph_agent(
    input_message: HumanMessage,
    session_id: str
) -> str:
    """
    Runs the conversation flow on `app`. 
    Returns the final AI response text.
    """
    # For streaming or batched usage, we simply feed in the input state.
    # We specify config with `thread_id` = session_id to unify checkpointer usage.
    config = {"configurable": {"thread_id": session_id}}

    final_text = None

    for event in app.stream({"messages": [input_message]}, config=config):
        # event is a dictionary of node_name -> node_output
        # We look for the newly produced messages in each node
        for node_name, node_data in event.items():
            if "messages" in node_data:
                # The last item in node_data["messages"] might be an AIMessage
                for msg in node_data["messages"]:
                    if isinstance(msg, AIMessage):
                        final_text = msg.content  # track most recent response
    # If we never get an AI response, fallback:
    if not final_text:
        final_text = "متأسفم، پاسخی از هوش مصنوعی دریافت نشد."
    return final_text
