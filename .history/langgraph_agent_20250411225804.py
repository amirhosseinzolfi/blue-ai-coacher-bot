# langgraph_agent.py
from typing import Literal, Union, Dict
import os
import getpass

from pymongo import MongoClient

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver

from telegram_bot import llm, llm_summary, AgentState, optimize_memory  # Import existing llms and AgentState

# --- Initialize Environment & LLM (using existing ones from telegram_bot.py) ---
# def _set_env(var: str, default: str | None = None):
#     if not os.environ.get(var):
#         if default:
#             os.environ[var] = default
#         else:
#             os.environ[var] = getpass.getpass(f"{var}: ")
#
# # Set OpenAI API key with default value
# _set_env("OPENAI_API_KEY", "234")
#
# llm = ChatOpenAI(
#     base_url="http://localhost:15209/v1",
#     model_name="gemini-2.0-flash",
#     temperature=0.5,
#     api_key="324"
# )

# embeddings = OllamaEmbeddings(model="nomic-embed-text") # Not used in this graph
model = llm  # Use the llm instance from telegram_bot.py

# --- Define Persistent MongoDB Checkpointer (assuming configuration is in telegram_bot.py or config.py) ---
from config import MONGO_CONNECTION_STRING, DATABASE_NAME # Assuming these are in config.py or telegram_bot.py
MONGODB_URI = MONGO_CONNECTION_STRING  # Use from config
mongodb_client = MongoClient(MONGODB_URI)

checkpointer = MongoDBSaver.from_conn_string(
            MONGO_CONNECTION_STRING,
            db_name=DATABASE_NAME,
            collection_name="langgraph_checkpoints"
        )

# --- Define State and Functions ---
# class State(MessagesState): # Use existing AgentState from telegram_bot.py
#     summary: str

def call_model(state: AgentState): # Use AgentState
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Union[Literal["summarize_conversation"], Literal[END]]: # Use AgentState
    messages = state["messages"]
    if len([msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]) > 10: # Adjusted threshold to match telegram_bot.py logic
        return "summarize_conversation"
    return END

def summarize_conversation_node(state: AgentState): # Renamed to avoid conflict with telegram_bot's function, and using AgentState
    # Use llm_summary from telegram_bot.py for summarization
    llm_summary_instance = llm_summary
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conv_messages])

    from prompts.prompts import SUMMARY_PROMPT_TEXT # Assuming prompts.prompts is accessible
    summary_prompt = SUMMARY_PROMPT_TEXT

    try:
        full_prompt = summary_prompt.format(conversation=conversation_text, existing_summary=summary)
    except Exception:
        full_prompt = f"{conversation_text}\n\n{summary}" if summary else conversation_text

    messages = [HumanMessage(content=full_prompt)]
    response = llm_summary_instance.invoke(messages) # Use llm_summary_instance here
    new_summary = response.content.strip()
    new_system = SystemMessage(content=f"[CONVERSATION SUMMARY]: {new_summary}")
    conv_messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))] # Re-filter to only summarize conversation messages
    new_messages = [new_system] + conv_messages[-2:] # Keep summary and last 2 conv messages


    return {"summary": new_summary, "messages": new_messages} # Return updated messages


# --- Build LangGraph ---
workflow = StateGraph(AgentState) # Use AgentState from telegram_bot.py
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation_node) # Renamed node function
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue, {"summarize_conversation": "summarize_conversation", END: END}) # Updated edge definition
workflow.add_edge("summarize_conversation", END)

app = workflow.compile(checkpointer=checkpointer)