from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
import os
import getpass

# --- Initialize LLM ---
llm = ChatOpenAI(
    base_url="http://localhost:15209/v1",
    model_name="gpt-4o",
    temperature=0.5,
    api_key="324"
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

memory = MemorySaver() # Using MemorySaver for now, can be replaced with MongoDB later

# Define state class to include conversation summary
class State(MessagesState):
    summary: str

# Use the initialized LLM
model = llm

# Define the logic to call the model
def call_model(state: State):
    # If a summary exists, add it as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

from typing import Union

# Define logic to decide whether to summarize or end the conversation
def should_continue(state: State) -> Union[Literal["summarize_conversation"], Literal[END]]:
    """Return the next node to execute."""
    messages = state["messages"]
    # Summarize conversation if it exceeds a certain length (e.g., 6 messages)
    if len(messages) > 6:
        return "summarize_conversation"
    return END

# Define function to summarize the conversation
def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    # Remove older messages, keep only the last two
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Define the LangGraph workflow
workflow = StateGraph(State)

# Define nodes for conversation and summarization
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# Set the entrypoint
workflow.add_edge(START, "conversation")

# Conditional edge to summarize or end conversation
workflow.add_conditional_edges(
    "conversation",
    should_continue,
    ["summarize_conversation", END]
)

# Edge from summarization to end
workflow.add_edge("summarize_conversation", END)

# Compile the workflow
app = workflow.compile(checkpointer=memory) # Using memory checkpointer

# Function to print updates from the agent in terminal
def print_update_terminal(update):
    for k, v in update.items():
        if "messages" in v:
            for m in v["messages"]:
                if isinstance(m, AIMessage):
                    print(f"\nAI: {m.content}")
                elif isinstance(m, HumanMessage):
                    pass # Don't print human message again as it's user input
        if "summary" in v:
            print("\nSummary:")
            print(v["summary"])

# Terminal interaction loop
config = {"configurable": {"thread_id": "terminal_chat"}} # Unique thread ID for terminal chat

print("Welcome to the LangGraph Terminal Chatbot!")
print("Type 'exit' or 'quit' to end the conversation.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Ending conversation.")
        break

    input_message = HumanMessage(content=user_input)
    print("AI is thinking...") # Indicate that the AI is processing

    for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
        print_update_terminal(event)
