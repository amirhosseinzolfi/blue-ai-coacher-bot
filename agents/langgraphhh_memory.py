import uuid
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from typing import TypedDict, List, Dict

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the state schema
class State(TypedDict):
    foo: str
    bar: List[str]

# Initialize the LLM
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gpt-4o",
    api_key="324",
    temperature=0.5
)
logging.info("Primary LangChain LLM initialized.")

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI
db = mongo_client["memory_db"]  # Database name
memory_collection = db["memories"]  # Collection name

# Define the nodes in the graph
def node_a(state: State) -> Dict[str, str]:
    prompt = "Generate a response for node A."
    response = llm.invoke(prompt)
    return {"foo": response.content, "bar": ["a"]}

def node_b(state: State) -> Dict[str, str]:
    prompt = "Generate a response for node B."
    response = llm.invoke(prompt)
    return {"foo": response.content, "bar": state["bar"] + ["b"]}

# Create the StateGraph
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# Set up the checkpointer
checkpointer = MemorySaver()

# Compile the graph with the checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# Define user ID and namespace for memory
user_id = "1"

# Function to load memory from MongoDB
def load_memory():
    memories = memory_collection.find({"user_id": user_id})
    for memory in memories:
        # Assuming memory has a structure with 'memory_id' and 'data'
        memory_id = memory['memory_id']
        data = memory['data']
        # You can store this in a suitable structure or directly use it
        print(f"Loaded memory: {memory_id} -> {data}")

# Function to update memory in MongoDB
def update_memory(data: dict):
    memory_id = str(uuid.uuid4())
    memory_entry = {
        "user_id": user_id,
        "memory_id": memory_id,
        "data": data
    }
    memory_collection.insert_one(memory_entry)
    logging.info(f"Memory updated: {memory_entry}")

# Load existing memory at the start
load_memory()

# Invoke the graph with a specific thread ID
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar": []}, config)

# Update memory after the first invocation
update_memory({"food_preference": "I like pizza"})

# Retrieve and print the latest state
latest_state = graph.get_state(config)
print("Latest State:", latest_state)

# Invoke the graph again to demonstrate persistence
config = {"configurable": {"thread_id": "2", "user_id": user_id}}
graph.invoke({"foo": "Hello again!", "bar": []}, config)

# Retrieve and print the updated state
updated_state = graph.get_state(config)
print("Updated State:", updated_state)
