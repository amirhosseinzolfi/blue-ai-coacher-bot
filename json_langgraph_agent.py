#!/usr/bin/env python3
# json_agent.py

import json
from pathlib import Path

from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
from langchain.agents import create_json_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

# ——————————————————————————————————————————————————————————————
# 1. JSON “Database” Helpers
# ——————————————————————————————————————————————————————————————

DB_PATH = Path("db.json")

def load_db() -> list:
    """Load the JSON DB (list of dicts)."""
    if not DB_PATH.exists():
        return []
    return json.loads(DB_PATH.read_text(encoding="utf-8"))

def save_db(data: list) -> None:
    """Save the list of dicts to the JSON DB."""
    DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ——————————————————————————————————————————————————————————————
# 2. Define LangChain Tools
# ——————————————————————————————————————————————————————————————

@tool("add_entry", return_direct=True)
def add_entry(entry_json: str) -> str:
    """
    Add a new JSON object to the database.
    entry_json: JSON-formatted string representing the entry.
    """
    entry = json.loads(entry_json)
    db = load_db()
    db.append(entry)
    save_db(db)
    return f"✅ Added entry: {entry}"

@tool("list_entries", return_direct=True)
def list_entries() -> str:
    """
    Return all database entries as a JSON-formatted string.
    """
    return json.dumps(load_db(), indent=2)

@tool("delete_entry", return_direct=True)
def delete_entry(query_json: str) -> str:
    """
    Delete entries matching the query.
    query_json: JSON-formatted dict of key/value pairs to match.
    """
    query = json.loads(query_json)
    db = load_db()
    new_db = [e for e in db if not all(e.get(k) == v for k, v in query.items())]
    save_db(new_db)
    removed = len(db) - len(new_db)
    return f"🗑️ Removed {removed} entr{'y' if removed==1 else 'ies'}."


# ——————————————————————————————————————————————————————————————
# 3. Initialize Custom LLM & Embeddings
# ——————————————————————————————————————————————————————————————

llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gemini-1.5-flash",
    temperature=0.5,
    api_key="324"
)

# (Optional) if you need embeddings later
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ——————————————————————————————————————————————————————————————
# 4. Create the JSON Agent
# ——————————————————————————————————————————————————————————————

def build_agent(initial_data: dict = None):
    """
    Build and return a LangChain JSON agent with your tools and LLM.
    initial_data: dict to initialize the JSON spec (e.g. {"users": []}).
    """
    if initial_data is None:
        initial_data = {}

    # Create the JSON spec & toolkit
    json_spec    = JsonSpec(dict_=initial_data, max_value_length=4000)
    json_toolkit = JsonToolkit(spec=json_spec)

    # Instantiate the agent
    agent = create_json_agent(
        llm=llm,
        toolkit=json_toolkit,
        verbose=True
    )
    return agent


# ——————————————————————————————————————————————————————————————
# 5. Demo / CLI
# ——————————————————————————————————————————————————————————————

if __name__ == "__main__":
    import sys

    # Initialize DB file if missing
    if not DB_PATH.exists():
        save_db([])

    agent = build_agent(initial_data={"entries": load_db()})

    print("\nJSON Agent CLI. Type 'exit' to quit.")
    while True:
        prompt = input("\n> ")
        if prompt.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            sys.exit(0)
        try:
            response = agent.run(prompt)
        except Exception as e:
            response = f"Error: {e}"
        print(response)
