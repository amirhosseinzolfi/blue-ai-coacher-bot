#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

# --- Database helpers ---
DB_PATH = Path("db.json")

def load_db() -> list:
    """Load the JSON database as a list of dicts."""
    if not DB_PATH.exists():
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data: list) -> None:
    """Save the list of dicts to the JSON database."""
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def validate_and_parse_json(json_str: str) -> tuple[dict, str]:
    """
    Validate and parse JSON string.
    Returns (parsed_dict, error_message). If successful, error_message is empty.
    """
    json_str = json_str.strip()
    
    # Remove any trailing text that might come after the JSON on subsequent lines
    if '\n' in json_str:
        json_str = json_str.split('\n')[0].strip()
    
    # Remove UTF-8 BOM if present, as it can cause json.loads to fail
    if json_str.startswith('\ufeff'):
        print("DEBUG: UTF-8 BOM detected and removed.")
        json_str = json_str.lstrip('\ufeff')
            
    if not json_str:
        return None, "Empty JSON string provided."
    
    # Debug: print what we received
    print(f"DEBUG: Final cleaned JSON string for parsing: {repr(json_str)}")
    
    try:
        parsed = json.loads(json_str)
        return parsed, ""
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON format - {e}. Ensure all strings are quoted and JSON is valid. Received: {repr(json_str[:100])}"

# --- Tool definitions ---
@tool("add_entry", return_direct=True)
def add_entry(entry_json: str) -> str:
    """
    Add a new JSON object to the database.
    entry_json: a JSON-formatted string representing the entry.
    """
    entry, error = validate_and_parse_json(entry_json)
    if error:
        return f"Error: {error}"
    
    db = load_db()
    db.append(entry)
    save_db(db)
    return f"Added entry: {entry}"

@tool("list_entries", return_direct=True)
def list_entries() -> str:
    """Return all database entries."""
    return json.dumps(load_db(), indent=2)

@tool("get_entry", return_direct=True)
def get_entry(query_json: str) -> str:
    """
    Retrieve entries matching some condition.
    query_json: JSON-formatted dict of key/value to filter by.
    """
    query, error = validate_and_parse_json(query_json)
    if error:
        return f"Error: {error}"
    
    results = [e for e in load_db() if all(e.get(k) == v for k, v in query.items())]
    return json.dumps(results, indent=2)

@tool("delete_entry", return_direct=True)
def delete_entry(query_json: str) -> str:
    """
    Delete entries matching some condition.
    query_json: JSON-formatted dict of key/value to filter by.
    """
    query, error = validate_and_parse_json(query_json)
    if error:
        return f"Error: {error}"
    
    db = load_db()
    new_db = [e for e in db if not all(e.get(k) == v for k, v in query.items())]
    save_db(new_db)
    deleted = len(db) - len(new_db)
    return f"Deleted {deleted} entr{'y' if deleted == 1 else 'ies'}."

# --- Initialize LLM and Embeddings ---
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gpt-4o",
    temperature=0.5,
    api_key="324"
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # optional, for semantic search

# --- Assemble the agent ---
tools = [add_entry, list_entries, get_entry, delete_entry]

# Create a more robust prompt template
prompt = PromptTemplate.from_template("""
You are a helpful assistant that manages a JSON database using ReAct format.

Available tools:
{tools}

Tool names: {tool_names}

CRITICAL RULES:
1. Action Input must be ONLY valid JSON on a single line.
2. Do NOT include any text, newlines, or comments after the JSON object.
3. Ensure all strings within the JSON are enclosed in double quotes.
4. Do NOT output "Observation:" as part of the Action Input.

CORRECT FORMAT:
Question: Add John to database  
Thought: I need to add John to the database.
Action: add_entry
Action Input: {{"name": "John"}}
Observation: [tool output will appear here]
Thought: [your analysis of the observation]
Final Answer: [your response to the user]

INCORRECT FORMAT (DO NOT DO THIS):
Action Input: {{"name": "John"}} 
This is extra text.
Observation: [This text should NOT be in Action Input]

Question: {input}
{agent_scratchpad}Thought:
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=3)

# --- CLI interface ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="JSON DB Manager Agent CLI (interactive or direct commands)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_add = subparsers.add_parser("add", help="Add a new entry")
    parser_add.add_argument(
        "entry",
        type=str,
        help="JSON string of the entry to add, e.g. '{\"id\":1,\"name\":\"Alice\"}'"
    )

    subparsers.add_parser("list", help="List all entries")

    parser_get = subparsers.add_parser("get", help="Get entries matching a JSON query")
    parser_get.add_argument(
        "query",
        type=str,
        help="JSON query string to filter entries, e.g. '{\"id\":1}'"
    )

    parser_delete = subparsers.add_parser("delete", help="Delete entries matching a JSON query")
    parser_delete.add_argument(
        "query",
        type=str,
        help="JSON query string to delete, e.g. '{\"id\":1}'"
    )

    subparsers.add_parser("interactive", help="Start interactive REPL mode")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "add":
        print(add_entry(args.entry))
    elif args.command == "list":
        print(list_entries())
    elif args.command == "get":
        print(get_entry(args.query))
    elif args.command == "delete":
        print(delete_entry(args.query))
    else:
        # Interactive REPL
        print("JSON-DB Manager Agent Interactive Mode. Type 'exit' or 'quit' to leave.")
        while True:
            try:
                user_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if user_input.strip().lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            try:
                response = agent_executor.invoke({"input": user_input})
                print(response["output"])
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
