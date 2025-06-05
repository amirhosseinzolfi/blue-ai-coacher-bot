import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize LLM
llm = ChatOpenAI(
    base_url="http://185.110.190.167:15203/v1",
    model_name="gemini-2.0-flash",
    temperature=0.5,
    api_key="324"
)

# Define State Structure
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    tasks: List[Dict[str, Any]]
    current_input: str
    actions: List[Dict[str, Any]]
    response: str

# Load existing tasks
def load_tasks():
    try:
        with open("tasks.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save tasks to JSON
def save_tasks(tasks):
    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)

# Define Processing Nodes
def process_input(state: AgentState) -> AgentState:
    # Prepare prompt with conversation history and current tasks
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task management assistant. Analyze the user input and conversation history to manage tasks.

Current Tasks:
{formatted_tasks}

Conversation History:
{formatted_history}

User Input: {input}

Output JSON with:
- "actions": List of operations (add/complete/refine)
- "response": Natural language response"""),
        ("human", "{input}")
    ])
    
    formatted_tasks = "\n".join(
        f"{t['id']}. {t['description']} ({t['status']})" 
        for t in state["tasks"]
    )
    
    formatted_history = "\n".join(
        f"{msg['role']}: {msg['content']}" 
        for msg in state["messages"]
    )

    # Get structured response from LLM
    chain = prompt | llm
    result = chain.invoke({
        "input": state["current_input"],
        "formatted_tasks": formatted_tasks,
        "formatted_history": formatted_history
    })
    
    # Parse LLM output
    try:
        parsed = json.loads(result.content)
        return {**state, 
                "actions": parsed.get("actions", []),
                "response": parsed.get("response", "")}
    except json.JSONDecodeError:
        return {**state, "actions": [], "response": "Could not process request"}

def update_tasks(state: AgentState) -> AgentState:
    tasks = state["tasks"].copy()
    
    for action in state["actions"]:
        action_type = action.get("type")
        
        if action_type == "add":
            new_id = max([t.get("id", 0) for t in tasks], default=0) + 1
            tasks.append({
                "id": new_id,
                "description": action.get("description"),
                "status": "todo"
            })
            
        elif action_type == "complete":
            task_id = action.get("id")
            for t in tasks:
                if t["id"] == task_id:
                    t["status"] = "done"
                    
        elif action_type == "refine":
            task_id = action.get("id")
            new_desc = action.get("new_description")
            for t in tasks:
                if t["id"] == task_id:
                    t["description"] = new_desc
    
    return {**state, "tasks": tasks}

def generate_response(state: AgentState) -> AgentState:
    state["messages"].append({
        "role": "assistant",
        "content": state["response"]
    })
    return state

# Build LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("process_input", process_input)
workflow.add_node("update_tasks", update_tasks)
workflow.add_node("save_tasks", lambda state: {**state, **save_tasks(state["tasks"])})
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("process_input")
workflow.add_edge("process_input", "update_tasks")
workflow.add_edge("update_tasks", "save_tasks")
workflow.add_edge("save_tasks", "generate_response")
workflow.add_edge("generate_response", END)

agent = workflow.compile()

# Terminal Interface
def main():
    tasks = load_tasks()
    messages = []
    
    print("Task Manager Agent - Type 'exit' to quit")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break
                
            messages.append({"role": "user", "content": user_input})
            
            result = agent.invoke({
                "messages": messages[:-1],
                "tasks": tasks,
                "current_input": user_input,
                "actions": [],
                "response": ""
            })
            
            tasks = result["tasks"]
            messages = result["messages"]
            
            print(f"Assistant: {result['response']}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()