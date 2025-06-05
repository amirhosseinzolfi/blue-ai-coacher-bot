# --- Define the Langgraph graph ---
builder = StateGraph(AgentState)

# Add the nodes
builder.add_node("load_tasks", load_tasks)
builder.add_node("route", route)
builder.add_node("add_task", add_task)
builder.add_node("check_task", check_task)
builder.add_node("refine_task", refine_task)
builder.add_node("list_tasks", list_tasks)
builder.add_node("respond", respond)
builder.add_node("summarize", summarize_conversation)
builder.add_node("save_tasks", save_tasks)
# Add the missing 'handle_action' node
builder.add_node("handle_action", lambda state: state) # This node doesn't perform any action

# Define the edges
builder.set_entry_point("load_tasks")

builder.add_edge("load_tasks", "route")

builder.add_conditional_edges(
    "route",
    should_summarize,
    {
        True: "summarize",
        False: "handle_action"
    }
)

builder.add_conditional_edges(
    "handle_action",
    lambda x: x.get("action"),
    {
        "add_task": "add_task",
        "check_task": "check_task",
        "refine_task": "refine_task",
        "list_tasks": "list_tasks",
        "respond": "respond",
    },
)

builder.add_edge("add_task", "save_tasks")
builder.add_edge("check_task", "save_tasks")
builder.add_edge("refine_task", "save_tasks")
builder.add_edge("list_tasks", "save_tasks")
builder.add_edge("respond", END)
builder.add_edge("summarize", "route") # After summarizing, go back to routing
builder.add_edge("save_tasks", "route")

# Compile the graph
task_management_agent = builder.compile()