import json
from langchain_core.tools import tool
from jira_api import get_tasks_in_active_sprint

@tool
def get_jira_active_sprint_tasks(project_key: str = None) -> str:
    """
    Fetches all tasks from the active sprint in Jira.
    You can optionally filter by a project_key (e.g., 'BAP').
    This tool should be used when the user asks about current tasks, sprint status, or team workload.
    """
    try:
        tasks = get_tasks_in_active_sprint(project_key=project_key)
        if not tasks:
            return "No tasks found in the active sprint."
        
        # Format tasks for better readability for the LLM
        formatted_tasks = []
        for task in tasks:
            task_details = {
                "id": task.get("id"),
                "name": task.get("name"),
                "status": task.get("status"),
                "priority": task.get("priority"),
                "assignee": task.get("assignee"),
                "reporter": task.get("creator"),
                "due_date": task.get("due_date"),
                "description": (task.get("description") or "")[:100] + "..." if task.get("description") else "No description."
            }
            formatted_tasks.append(task_details)
            
        return json.dumps(formatted_tasks, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"An error occurred while fetching Jira tasks: {e}"

# List of tools to be exported
jira_tools = [get_jira_active_sprint_tasks]
