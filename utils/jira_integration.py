"""
Jira Integration Utility
-----------------------
Handles fetching active sprint tasks from Jira API for LLM context.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import JIRA_ENABLED, JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY

logger = logging.getLogger(__name__)

# Initialize Jira connection if enabled
jira_client = None
if JIRA_ENABLED:
    try:
        from jira import JIRA
        options = {'server': JIRA_SERVER}
        jira_client = JIRA(options, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
        logger.info("Jira integration initialized successfully")
    except ImportError:
        logger.warning("Jira library not installed. Install with: pip install jira")
        JIRA_ENABLED = False
    except Exception as e:
        logger.error(f"Failed to initialize Jira connection: {e}")
        jira_client = None

def get_active_sprint_tasks(project_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all active sprint tasks from Jira using the refined function from jira_api.py
    
    Args:
        project_key: Optional project key to filter tasks (e.g., 'BAP')
        
    Returns:
        List of task dictionaries with relevant information
    """
    if not JIRA_ENABLED or not jira_client:
        return []
    
    try:
        # Use the refined active sprint function that excludes completed tasks
        jql = 'sprint in openSprints() AND status NOT IN ("Done", "Closed", "Resolved", "Complete", "Completed")'
        if project_key:
            jql = f'project = "{project_key}" AND {jql}'
        jql += ' ORDER BY assignee, status, priority DESC'
        
        # Fetch issues
        issues = jira_client.search_issues(jql, maxResults=1000)
        
        tasks = []
        for issue in issues:
            fields = issue.fields
            
            # Additional filter to ensure no completed tasks
            status_name = fields.status.name if getattr(fields, 'status', None) else ""
            done_statuses = ["done", "closed", "resolved", "complete", "completed", "finished"]
            
            if status_name.lower() not in done_statuses:
                task_info = {
                    'key': issue.key,
                    'name': fields.summary,
                    'description': fields.description or "",
                    'priority': (fields.priority.name if getattr(fields, 'priority', None) else "No Priority"),
                    'due_date': fields.duedate or "",
                    'status': status_name,
                    'assignee': (fields.assignee.displayName if getattr(fields, 'assignee', None) else "Unassigned"),
                    'reporter': (fields.reporter.displayName if getattr(fields, 'reporter', None) else "Unknown"),
                    'url': f"{JIRA_SERVER}/browse/{issue.key}"
                }
                tasks.append(task_info)
        
        logger.info(f"Fetched {len(tasks)} active sprint tasks from Jira (excluding completed)")
        return tasks
        
    except Exception as e:
        logger.error(f"Error fetching active sprint tasks from Jira: {e}")
        return []

def clean_description(description: str, max_length: int = 100) -> str:
    """
    Clean and truncate task description for better readability.
    
    Args:
        description: Raw description text
        max_length: Maximum character length
        
    Returns:
        Cleaned and truncated description
    """
    if not description or not description.strip():
        return "No description"
    
    # Remove common Jira formatting artifacts
    cleaned = description.strip()
    
    # Remove markdown/confluence markup
    cleaned = cleaned.replace('h1. ', '').replace('h2. ', '').replace('h3. ', '')
    cleaned = cleaned.replace('*', '').replace('_', '').replace('~', '')
    
    # Remove specific Jira artifacts
    artifacts_to_remove = [
        '{adf:display=block}',
        '{"type":"taskList"',
        '{"type":"taskItem"',
        'cont...'
    ]
    
    for artifact in artifacts_to_remove:
        cleaned = cleaned.replace(artifact, '')
    
    # Clean up multiple spaces and newlines
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n+', ' ', cleaned)
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length-3] + "..."
    
    return cleaned.strip() or "No description"

def format_jira_context(tasks: List[Dict[str, Any]]) -> str:
    """
    Format Jira active sprint tasks into a structured, concise context string for LLM.
    
    Args:
        tasks: List of task dictionaries from active sprint
        
    Returns:
        Formatted string containing active sprint task information
    """
    if not tasks:
        return ""
    
    context_lines = ["\n## ACTIVE SPRINT TASKS"]
    context_lines.append(f"Current sprint: {len(tasks)} active tasks")
    
    # Group tasks by assignee for better organization
    tasks_by_assignee = {}
    for task in tasks:
        assignee = task['assignee']
        if assignee not in tasks_by_assignee:
            tasks_by_assignee[assignee] = []
        tasks_by_assignee[assignee].append(task)
    
    for assignee, user_tasks in tasks_by_assignee.items():
        context_lines.append(f"\n### {assignee} ({len(user_tasks)} tasks)")
        
        for i, task in enumerate(user_tasks, 1):
            # Create structured task entry
            task_entry = f"{i}. **{task['name']}**"
            
            # Add essential details in compact format
            details = []
            if task['priority'] != "No Priority":
                details.append(f"Priority: {task['priority']}")
            if task['due_date']:
                details.append(f"Due: {task['due_date']}")
            if task['status']:
                details.append(f"Status: {task['status']}")
            if task['reporter'] != "Unknown":
                details.append(f"Reporter: {task['reporter']}")
            
            if details:
                task_entry += f" ({', '.join(details)})"
            
            # Add cleaned description
            cleaned_desc = clean_description(task['description'], max_length=80)
            if cleaned_desc != "No description":
                task_entry += f"\n   Description: {cleaned_desc}"
            
            context_lines.append(task_entry)
    
    return "\n".join(context_lines)

def get_jira_context_for_chat(chat_id: str) -> str:
    """
    Get Jira active sprint context specifically formatted for a chat session.
    
    Args:
        chat_id: The chat ID to potentially filter tasks by
        
    Returns:
        Formatted Jira active sprint context string
    """
    if not JIRA_ENABLED:
        return ""
    
    try:
        # Get active sprint tasks with project filter
        project_key = JIRA_PROJECT_KEY if JIRA_PROJECT_KEY else None
        tasks = get_active_sprint_tasks(project_key=project_key)
        
        if tasks:
            logger.info(f"Retrieved {len(tasks)} active sprint tasks for chat context")
            return format_jira_context(tasks)
        else:
            logger.info("No active sprint tasks found")
            return ""
            
    except Exception as e:
        logger.error(f"Error getting Jira active sprint context for chat {chat_id}: {e}")
        return ""
