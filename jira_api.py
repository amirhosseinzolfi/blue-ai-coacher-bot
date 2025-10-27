import os
import sys
import json
import logging
from datetime import datetime
import argparse

import requests
from jira import JIRA

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)

# ---------- Try colorful tables ----------
USE_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    USE_RICH = True
except Exception:
    console = None
    USE_RICH = False

# ---------- Jira Config ----------
JIRA_SERVER = "https://blufyorg.atlassian.net"
JIRA_EMAIL = "bluefy.org@gmail.com"
JIRA_USER_ID = "63f9be4d40328c12e4edde22"
JIRA_API_TOKEN = "***REMOVED***"

# ---------- Jira Connection ----------
options = {'server': JIRA_SERVER}
try:
    jira = JIRA(options, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    user = jira.current_user()
    logging.info(f"Authenticated as {user}")
except Exception as e:
    print(f"Error connecting to Jira: {e}")
    sys.exit(1)

# ---------- Utilities ----------
def convert_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"Error converting date: {e}")
        return None

def get_all_projects():
    return jira.projects()

def find_task_by_name(task_name):
    jql = f'summary ~ "{task_name}"'
    try:
        issues = jira.search_issues(jql, maxResults=50)
        return issues[0] if issues else None
    except Exception as e:
        print(f"Error finding task: {e}")
        return None

def create_task(task_name, task_description=None, task_priority=2, task_due_date=None, task_labels="", task_status="To Do"):
    project_key = input("Enter the project key: ").strip()
    if not project_key:
        print("Project key is required.")
        return

    priority_mapping = {1: 'Highest', 2: 'High', 3: 'Medium', 4: 'Low', 5: 'Lowest'}
    priority_name = priority_mapping.get(task_priority, 'Medium')

    issue_dict = {
        'project': {'key': project_key},
        'summary': task_name,
        'description': task_description or '',
        'issuetype': {'name': 'Task'},
        'priority': {'name': priority_name},
        'labels': [label.strip() for label in task_labels.split(',') if label.strip()],
        'assignee': {'accountId': JIRA_USER_ID},
    }

    if task_due_date:
        due_date = convert_date(task_due_date)
        if not due_date:
            return
        issue_dict['duedate'] = due_date

    try:
        new_issue = jira.create_issue(fields=issue_dict)
        print(f"Task Created: {new_issue.key}")
        return {'key': new_issue.key, 'url': f"{JIRA_SERVER}/browse/{new_issue.key}"}
    except Exception as e:
        print(f"Error creating task: {e}")
        return None

def delete_task_by_name(task_name):
    issue = find_task_by_name(task_name)
    if not issue:
        print(f"Task '{task_name}' not found.")
        return
    try:
        issue.delete()
        print(f"Task '{task_name}' deleted successfully.")
    except Exception as e:
        print(f"Failed to delete the task '{task_name}'. Error: {e}")

def get_tasks_by_project_name(project_name):
    projects = get_all_projects()
    for project in projects:
        if project.name.lower() == project_name.lower():
            tasks = []
            jql = f'project = "{project.key}" ORDER BY created DESC'
            try:
                issues = jira.search_issues(jql, maxResults=200)
                if issues:
                    for issue in issues:
                        tasks.append(issue_to_dict(issue))
                    display_tasks(tasks)
                    return issues
                else:
                    print(f"No tasks found in project '{project_name}'.")
                    return []
            except Exception as e:
                print(f"Error fetching tasks: {e}")
                return []
    print(f"Project '{project_name}' not found.")
    return []

def update_task_by_name(task_name, new_task_name=None, new_task_description=None, new_task_priority=None, new_task_due_date=None, new_task_labels=None, new_task_status=None):
    issue = find_task_by_name(task_name)
    if not issue:
        print(f"Task '{task_name}' not found.")
        return

    updates = {}

    if new_task_name:
        updates['summary'] = new_task_name
    if new_task_description:
        updates['description'] = new_task_description
    if new_task_priority:
        priority_mapping = {1: 'Highest', 2: 'High', 3: 'Medium', 4: 'Low', 5: 'Lowest'}
        updates['priority'] = {'name': priority_mapping.get(new_task_priority, 'Medium')}
    if new_task_due_date:
        due_date = convert_date(new_task_due_date)
        if not due_date:
            return
        updates['duedate'] = due_date
    if new_task_labels:
        updates['labels'] = [label.strip() for label in new_task_labels.split(',') if label.strip()]

    if updates:
        try:
            issue.update(fields=updates)
            print(f"Task '{task_name}' updated successfully.")
        except Exception as e:
            print(f"Failed to update task '{task_name}'. Error: {e}")
    else:
        print("No updates provided.")

    if new_task_status:
        try:
            transitions = jira.transitions(issue)
            transition_id = next((t['id'] for t in transitions if t['name'].lower() == new_task_status.lower()), None)
            if transition_id:
                jira.transition_issue(issue, transition_id)
                print(f"Task status updated to '{new_task_status}'.")
            else:
                print(f"Status '{new_task_status}' not found in available transitions.")
        except Exception as e:
            print(f"Failed to update task status. Error: {e}")

def task_finder(start_date=None, end_date=None):
    jql_parts = []
    date_field = 'due'
    if start_date:
        jql_parts.append(f'{date_field} >= "{start_date}"')
    if end_date:
        jql_parts.append(f'{date_field} <= "{end_date}"')
    jql_query = (' AND '.join(jql_parts) if jql_parts else '') + f' ORDER BY {date_field} ASC'

    try:
        issues = jira.search_issues(jql_query, maxResults=200)
        return [issue_to_dict(i) for i in issues] if issues else []
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        return []

# ---------- NEW: Active Sprint Fetch ----------
def get_tasks_in_active_sprint(project_key=None):
    """
    Returns all issues across ALL users in active sprint(s) excluding Done tasks.
    Optional: filter by a specific project key (e.g., 'BAP').
    """
    jql = 'sprint in openSprints() AND status NOT IN ("Done", "Closed", "Resolved", "Complete", "Completed")'
    if project_key:
        jql = f'project = "{project_key}" AND {jql}'
    jql += ' ORDER BY assignee, status, priority DESC'

    try:
        issues = jira.search_issues(jql, maxResults=1000)
        tasks = []
        
        for issue in issues:
            # Additional filter to ensure no completed tasks
            status_name = issue.fields.status.name if getattr(issue.fields, 'status', None) else ""
            done_statuses = ["done", "closed", "resolved", "complete", "completed", "finished"]
            
            if status_name.lower() not in done_statuses:
                tasks.append(issue_to_dict(issue))
        
        return tasks
    except Exception as e:
        print(f"Error fetching active sprint tasks: {e}")
        return []

# ---------- Display Helpers ----------
def issue_to_dict(issue):
    fields = issue.fields
    return {
        'id': issue.key,
        'name': fields.summary,
        'description': fields.description or "",
        'priority': (fields.priority.name if getattr(fields, 'priority', None) else "No Priority"),
        'due_date': fields.duedate or "",
        'labels': fields.labels or [],
        'status': fields.status.name if getattr(fields, 'status', None) else "",
        'creator': fields.creator.displayName if getattr(fields, 'creator', None) else "",
        'assignee': (fields.assignee.displayName if getattr(fields, 'assignee', None) else "Unassigned"),
        'url': f"{JIRA_SERVER}/browse/{issue.key}"
    }

def display_tasks(tasks):
    if not tasks:
        print("No tasks found.")
        return

    if USE_RICH:
        table = Table(title="Jira Issues", box=box.SIMPLE_HEAVY)
        table.add_column("Key", style="bold cyan")
        table.add_column("Summary", style="white")
        table.add_column("Status", style="bold magenta")
        table.add_column("Priority", style="yellow")
        table.add_column("Assignee", style="green")
        table.add_column("Due", style="bright_blue")
        table.add_column("Labels", style="dim")
        table.add_column("URL", style="blue underline")

        for t in tasks:
            table.add_row(
                t['id'],
                (t['name'] or "")[:80],
                t['status'] or "",
                t['priority'] or "",
                t['assignee'] or "",
                t['due_date'] or "",
                ", ".join(t['labels']) if t['labels'] else "",
                t['url'],
            )
        console.print(table)
    else:
        # Plain fallback
        for t in tasks:
            print(f"Task: {t['name']}")
            print(f"  ID: {t['id']}")
            print(f"  Status: {t['status']}")
            print(f"  Priority: {t['priority']}")
            print(f"  Assignee: {t['assignee']}")
            print(f"  Due Date: {t['due_date']}")
            print(f"  Labels: {', '.join(t['labels']) if t['labels'] else 'None'}")
            print(f"  URL: {t['url']}")
            print("-" * 60)

# ---------- Simple UI Flows ----------
def task_finder_ui():
    print("Enter the date range to find tasks (leave empty to retrieve all tasks):")
    start_date = input("Start Date (yyyy-mm-dd, Optional): ").strip()
    end_date = input("End Date (yyyy-mm-dd, Optional): ").strip()
    tasks = task_finder(start_date=start_date or None, end_date=end_date or None)
    display_tasks(tasks)

def create_task_ui():
    task_name = input("Enter Task Name: ").strip()
    task_description = input("Enter Task Description (Optional): ").strip()
    task_priority_input = input("Enter Task Priority (1: Highest, 2: High, 3: Medium, 4: Low, 5: Lowest) [Default: 3]: ").strip()
    task_priority = int(task_priority_input) if task_priority_input else 3
    task_due_date = input("Enter Task Due Date (yyyy-mm-dd) (Optional): ").strip()
    task_labels = input("Enter Task Labels (comma-separated, Optional): ").strip()
    task_status = input("Enter Task Status [Default: To Do]: ").strip() or "To Do"
    response = create_task(task_name, task_description, task_priority, task_due_date or None, task_labels, task_status)
    print("Task Created:" if response else "Failed to create task.", response or "")

def delete_task_ui():
    task_name = input("Enter Task Name to delete: ").strip()
    delete_task_by_name(task_name=task_name)

def update_task_ui():
    task_name = input("Enter Task Name to update: ").strip()
    new_task_name = input("New Task Name (Optional): ").strip()
    new_task_description = input("New Task Description (Optional): ").strip()
    new_task_status = input("New Task Status (Optional): ").strip()
    new_task_priority_input = input("New Task Priority (1..5) (Optional): ").strip()
    new_task_priority = int(new_task_priority_input) if new_task_priority_input else None
    new_task_due_date = input("New Task Due Date (yyyy-mm-dd) (Optional): ").strip()
    new_task_labels = input("New Task Labels (comma-separated, Optional): ").strip()

    update_task_by_name(
        task_name=task_name,
        new_task_name=new_task_name or None,
        new_task_description=new_task_description or None,
        new_task_priority=new_task_priority or None,
        new_task_due_date=new_task_due_date or None,
        new_task_labels=new_task_labels or None,
        new_task_status=new_task_status or None
    )

def find_tasks_by_project_name_ui():
    project_name = input("Enter the project name to retrieve tasks from: ").strip()
    get_tasks_by_project_name(project_name)

def show_active_sprint_ui():
    proj = input("Project key to filter (Optional, e.g., BAP): ").strip()
    proj = proj if proj else None
    tasks = get_tasks_in_active_sprint(project_key=proj)
    display_tasks(tasks)

def main_menu():
    print("\nJira Task Manager")
    print("=================")
    print("1. Create a new task")
    print("2. Delete a task")
    print("3. Update a task")
    print("4. Find tasks by due date range")
    print("5. Find tasks by project name")
    print("6. Show tasks in active sprint (all users, optional project filter)")
    print("7. Exit")
    return input("Choose an option: ").strip()

# ---------- CLI Entry ----------
def main():
    parser = argparse.ArgumentParser(description="Jira Task Manager")
    parser.add_argument("--active-sprint", action="store_true", help="Show all issues for all users in active sprint(s)")
    parser.add_argument("--project", type=str, help="Optional project key filter (e.g., BAP)")
    args, unknown = parser.parse_known_args()

    # Command-line fast path
    if args.active_sprint:
        tasks = get_tasks_in_active_sprint(project_key=args.project)
        display_tasks(tasks)
        return

    # Fallback to interactive menu
    while True:
        choice = main_menu()
        if choice == '1':
            create_task_ui()
        elif choice == '2':
            delete_task_ui()
        elif choice == '3':
            update_task_ui()
        elif choice == '4':
            task_finder_ui()
        elif choice == '5':
            find_tasks_by_project_name_ui()
        elif choice == '6':
            show_active_sprint_ui()
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
