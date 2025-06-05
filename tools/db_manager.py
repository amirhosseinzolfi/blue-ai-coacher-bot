import sqlite3
from datetime import datetime

class TaskManager:
    def __init__(self, db_name="tasks.db"):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_table()

    def _connect(self):
        """Establish connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def _create_table(self):
        """Create the tasks table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                task_description TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add_task(self, user_id: str, task_description: str) -> int:
        """Add a new task for a user."""
        created_at = datetime.utcnow().isoformat()
        self.cursor.execute("""
            INSERT INTO tasks (user_id, task_description, created_at)
            VALUES (?, ?, ?)
        """, (user_id, task_description, created_at))
        self.conn.commit()
        return self.cursor.lastrowid

    def update_task_status(self, task_id: int, status: str = 'done') -> bool:
        """Update the status of a task (e.g., mark as done)."""
        self.cursor.execute("""
            UPDATE tasks SET status = ? WHERE id = ?
        """, (status, task_id))
        self.conn.commit()
        return self.cursor.rowcount > 0 # Returns True if a row was updated

    def delete_task(self, task_id: int) -> bool:
        """Delete a task by its ID."""
        self.cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0 # Returns True if a row was deleted

    def get_user_tasks(self, user_id: str, status_filter: str = None) -> list:
        """Retrieve tasks for a specific user, optionally filtering by status."""
        query = "SELECT id, task_description, status, created_at FROM tasks WHERE user_id = ?"
        params = [user_id]
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        query += " ORDER BY created_at DESC"
        self.cursor.execute(query, tuple(params))
        tasks = self.cursor.fetchall()
        # Format tasks as list of dictionaries for easier handling
        return [{"id": row[0], "description": row[1], "status": row[2], "created_at": row[3]} for row in tasks]

    def find_task_by_description(self, user_id: str, description_keywords: str) -> list:
        """Find tasks based on keywords in the description for a specific user."""
        # Simple keyword matching, might need refinement (e.g., fuzzy matching)
        query = "SELECT id, task_description, status, created_at FROM tasks WHERE user_id = ? AND task_description LIKE ?"
        params = (user_id, f"%{description_keywords}%")
        self.cursor.execute(query, params)
        tasks = self.cursor.fetchall()
        return [{"id": row[0], "description": row[1], "status": row[2], "created_at": row[3]} for row in tasks]


    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

# Example usage (optional, for testing)
if __name__ == "__main__":
    task_manager = TaskManager()
    print("TaskManager initialized.")
    # Add a task
    # task_id = task_manager.add_task("user123", "Schedule meeting with team")
    # print(f"Added task with ID: {task_id}")
    # # List tasks
    # tasks = task_manager.get_user_tasks("user123")
    # print("User tasks:", tasks)
    # # Mark task as done
    # if tasks:
    #     task_manager.update_task_status(tasks[0]['id'], 'done')
    #     print(f"Updated task {tasks[0]['id']} status to done.")
    #     tasks = task_manager.get_user_tasks("user123")
    #     print("User tasks after update:", tasks)
    # # Delete task
    # if tasks:
    #     task_manager.delete_task(tasks[0]['id'])
    #     print(f"Deleted task {tasks[0]['id']}.")
    #     tasks = task_manager.get_user_tasks("user123")
    #     print("User tasks after delete:", tasks)

    task_manager.close()
    print("TaskManager connection closed.")