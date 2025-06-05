import sqlite3
import datetime
import threading
import logging
from typing import Dict, List, Optional, Any
from config import DATABASE_NAME  # DATABASE_NAME now becomes the sqlite file path (e.g. "blue_business.db")

class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    @property
    def message_type(self):
        return {
            "user": "HumanMessage",
            "assistant": "AIMessage",
            "system": "SystemMessage"
        }.get(self.role, "HumanMessage")

class DatabaseManager:
    def __init__(self, db_path: str):
        # Open sqlite connection with thread safety
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        logging.info("Connected to SQLite at %s", db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        with self.conn:
            # Table for business info (includes user_report and session_id)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS business_info (
                    chat_id TEXT NOT NULL, 
                    chat_type TEXT DEFAULT 'private' NOT NULL,
                    business_info TEXT,
                    user_report TEXT,
                    session_id TEXT,
                    updated_at DATETIME,
                    PRIMARY KEY (chat_id, chat_type) 
                )
            """)
            
            # Table for chat history with composite primary key
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    UNIQUE(chat_id, session_id, timestamp)
                )
            """)
            
            # Table for settings with composite primary key
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    chat_id TEXT NOT NULL,
                    chat_type TEXT DEFAULT 'private' NOT NULL,
                    setting_type TEXT NOT NULL,
                    value TEXT,
                    updated_at DATETIME,
                    PRIMARY KEY(chat_id, chat_type, setting_type)
                )
            """)
            
            # Table for user tasks
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    entry TEXT NOT NULL,
                    task_date TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    UNIQUE(chat_id, user_name, entry, task_date)
                )
            """)

    def save_business_info(self, chat_id: str, info: str, chat_type: str = "private") -> None:
        with self._lock, self.conn:
            self.conn.execute("""
                INSERT INTO business_info (chat_id, chat_type, business_info, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, chat_type) DO UPDATE SET
                    business_info=excluded.business_info,
                    updated_at=excluded.updated_at
            """, (chat_id, chat_type, info, datetime.datetime.utcnow()))
    
    def get_business_info(self, chat_id: str, chat_type: str = "private") -> str:
        cur = self.conn.execute("""
            SELECT business_info FROM business_info
            WHERE chat_id=? AND chat_type=?
        """, (chat_id, chat_type))
        row = cur.fetchone()
        return row["business_info"] if row and row["business_info"] else ""
    
    def save_chat_message(self, chat_id: str, role: str, content: str, session_id: str) -> None:
        with self._lock, self.conn:
            timestamp = datetime.datetime.utcnow()
            try:
                self.conn.execute("""
                    INSERT INTO chat_history (chat_id, role, content, session_id, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (chat_id, role, content, session_id, timestamp))
            except sqlite3.IntegrityError:
                logging.warning(f"IntegrityError on save_chat_message for chat {chat_id}, session {session_id}, ts {timestamp}. Content might be duplicate or timestamp collision.")
            except Exception as e:
                logging.error(f"Error saving message: {e}", exc_info=True)

    def get_chat_history(self, chat_id: str, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get chat history as a list of ChatMessage objects for a specific session."""
        query = """
            SELECT role, content 
            FROM chat_history 
            WHERE chat_id=? AND session_id=?
            ORDER BY timestamp ASC LIMIT ?
        """
        params = [str(chat_id), session_id, limit]  # Ensure chat_id is string here
        
        cur = self.conn.execute(query, tuple(params))
        messages = []
        for row in cur.fetchall():
            messages.append(ChatMessage(row['role'], row['content']))
        return messages

    def get_messages_for_session(self, chat_id: str, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get all messages for a specific session."""
        return self.get_chat_history(chat_id, session_id=session_id, limit=limit)

    def save_setting(self, chat_id: str, setting_type: str, value: Any, chat_type: str = "private") -> None:
        # Ensure chat_id is string, as it's part of the PK
        s_chat_id = str(chat_id)
        s_value = str(value)
        now = datetime.datetime.utcnow()

        with self._lock, self.conn:
            # Attempt an UPDATE first
            cursor = self.conn.execute("""
                UPDATE settings 
                SET value = ?, updated_at = ?
                WHERE chat_id = ? AND chat_type = ? AND setting_type = ?
            """, (s_value, now, s_chat_id, chat_type, setting_type))
            
            # If no rows were updated, then INSERT
            if cursor.rowcount == 0:
                self.conn.execute("""
                    INSERT INTO settings (chat_id, chat_type, setting_type, value, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (s_chat_id, chat_type, setting_type, s_value, now))
    
    def get_setting(self, chat_id: str, setting_type: str, chat_type: str = "private") -> Optional[Any]:
        cur = self.conn.execute("""
            SELECT value FROM settings WHERE chat_id=? AND chat_type=? AND setting_type=?
        """, (chat_id, chat_type, setting_type))
        row = cur.fetchone()
        return row["value"] if row else None
    
    def save_conversation_summary(self, chat_id: str, summary: str, chat_type: str = "private") -> None:
        self.save_setting(chat_id, "conversation_summary", summary, chat_type)
    
    def get_conversation_summary(self, chat_id: str, chat_type: str = "private") -> str:
        val = self.get_setting(chat_id, "conversation_summary", chat_type)
        return val if val else ""
    
    def save_user_report(self, chat_id: str, report: str, chat_type: str = "private") -> None:
        with self._lock, self.conn:
            self.conn.execute("""
                INSERT INTO business_info (chat_id, chat_type, user_report, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, chat_type) DO UPDATE SET
                    user_report=excluded.user_report,
                    updated_at=excluded.updated_at
            """, (chat_id, chat_type, report, datetime.datetime.utcnow()))
    
    def save_session_id(self, chat_id: str, session_id: str, chat_type: str = "private") -> None:
        with self._lock, self.conn:
            self.conn.execute("""
                INSERT INTO business_info (chat_id, chat_type, session_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, chat_type) DO UPDATE SET
                    session_id=excluded.session_id,
                    updated_at=excluded.updated_at
            """, (chat_id, chat_type, session_id, datetime.datetime.utcnow()))
    
    def get_session_id(self, chat_id: str, chat_type: str = "private") -> Optional[str]:
        try:
            cur = self.conn.execute(
                "SELECT session_id FROM business_info WHERE chat_id=? AND chat_type=?",
                (chat_id, chat_type)
            )
            row = cur.fetchone()
            return row["session_id"] if row and row["session_id"] else None
        except sqlite3.InterfaceError as e:
            logging.error(f"InterfaceError in get_session_id: {e} (chat_id={chat_id}, chat_type={chat_type})")
            return None

    def clear_chat_history(self, chat_id: str, session_id: Optional[str] = None) -> None:
        with self._lock, self.conn:
            if session_id:
                logging.info(f"Clearing chat history for chat_id={chat_id}, session_id={session_id}")
                self.conn.execute("DELETE FROM chat_history WHERE chat_id=? AND session_id=?", (chat_id, session_id))
            else:
                logging.warning(f"Clearing ALL chat history for chat_id={chat_id} (no session_id specified)")
                self.conn.execute("DELETE FROM chat_history WHERE chat_id=?", (chat_id,))
    
    def save_task(self, chat_id: str, user_name: str, entry: str, task_date: str = None, timestamp: str = None) -> bool:
        """Save a user task to the database"""
        if task_date is None:
            task_date = datetime.date.today().isoformat()
        
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        try:
            with self._lock, self.conn:
                self.conn.execute("""
                    INSERT OR IGNORE INTO user_tasks (chat_id, user_name, entry, task_date, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (chat_id, user_name, entry, task_date, timestamp))
                return True
        except Exception as e:
            logging.error(f"Error saving task to database: {e}", exc_info=True)
            return False

    def get_tasks_for_date(self, chat_id: str, task_date: str = None) -> List[Dict[str, Any]]:
        """Get all tasks for a specific date"""
        if task_date is None:
            task_date = datetime.date.today().isoformat()
        
        try:
            cur = self.conn.execute("""
                SELECT user_name, entry, timestamp 
                FROM user_tasks
                WHERE chat_id = ? AND task_date = ?
                ORDER BY timestamp ASC
            """, (chat_id, task_date))
            
            tasks = []
            for row in cur.fetchall():
                tasks.append({
                    "user": row["user_name"],
                    "entry": row["entry"],
                    "timestamp": row["timestamp"]
                })
            return tasks
        except Exception as e:
            logging.error(f"Error retrieving tasks from database: {e}", exc_info=True)
            return []

    def get_all_tasks_by_chat(self, chat_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all tasks organized by date for a chat"""
        try:
            cur = self.conn.execute("""
                SELECT task_date, user_name, entry, timestamp 
                FROM user_tasks
                WHERE chat_id = ?
                ORDER BY task_date DESC, timestamp ASC
            """, (chat_id,))
            
            tasks_by_date = {}
            for row in cur.fetchall():
                date = row["task_date"]
                if date not in tasks_by_date:
                    tasks_by_date[date] = []
                
                tasks_by_date[date].append({
                    "user": row["user_name"],
                    "entry": row["entry"],
                    "timestamp": row["timestamp"]
                })
            
            return tasks_by_date
        except Exception as e:
            logging.error(f"Error retrieving all tasks from database: {e}", exc_info=True)
            return {}

    def close(self):
        try:
            self.conn.close()
            logging.info("SQLite connection closed successfully")
        except Exception as e:
            logging.error("Error closing SQLite connection: %s", str(e))
            
# Use a thread-local storage for the database manager instance
_thread_local = threading.local()

def get_db_manager():
    if not hasattr(_thread_local, 'db_manager'):
        _thread_local.db_manager = DatabaseManager(DATABASE_NAME)
    return _thread_local.db_manager

_db_manager = get_db_manager()
db_manager = _db_manager

# Exported helper functions for external modules.
def save_business_info(chat_id: str, info: str, chat_type: str = "private") -> None:
    _db_manager.save_business_info(chat_id, info, chat_type)

def get_business_info(chat_id: str, chat_type: str = "private") -> str:
    return _db_manager.get_business_info(chat_id, chat_type)

def save_user_info(chat_id: str, info: str, date: Optional[str] = None, chat_type: str = "private") -> None:
    today = date or datetime.datetime.now().strftime("%Y-%m-%d")
    import json
    _db_manager.save_setting(chat_id, "user_info", json.dumps({"info": info, "date": today}), chat_type)

def get_user_info(chat_id: str, date: Optional[str] = None, chat_type: str = "private") -> str:
    setting_val = _db_manager.get_setting(chat_id, "user_info", chat_type)
    if setting_val:
        import json
        try:
            data = json.loads(setting_val)
            return data.get("info", "")
        except json.JSONDecodeError:
            return ""
    return ""

# Module‐level wrapper to ensure a session exists and save messages
def save_message_to_history(chat_id: str, role: str, content: str, session_id: Optional[str] = None) -> None:
    try:
        effective_chat_id = str(chat_id)
        logging.debug(f"save_message_to_history called: chat_id={chat_id}, role={role}, "
                      f"content_type={type(content)}, incoming_session_id={session_id}")

        # determine or create session
        if not session_id:
            session_id = db_manager.get_session_id(effective_chat_id)
            if not session_id:
                session_id = f"{effective_chat_id}_default_{int(datetime.datetime.now().timestamp())}"
                db_manager.save_session_id(effective_chat_id, session_id)

        db_manager.save_chat_message(effective_chat_id, role, content, session_id)

    except sqlite3.InterfaceError as e:
        logging.error(f"InterfaceError in save_message_to_history: {e} "
                      f"(chat_id={chat_id}, role={role}, session_id={session_id})")
        # attempt to reopen the connection
        try:
            new_conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
            new_conn.row_factory = sqlite3.Row
            db_manager.conn = new_conn
            logging.info("Reopened SQLite connection after InterfaceError")
        except Exception as ex:
            logging.error(f"Failed to reopen SQLite connection: {ex}")
    except Exception as e:
        logging.error(f"Unexpected error in save_message_to_history: {e}", exc_info=True)

def get_chat_history(chat_id: str, session_id: str, limit: int = 50) -> List[ChatMessage]:
    return _db_manager.get_chat_history(str(chat_id), session_id, limit=limit)  # Ensure chat_id is string

def get_chat_messages(chat_id: str, session_id: Optional[str] = None, limit: int = 50) -> List[ChatMessage]:
    effective_chat_id = str(chat_id)  # Ensure chat_id is string
    if not session_id:
        session_id = db_manager.get_session_id(effective_chat_id)
    if session_id:
        return db_manager.get_messages_for_session(effective_chat_id, session_id, limit=limit)
    else:
        logging.warning(f"No session_id available for chat {chat_id} in get_chat_messages. Returning empty list.")
        return []

def save_ai_tone(chat_id: str, tone: str, chat_type: str = "private") -> None:
    _db_manager.save_setting(chat_id, "ai_tone", tone, chat_type)

def get_ai_tone(chat_id: str, chat_type: str = "private") -> str:
    val = _db_manager.get_setting(chat_id, "ai_tone", chat_type)
    return val if val else "دوستانه"

def start_new_session(chat_id: str, chat_type: str = "private") -> str:
    new_session_id = f"{chat_id}_{int(datetime.datetime.now().timestamp())}"
    with _db_manager._lock, _db_manager.conn:
        cur = _db_manager.conn.execute(
            "SELECT business_info, user_report FROM business_info WHERE chat_id=? AND chat_type=?",
            (chat_id, chat_type)
        )
        row = cur.fetchone()
        existing_business_info = row["business_info"] if row and row["business_info"] else ""
        existing_user_report = row["user_report"] if row and row["user_report"] else ""

        _db_manager.conn.execute(
            """
            INSERT OR REPLACE INTO business_info (chat_id, chat_type, session_id, business_info, user_report, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chat_id, chat_type, new_session_id, existing_business_info, existing_user_report, datetime.datetime.utcnow())
        )
        _db_manager.conn.execute(
            "DELETE FROM settings WHERE chat_id=? AND chat_type=? AND setting_type='conversation_summary'",
            (chat_id, chat_type)
        )
    return new_session_id

# Add these to exported helper functions
def save_user_task(chat_id: str, user_name: str, entry: str, task_date: str = None, timestamp: str = None) -> bool:
    """Helper function to save a user task"""
    return db_manager.save_task(chat_id, user_name, entry, task_date, timestamp)

def get_user_tasks_for_date(chat_id: str, task_date: str = None) -> List[Dict[str, Any]]:
    """Helper function to get tasks for a date"""
    return db_manager.get_tasks_for_date(chat_id, task_date)

def get_all_user_tasks(chat_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Helper function to get all tasks for a chat"""
    return db_manager.get_all_tasks_by_chat(chat_id)

def export_tasks_to_json_format(chat_id: str) -> Dict[str, Any]:
    """Convert database tasks to the JSON file format"""
    db_tasks = db_manager.get_all_tasks_by_chat(chat_id)
    json_format = {}
    
    for date, tasks in db_tasks.items():
        if date not in json_format:
            json_format[date] = []
        
        for task in tasks:
            json_format[date].append({
                "entry": task["entry"],
                "timestamp": task["timestamp"],
                "user": task["user"]
            })
    
    return json_format

__all__ = [
    'save_business_info',
    'get_business_info',
    'save_user_info',
    'get_user_info',
    'save_message_to_history',
    'get_chat_history',
    'get_chat_messages',
    'save_ai_tone',
    'get_ai_tone',
    'db_manager',
    'save_user_report',
    'save_session_id',
    'get_session_id',
    'start_new_session',
    'ChatMessage',
    'save_user_task',
    'get_user_tasks_for_date',
    'get_all_user_tasks',
    'export_tasks_to_json_format'
]
