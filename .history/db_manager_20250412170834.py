import datetime
from typing import Dict, List, Optional, Union
from pymongo import MongoClient, ASCENDING, IndexModel
from config import MONGO_CONNECTION_STRING, DATABASE_NAME

class DatabaseManager:
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Initialize collections
        self.business_info = self.db["business_info"]
        self.chat_history = self.db["chat_history"]
        self.user_info = self.db["user_info"]
        self.settings = self.db["settings"]
        self.chat_sessions = self.db["chat_sessions"]
        self.chat_metadata = self.db["chat_metadata"]
        
        # Additional collections for permanent storage
        self.chat_state = self.db["chat_state"]  # For storing permanent chat state
        self.ai_tones = self.db["ai_tones"]       # For storing AI tones
        self.chat_summaries = self.db["chat_summaries"]  # For storing conversation summaries
        
        # Create necessary indexes
        self._setup_indexes()
        # Create additional indexes
        self._setup_additional_indexes()
    
    def _setup_indexes(self):
        """Setup necessary indexes with chat_type support for improved query performance."""
        self.business_info.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True),
            IndexModel([("updated_at", ASCENDING)])
        ])
        self.chat_history.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", ASCENDING)])
        ])
        self.chat_sessions.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)])
        ])
        self.chat_metadata.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True)
        ])

    def _setup_additional_indexes(self):
        """Setup additional indexes for permanent data storage."""
        self.chat_state.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True),
            IndexModel([("updated_at", ASCENDING)])
        ])
        self.ai_tones.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True)
        ])
        self.chat_summaries.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)])
        ])

    def save_chat_metadata(self, chat_id: str, chat_type: str, metadata: Dict) -> None:
        self.chat_metadata.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {**metadata, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_chat_metadata(self, chat_id: str, chat_type: str) -> Dict:
        doc = self.chat_metadata.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return doc if doc else {}

    def save_business_info(self, chat_id: str, info: str, chat_type: str = "private") -> None:
        self.business_info.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {"business_info": info, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_business_info(self, chat_id: str, chat_type: str = "private") -> str:
        doc = self.business_info.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return doc.get("business_info", "") if doc else ""

    def save_chat_message(self, chat_id: str, role: str, content: str, 
                          session_id: str, chat_type: str = "private") -> None:
        message = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "session_id": session_id,
            "chat_type": chat_type,
            "timestamp": datetime.datetime.utcnow()
        }
        self.chat_history.insert_one(message)

    def get_chat_history(self, chat_id: str, chat_type: str = "private", 
                         session_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        query = {"chat_id": chat_id, "chat_type": chat_type}
        if session_id:
            query["session_id"] = session_id
        return list(self.chat_history.find(query).sort("timestamp", -1).limit(limit))

    def save_setting(self, chat_id: str, setting_type: str, value: any,
                     chat_type: str = "private") -> None:
        self.settings.update_one(
            {"chat_id": chat_id, "setting_type": setting_type},
            {"$set": {"value": value, "chat_type": chat_type, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_setting(self, chat_id: str, setting_type: str) -> Optional[any]:
        doc = self.settings.find_one({"chat_id": chat_id, "setting_type": setting_type})
        return doc.get("value") if doc else None

    def load_chat_context(self, chat_id: str, chat_type: str = "private") -> Dict:
        return {
            "business_info": self.get_business_info(chat_id, chat_type),
            "chat_history": self.get_chat_history(chat_id, chat_type),
            "settings": {
                "ai_tone": self.get_setting(chat_id, "ai_tone"),
                "language": self.get_setting(chat_id, "language"),
            }
        }

    def clear_chat_history(self, chat_id: str) -> None:
        self.chat_history.delete_many({"chat_id": chat_id})

    def save_conversation_summary(self, chat_id: str, session_id: str, summary: str) -> None:
        # Save the summary in the chat_summaries collection for a given session.
        self.chat_summaries.update_one(
            {"chat_id": chat_id, "chat_type": "private", "session_id": session_id},
            {"$set": {"value": summary, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_conversation_summary(self, chat_id: str, session_id: str) -> str:
        doc = self.chat_summaries.find_one({"chat_id": chat_id, "chat_type": "private", "session_id": session_id})
        return doc.get("value", "") if doc else ""

    def save_chat_state(self, chat_id: str, chat_type: str, state_data: Dict) -> None:
        self.chat_state.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {**state_data, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_chat_state(self, chat_id: str, chat_type: str) -> Dict:
        state = self.chat_state.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return state if state else {}

    def save_ai_tone(self, chat_id: str, tone: str, chat_type: str = "private") -> None:
        self.ai_tones.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {"tone": tone, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_ai_tone(self, chat_id: str, chat_type: str = "private") -> str:
        doc = self.ai_tones.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return doc.get("tone", "دوستانه") if doc else "دوستانه"

    def save_conversation_context(self, chat_id: str, context_data: Dict, chat_type: str = "private") -> None:
        self.chat_state.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {"context": context_data, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def load_conversation_context(self, chat_id: str, chat_type: str = "private") -> Dict:
        state = self.chat_state.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return state.get("context", {}) if state else {}

    def load_all_chat_data(self, chat_id: str, chat_type: str = "private") -> Dict:
        return {
            "business_info": self.get_business_info(chat_id, chat_type),
            "ai_tone": self.get_ai_tone(chat_id, chat_type),
            "chat_state": self.get_chat_state(chat_id, chat_type),
            "context": self.load_conversation_context(chat_id, chat_type)
        }

    def save_session_id(self, chat_id: str, session_id: str, chat_type: str = "private") -> None:
        self.chat_sessions.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {"session_id": session_id, "created_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_session_id(self, chat_id: str, chat_type: str = "private") -> Optional[str]:
        doc = self.chat_sessions.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return doc.get("session_id") if doc else None

    def close(self):
        self.client.close()

# Create a global instance of the database manager.
_db_manager = DatabaseManager(MONGO_CONNECTION_STRING, DATABASE_NAME)

# Exported helper functions for external modules.
def save_business_info(chat_id: str, info: str, chat_type: str = "private") -> None:
    _db_manager.save_business_info(chat_id, info, chat_type)

def get_business_info(chat_id: str) -> str:
    return _db_manager.get_business_info(chat_id, "private")

def save_user_info(chat_id: str, info: str, date: Optional[str] = None) -> None:
    today = date or datetime.datetime.now().strftime("%Y-%m-%d")
    _db_manager.save_setting(chat_id, "user_info", {"info": info, "date": today})

def get_user_info(chat_id: str, date: Optional[str] = None) -> str:
    setting = _db_manager.get_setting(chat_id, "user_info")
    return setting.get("info", "") if isinstance(setting, dict) and "info" in setting else ""

def save_message_to_history(chat_id: str, role: str, content: str, session_id: Optional[str] = None) -> None:
    _db_manager.save_chat_message(chat_id, role, content, session_id or f"{chat_id}_default")

def get_chat_history(chat_id: str, limit: int = 50, session_id: Optional[str] = None) -> List[Dict]:
    return _db_manager.get_chat_history(chat_id, "private", session_id, limit=limit)

def load_chat_data(chat_id: str, chat_type: str = "private") -> Dict:
    return _db_manager.load_all_chat_data(chat_id, chat_type)

def save_ai_tone(chat_id: str, tone: str, chat_type: str = "private") -> None:
    _db_manager.save_ai_tone(chat_id, tone, chat_type)

def get_ai_tone(chat_id: str, chat_type: str = "private") -> str:
    return _db_manager.get_ai_tone(chat_id, chat_type)

def save_session_id(chat_id: str, session_id: str) -> None:
    _db_manager.save_session_id(chat_id, session_id)

def get_session_id(chat_id: str) -> Optional[str]:
    return _db_manager.get_session_id(chat_id, "private")

def save_conversation_summary(chat_id: str, session_id: str, summary: str) -> None:
    _db_manager.save_conversation_summary(chat_id, session_id, summary)

def get_conversation_summary(chat_id: str, session_id: str) -> str:
    return _db_manager.get_conversation_summary(chat_id, session_id)

# Export the global instance for direct use.
db_manager = _db_manager

__all__ = [
    'save_business_info',
    'get_business_info',
    'save_user_info',
    'get_user_info',
    'save_message_to_history',
    'get_chat_history',
    'load_chat_data',
    'save_ai_tone',
    'get_ai_tone',
    'save_session_id',
    'get_session_id',
    'save_conversation_summary',
    'get_conversation_summary',
    'db_manager'
]