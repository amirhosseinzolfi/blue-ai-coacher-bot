import datetime
from typing import Dict, List, Optional, Union
from pymongo import MongoClient, ASCENDING, IndexModel
from config import MONGO_CONNECTION_STRING, DATABASE_NAME

class DatabaseManager:
    def __init__(self, connection_string: str, database_name: str):
        # Use additional write concern parameters for durability and retryable writes.
        self.client = MongoClient(connection_string, retryWrites=True, w="majority", wtimeoutMS=10000)
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
        self.ai_tones = self.db["ai_tones"]      # For storing AI tones
        self.chat_summaries = self.db["chat_summaries"]  # For storing conversation summaries
        
        # Create necessary indexes
        self._setup_indexes()
        # Create additional indexes
        self._setup_additional_indexes()
    
    def _setup_indexes(self):
        """Setup necessary indexes with chat_type support for improved query performance."""
        # Business info indexes: unique on (chat_id, chat_type) and an index on updated_at.
        self.business_info.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True),
            IndexModel([("updated_at", ASCENDING)])
        ])
        
        # Chat history indexes: on (chat_id, chat_type), session_id, and timestamp.
        self.chat_history.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", ASCENDING)])
        ])
        
        # Chat sessions indexes: on (chat_id, chat_type) and created_at.
        self.chat_sessions.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)])
        ])
        
        # Chat metadata index: unique on (chat_id, chat_type).
        self.chat_metadata.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True)
        ])

    def _setup_additional_indexes(self):
        """Setup additional indexes for permanent data storage."""
        # Chat state indexes
        self.chat_state.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True),
            IndexModel([("updated_at", ASCENDING)])
        ])
        
        # AI tones indexes
        self.ai_tones.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)], unique=True)
        ])
        
        # Summaries indexes
        self.chat_summaries.create_indexes([
            IndexModel([("chat_id", ASCENDING), ("chat_type", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)])
        ])

    def save_chat_metadata(self, chat_id: str, chat_type: str, metadata: Dict) -> None:
        """Save or update chat metadata (for group or private chat info)."""
        self.chat_metadata.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {**metadata, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_chat_metadata(self, chat_id: str, chat_type: str) -> Dict:
        """Retrieve chat metadata for the specified chat."""
        doc = self.chat_metadata.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return doc if doc else {}

    def save_business_info(self, chat_id: str, info: str, chat_type: str = "private") -> None:
        """Save business information along with chat type."""
        self.business_info.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {"business_info": info, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_business_info(self, chat_id: str, chat_type: str = "private") -> str:
        """Retrieve business information for the specified chat."""
        doc = self.business_info.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return doc.get("business_info", "") if doc else ""

    def save_chat_message(self, chat_id: str, role: str, content: str, 
                          session_id: str, chat_type: str = "private") -> None:
        """Save a chat message to the history collection."""
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
        """Retrieve recent chat messages with optional session filtering."""
        query = {"chat_id": chat_id, "chat_type": chat_type}
        if session_id:
            query["session_id"] = session_id
        return list(self.chat_history.find(query).sort("timestamp", -1).limit(limit))

    def save_setting(self, chat_id: str, setting_type: str, value: any,
                     chat_type: str = "private") -> None:
        """Save or update a user/chat setting (such as AI tone or user info)."""
        self.settings.update_one(
            {"chat_id": chat_id, "setting_type": setting_type},
            {"$set": {"value": value, "chat_type": chat_type, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_setting(self, chat_id: str, setting_type: str) -> Optional[any]:
        """Retrieve a specific setting for a chat."""
        doc = self.settings.find_one({"chat_id": chat_id, "setting_type": setting_type})
        return doc.get("value") if doc else None

    def load_chat_context(self, chat_id: str, chat_type: str = "private") -> Dict:
        """Load chat context including business info, recent history, and settings."""
        return {
            "business_info": self.get_business_info(chat_id, chat_type),
            "chat_history": self.get_chat_history(chat_id, chat_type),
            "settings": {
                "ai_tone": self.get_setting(chat_id, "ai_tone"),
                "language": self.get_setting(chat_id, "language"),
            }
        }

    def clear_chat_history(self, chat_id: str) -> None:
        """Clear all chat history for a specified chat."""
        self.chat_history.delete_many({"chat_id": chat_id})

    def save_conversation_summary(self, chat_id: str, summary: str) -> None:
        """Save or update the conversation summary as a setting."""
        self.settings.update_one(
            {"chat_id": chat_id, "setting_type": "conversation_summary"},
            {"$set": {"value": summary, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )

    def get_conversation_summary(self, chat_id: str) -> str:
        """Retrieve the latest conversation summary."""
        doc = self.settings.find_one({"chat_id": chat_id, "setting_type": "conversation_summary"})
        return doc.get("value", "") if doc else ""

    def save_chat_state(self, chat_id: str, chat_type: str, state_data: Dict) -> None:
        """Save permanent chat state data."""
        self.chat_state.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {
                **state_data,
                "updated_at": datetime.datetime.utcnow()
            }},
            upsert=True
        )

    def get_chat_state(self, chat_id: str, chat_type: str) -> Dict:
        """Get permanent chat state data."""
        state = self.chat_state.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return state if state else {}

    def save_ai_tone(self, chat_id: str, tone: str, chat_type: str = "private") -> None:
        """Save AI tone preference permanently."""
        self.ai_tones.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {
                "tone": tone,
                "updated_at": datetime.datetime.utcnow()
            }},
            upsert=True
        )

    def get_ai_tone(self, chat_id: str, chat_type: str = "private") -> str:
        """Get saved AI tone preference."""
        doc = self.ai_tones.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return doc.get("tone", "دوستانه") if doc else "دوستانه"

    def save_conversation_context(self, chat_id: str, context_data: Dict, chat_type: str = "private") -> None:
        """Save complete conversation context including business info and settings."""
        self.chat_state.update_one(
            {"chat_id": chat_id, "chat_type": chat_type},
            {"$set": {
                "context": context_data,
                "updated_at": datetime.datetime.utcnow()
            }},
            upsert=True
        )

    def load_conversation_context(self, chat_id: str, chat_type: str = "private") -> Dict:
        """Load complete conversation context."""
        state = self.chat_state.find_one({"chat_id": chat_id, "chat_type": chat_type})
        return state.get("context", {}) if state else {}

    def load_all_chat_data(self, chat_id: str, chat_type: str = "private") -> Dict:
        """Load all permanent data for a chat."""
        return {
            "business_info": self.get_business_info(chat_id, chat_type),
            "ai_tone": self.get_ai_tone(chat_id, chat_type),
            "chat_state": self.get_chat_state(chat_id, chat_type),
            "context": self.load_conversation_context(chat_id, chat_type)
        }

    def close(self):
        """Close the MongoDB client connection."""
        self.client.close()

# Global instance of the database manager.
_db_manager = DatabaseManager(MONGO_CONNECTION_STRING, DATABASE_NAME)

# Exported helper functions for external modules.
def save_business_info(chat_id: str, info: str, chat_type: str = "private") -> None:
    _db_manager.save_business_info(chat_id, info, chat_type)

def get_business_info(chat_id: str) -> str:
    return _db_manager.get_business_info(chat_id, "private")

def save_user_info(chat_id: str, info: str, date: Optional[str] = None) -> None:
    # Save user info as a setting with today's date.
    today = date or datetime.datetime.now().strftime("%Y-%m-%d")
    _db_manager.save_setting(chat_id, "user_info", {"info": info, "date": today})

def get_user_info(chat_id: str, date: Optional[str] = None) -> str:
    setting = _db_manager.get_setting(chat_id, "user_info")
    return setting.get("info", "") if isinstance(setting, dict) and "info" in setting else ""

def save_message_to_history(chat_id: str, role: str, content: str, session_id: Optional[str] = None) -> None:
    _db_manager.save_chat_message(chat_id, role, content, session_id or f"{chat_id}_default")

def get_chat_history(chat_id: str, limit: int = 50) -> List[Dict]:
    return _db_manager.get_chat_history(chat_id, "private", limit=limit)

def load_chat_data(chat_id: str, chat_type: str = "private") -> Dict:
    return _db_manager.load_all_chat_data(chat_id, chat_type)

def save_ai_tone(chat_id: str, tone: str, chat_type: str = "private") -> None:
    _db_manager.save_ai_tone(chat_id, tone, chat_type)

def get_ai_tone(chat_id: str, chat_type: str = "private") -> str:
    return _db_manager.get_ai_tone(chat_id, chat_type)

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
    'db_manager'
]
