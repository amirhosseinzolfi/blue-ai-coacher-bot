import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "blue_business_db"
BUSINESS_INFO_COLLECTION = "business_info"
CONVERSATION_HISTORY_COLLECTION = "conversation_history"

if not MONGO_URI:
    logging.error("MONGO_URI environment variable not set.")
    # Depending on your application's needs, you might want to exit or raise an exception here.
    # For now, we'll allow it to proceed but database operations will fail.
    client = None
else:
    try:
        client = MongoClient(MONGO_URI)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        logging.info("Successfully connected to MongoDB.")
        db = client[DB_NAME]
        business_info_coll = db[BUSINESS_INFO_COLLECTION]
        conversation_history_coll = db[CONVERSATION_HISTORY_COLLECTION]
        # Create indexes for faster lookups
        business_info_coll.create_index("chat_id", unique=True)
        conversation_history_coll.create_index("chat_id") # Not unique, as one chat can have many messages
    except ConnectionFailure:
        logging.error("Failed to connect to MongoDB.")
        client = None
    except Exception as e:
        logging.error(f"An error occurred during MongoDB initialization: {e}")
        client = None

def is_db_connected():
    """Check if the MongoDB client is connected."""
    return client is not None

def save_business_info(chat_id: int, info: dict):
    """Saves or updates business information for a given chat_id."""
    if not is_db_connected():
        logging.error("Database not connected. Cannot save business info.")
        return False
    try:
        business_info_coll.update_one(
            {"chat_id": chat_id},
            {"$set": info},
            upsert=True  # Creates the document if it doesn't exist
        )
        logging.info(f"Saved/Updated business info for chat_id: {chat_id}")
        return True
    except Exception as e:
        logging.error(f"Error saving business info for chat_id {chat_id}: {e}")
        return False

def load_business_info(chat_id: int) -> dict | None:
    """Loads business information for a given chat_id."""
    if not is_db_connected():
        logging.error("Database not connected. Cannot load business info.")
        return None
    try:
        info = business_info_coll.find_one({"chat_id": chat_id})
        if info:
            logging.info(f"Loaded business info for chat_id: {chat_id}")
            # Remove MongoDB's internal _id before returning
            info.pop('_id', None)
            return info
        else:
            logging.info(f"No business info found for chat_id: {chat_id}")
            return None
    except Exception as e:
        logging.error(f"Error loading business info for chat_id {chat_id}: {e}")
        return None

def save_conversation_message(chat_id: int, message: dict):
    """Saves a single conversation message for a given chat_id."""
    if not is_db_connected():
        logging.error("Database not connected. Cannot save conversation message.")
        return False
    try:
        # Add chat_id to the message document itself for easier querying
        message_doc = {"chat_id": chat_id, **message}
        conversation_history_coll.insert_one(message_doc)
        # Optional: Add TTL index if you want messages to expire automatically
        # Ensure you have created a TTL index on a date field in MongoDB:
        # conversation_history_coll.create_index("timestamp", expireAfterSeconds=2592000) # 30 days
        logging.debug(f"Saved message for chat_id: {chat_id}")
        return True
    except Exception as e:
        logging.error(f"Error saving message for chat_id {chat_id}: {e}")
        return False

def load_conversation_history(chat_id: int, limit: int = 50) -> list[dict]:
    """Loads the most recent conversation history for a given chat_id."""
    if not is_db_connected():
        logging.error("Database not connected. Cannot load conversation history.")
        return []
    try:
        # Find messages, sort by a timestamp field (descending) if available, limit results
        # Assuming messages have a 'timestamp' field. If not, MongoDB's natural order might be used,
        # or you might need to adjust the sorting key based on your message structure.
        # If no timestamp, sorting by '_id' (descending) often gives recent messages.
        history = list(conversation_history_coll.find({"chat_id": chat_id})
                       .sort([("_id", -1)]) # Sort by insertion time (descending) as a proxy for timestamp
                       .limit(limit))
        # Reverse the list to get chronological order (oldest first)
        history.reverse()
        logging.info(f"Loaded {len(history)} messages for chat_id: {chat_id}")
        # Remove MongoDB's internal _id before returning
        for msg in history:
            msg.pop('_id', None)
        return history
    except Exception as e:
        logging.error(f"Error loading conversation history for chat_id {chat_id}: {e}")
        return []

# Optional: Function to clear history (useful for testing or specific commands)
def clear_conversation_history(chat_id: int):
    """Deletes all conversation history for a given chat_id."""
    if not is_db_connected():
        logging.error("Database not connected. Cannot clear conversation history.")
        return False
    try:
        result = conversation_history_coll.delete_many({"chat_id": chat_id})
        logging.info(f"Deleted {result.deleted_count} messages for chat_id: {chat_id}")
        return True
    except Exception as e:
        logging.error(f"Error clearing conversation history for chat_id {chat_id}: {e}")
        return False

def close_db_connection():
    """Closes the MongoDB connection if it's open."""
    global client
    if client:
        try:
            client.close()
            logging.info("MongoDB connection closed.")
            client = None
        except Exception as e:
            logging.error(f"Error closing MongoDB connection: {e}")

# You might add more functions here as needed, e.g., for updating specific messages, etc.
