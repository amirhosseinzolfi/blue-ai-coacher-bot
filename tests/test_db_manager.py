import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from pymongo import MongoClient
from db_manager import DatabaseManager, db_manager

# Test configuration
TEST_MONGO_URI = "mongodb://localhost:27017/"
TEST_DB_NAME = "test_blue_business"

@pytest.fixture(autouse=True)
def setup_test_db():
    """Setup test database and cleanup after tests"""
    # Create test database instance
    test_db = DatabaseManager(TEST_MONGO_URI, TEST_DB_NAME)
    
    # Clear all collections before each test
    for collection in test_db.db.list_collection_names():
        test_db.db[collection].delete_many({})
    
    yield test_db
    
    # Cleanup after tests
    test_db.client.drop_database(TEST_DB_NAME)
    test_db.close()

class TestDatabaseManager:
    def test_save_and_get_business_info(self, setup_test_db):
        """Test saving and retrieving business information"""
        db = setup_test_db
        chat_id = "test123"
        info = "Test Business Info"
        
        # Test save
        db.save_business_info(chat_id, info)
        
        # Test retrieve
        retrieved = db.get_business_info(chat_id)
        assert retrieved == info
        
        # Test update
        new_info = "Updated Business Info"
        db.save_business_info(chat_id, new_info)
        updated = db.get_business_info(chat_id)
        assert updated == new_info

    def test_chat_message_operations(self, setup_test_db):
        """Test chat message operations"""
        db = setup_test_db
        chat_id = "chat123"
        session_id = "session123"
        
        # Test save message
        message_content = "Test message"
        db.save_chat_message(chat_id, "user", message_content, session_id)
        
        # Test retrieve messages
        messages = db.get_chat_history(chat_id)
        assert len(messages) == 1
        assert messages[0]["content"] == message_content
        assert messages[0]["role"] == "user"

    def test_settings_operations(self, setup_test_db):
        """Test settings operations"""
        db = setup_test_db
        chat_id = "chat123"
        
        # Test save setting
        setting_value = {"tone": "friendly"}
        db.save_setting(chat_id, "ai_tone", setting_value)
        
        # Test get setting
        retrieved = db.get_setting(chat_id, "ai_tone")
        assert retrieved == setting_value

    def test_chat_state_operations(self, setup_test_db):
        """Test chat state operations"""
        db = setup_test_db
        chat_id = "chat123"
        chat_type = "private"
        
        # Test save state
        state_data = {"last_command": "start", "user_name": "test_user"}
        db.save_chat_state(chat_id, chat_type, state_data)
        
        # Test get state
        retrieved = db.get_chat_state(chat_id, chat_type)
        assert retrieved["last_command"] == state_data["last_command"]
        assert retrieved["user_name"] == state_data["user_name"]

    def test_ai_tone_operations(self, setup_test_db):
        """Test AI tone operations"""
        db = setup_test_db
        chat_id = "chat123"
        tone = "professional"
        
        # Test save tone
        db.save_ai_tone(chat_id, tone)
        
        # Test get tone
        retrieved = db.get_ai_tone(chat_id)
        assert retrieved == tone
        
        # Test default tone
        default_tone = db.get_ai_tone("nonexistent_chat")
        assert default_tone == "دوستانه"

    def test_conversation_context(self, setup_test_db):
        """Test conversation context operations"""
        db = setup_test_db
        chat_id = "chat123"
        context_data = {
            "topic": "business",
            "last_interaction": datetime.now().isoformat(),
            "user_preferences": {"language": "fa"}
        }
        
        # Test save context
        db.save_conversation_context(chat_id, context_data)
        
        # Test load context
        retrieved = db.load_conversation_context(chat_id)
        assert retrieved["topic"] == context_data["topic"]
        assert retrieved["user_preferences"] == context_data["user_preferences"]

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, setup_test_db):
        """Test concurrent database operations"""
        db = setup_test_db
        chat_id = "chat123"
        
        import asyncio
        
        async def save_messages(count: int):
            for i in range(count):
                db.save_chat_message(
                    chat_id,
                    "user",
                    f"Message {i}",
                    f"session_{i}"
                )
        
        # Run concurrent message saves
        await asyncio.gather(
            save_messages(5),
            save_messages(5)
        )
        
        # Verify all messages were saved
        messages = db.get_chat_history(chat_id)
        assert len(messages) == 10

    def test_error_handling(self, setup_test_db):
        """Test error handling in database operations"""
        db = setup_test_db
        
        # Test invalid chat_id
        with pytest.raises(Exception):
            db.save_business_info(None, "test")
        
        # Test invalid session_id
        with pytest.raises(Exception):
            db.save_chat_message("chat123", "user", "test", None)

    def test_performance(self, setup_test_db):
        """Test performance of database operations"""
        db = setup_test_db
        chat_id = "chat123"
        
        # Measure bulk insert performance
        start_time = datetime.now()
        for i in range(100):
            db.save_chat_message(
                chat_id,
                "user",
                f"Message {i}",
                f"session_{i}"
            )
        duration = (datetime.now() - start_time).total_seconds()
        
        # Assert reasonable performance
        assert duration < 5.0  # Should complete within 5 seconds
        
        # Test query performance
        start_time = datetime.now()
        messages = db.get_chat_history(chat_id, limit=50)
        query_duration = (datetime.now() - start_time).total_seconds()
        
        # Assert query performance
        assert query_duration < 1.0  # Should complete within 1 second
        assert len(messages) == 50

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
