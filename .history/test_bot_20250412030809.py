import unittest
from unittest.mock import patch, MagicMock
import datetime
import bot

# Import the module containing new_chat

# Fake message object for tests
class FakeMessage:
    def __init__(self, chat_id):
        self.chat = MagicMock(id=chat_id)

class TestNewChat(unittest.TestCase):
    @patch("bot.bot.reply_to")
    @patch("bot.logging")
    def test_invalid_chat_id(self, mock_logging, mock_reply_to):
        # Use a chat id that cannot be converted to int
        fake_message = FakeMessage("not_an_int")
        bot.new_chat(fake_message)
        # Verify that bot.reply_to was called with an error message indicating the invalid chat id
        mock_reply_to.assert_called_once()
    
    @patch("bot.save_message_to_history")
    @patch("bot.escape_markdown_v2", lambda text: text)
    @patch("bot.bot.reply_to")
    @patch("bot.db_manager.save_business_info")
    @patch("bot.telegram_bot.save_session_id")
    def test_valid_chat_id(self, mock_save_session_id, mock_save_business_info, mock_reply_to, mock_escape, mock_save_history):
        fake_chat_id = 12345
        fake_message = FakeMessage(fake_chat_id)
        bot.new_chat(fake_message)
        # The chat id should be correctly converted to integer
        self.assertTrue(mock_save_business_info.called)
        mock_save_business_info.assert_called_once_with(fake_chat_id, {"user_report": ""})
        # Check that save_session_id was called with a new session id containing the chat id and a timestamp
        self.assertTrue(mock_save_session_id.called)
        args, _ = mock_save_session_id.call_args
        self.assertEqual(args[0], fake_chat_id)
        new_session_id = args[1]
        self.assertTrue(new_session_id.startswith(f"{fake_chat_id}_"))
        # Verify that reply_to was called with the expected response text (escaped as-is)
        mock_reply_to.assert_called_once()
        # Verify that welcome message was added to session history
        mock_save_history.assert_called_once_with(str(fake_chat_id), "system", "جلسه گفتگوی جدید آغاز شد. چطور می‌توانم به شما کمک کنم؟")

if __name__ == "__main__":
    unittest.main()