import threading
import logging
from utils.rich_logger import setup_logger

logger = setup_logger(level=logging.INFO)

class MessageCounter:
    def __init__(self):
        self._counters = {}
        self._lock = threading.Lock()

    def increment_and_check(self, chat_id: str) -> bool:
        with self._lock:
            self._counters[chat_id] = self._counters.get(chat_id, 0) + 1
            if self._counters[chat_id] >= 5:  # Summarize every 5 messages
                self._counters[chat_id] = 0
                return True
            return False

    def reset(self, chat_id: str):
        with self._lock:
            self._counters[chat_id] = 0

message_counter = MessageCounter()

# Constants
PRIMARY_LLM_MODEL = "gpt-4o"
BUSINESS_LLM_MODEL = "gemini-2.0-flash"
USER_REPORT_LLM_MODEL = "gpt-4o"
SUMMARY_LLM_MODEL = "gemini-2.0-flash"
