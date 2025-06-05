import schedule
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Set

from langgraph_code import logger, new_chat_session
from db_manager import db_manager

class DailyResetScheduler:
    def __init__(self):
        self.active_chats: Set[str] = set()
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        
    def add_chat(self, chat_id: str):
        with self.lock:
            self.active_chats.add(str(chat_id))
            
    def reset_all_sessions(self):
        current_time = datetime.now().strftime("%H:%M")
        logger.info(f"🔄 Running scheduled session reset at {current_time}")
        with self.lock:
            for chat_id in self.active_chats:
                try:
                    new_session_id = new_chat_session(chat_id)
                    logger.info(f"✅ Daily reset: Created new session {new_session_id} for chat {chat_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to reset session for chat {chat_id}: {e}")
    
    def run_scheduler(self):
        schedule.every().day.at("00:00").do(self.reset_all_sessions)
        self._running = True
        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    def start(self):
        if not self._thread:
            self._thread = threading.Thread(target=self.run_scheduler, daemon=True)
            self._thread.start()
            logger.info("🕒 Daily reset scheduler started")
            
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None

# Create global instance
scheduler = DailyResetScheduler()
