import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from telegram_bot import run_agent, new_chat_session, get_user_business_info

# Mock data for testing
def test_new_chat_session():
    chat_id = "12345"
    session_id = new_chat_session(chat_id)
    assert session_id.startswith(chat_id)
    assert len(session_id.split("_")) == 2

def test_get_user_business_info():
    chat_id = "12345"
    info = get_user_business_info(chat_id)
    assert isinstance(info, str)  # Ensure it returns a string
    assert info == ""  # Assuming no data exists for this chat_id

def test_run_agent():
    query = "What is the weather today?"
    chat_id = "12345"
    message_id = "1"
    response = run_agent(query, chat_id, message_id)
    assert isinstance(response, str)
    assert len(response) > 0