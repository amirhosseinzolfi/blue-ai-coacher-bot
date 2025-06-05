import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from telegram_bot import run_agent, new_chat_session, get_user_business_info
import sys
import os
import datetime
from langchain.schema import HumanMessage, AIMessage, SystemMessage

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
    # filepath: /root/blue_business/tests/test_telegram_bot.py
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


    # Import necessary types from langchain

    # Dummy response class to simulate LLM output
    class DummyResponse:
        def __init__(self, content):
            self.content = content

    def test_telegram_bot_functions(monkeypatch):
        # Test new_chat_session
        chat_id = "test123"
        session_id = new_chat_session(chat_id)
        assert session_id.startswith(chat_id)
        assert "_" in session_id

        # Test get_user_business_info (expected to be empty string if no data exists)
        info = get_user_business_info(chat_id)
        assert isinstance(info, str)

        # Override LLM invocation methods to return dummy responses
        monkeypatch.setattr("telegram_bot.llm.invoke", lambda messages: DummyResponse("Dummy LLM Response"))
        monkeypatch.setattr("telegram_bot.user_llm.invoke", lambda messages: DummyResponse("Dummy User Report Response"))
        monkeypatch.setattr("telegram_bot.llm_summary.invoke", lambda messages: DummyResponse("Dummy Summary Response"))

        # Call run_agent and check response
        query = "Test query for run_agent"
        message_id = "msg1"
        response = run_agent(query, chat_id, message_id, username="tester")
        assert isinstance(response, str)
        assert len(response.strip()) > 0