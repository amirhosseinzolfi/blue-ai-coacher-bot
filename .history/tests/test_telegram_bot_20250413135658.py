import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from telegram_bot import run_agent, new_chat_session, get_user_business_info

# Dummy response class to simulate LLM output
class DummyResponse:
    def __init__(self, content):
        self.content = content

def test_combined_telegram_bot_functions(monkeypatch):
    # Step 1: Test new_chat_session functionality
    chat_id = "combinedTest123"
    session_id = new_chat_session(chat_id)
    assert session_id.startswith(chat_id), "Session ID should start with the chat ID."
    assert "_" in session_id, "Session ID should contain an underscore."

    # Step 2: Test get_user_business_info functionality
    info = get_user_business_info(chat_id)
    assert isinstance(info, str), "Business info should be a string."

    # Step 3: Override LLM invocation methods to return dummy responses.
    monkeypatch.setattr("telegram_bot.llm.invoke", lambda messages: DummyResponse("Dummy LLM Response"))
    monkeypatch.setattr("telegram_bot.user_llm.invoke", lambda messages: DummyResponse("Dummy User Report Response"))
    monkeypatch.setattr("telegram_bot.llm_summary.invoke", lambda messages: DummyResponse("Dummy Summary Response"))

    # Step 4: Test run_agent functionality
    query = "Test query for run_agent"
    message_id = "msg1"
    # Adding an optional username parameter to simulate realistic conditions.
    response = run_agent(query, chat_id, message_id, username="tester")
    assert isinstance(response, str), "Agent response should be a string."
    assert len(response.strip()) > 0, "Agent response should not be empty."