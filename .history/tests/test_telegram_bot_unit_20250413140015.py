import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from telegram_bot import new_chat_session, get_user_business_info, run_agent

# Dummy response class to simulate LLM output
class DummyResponse:
    def __init__(self, content):
        self.content = content

# Dummy invoke function that returns our dummy response
def dummy_invoke(messages, **kwargs):
    return DummyResponse("Dummy LLM Response")

# Monkeypatch LLM invocations automatically for all tests
@pytest.fixture(autouse=True)
def patch_llm(monkeypatch):
    monkeypatch.setattr("telegram_bot.llm.invoke", dummy_invoke)
    monkeypatch.setattr("telegram_bot.user_llm.invoke", dummy_invoke)
    monkeypatch.setattr("telegram_bot.llm_summary.invoke", dummy_invoke)

def test_new_chat_session():
    # Test that a new chat session id is generated correctly
    chat_id = "test_chat"
    session_id = new_chat_session(chat_id)
    assert session_id.startswith(chat_id)
    parts = session_id.split("_")
    assert len(parts) == 2

def test_get_user_business_info():
    # Test that get_user_business_info returns a string (expected empty if no data)
    chat_id = "new_chat_for_test"
    info = get_user_business_info(chat_id)
    assert isinstance(info, str)
    assert info == ""

def test_run_agent():
    # Test that run_agent uses our dummy LLM response
    chat_id = "agent_test_chat"
    message_id = "1"
    query = "What is the weather like?"
    username = "tester"
    response = run_agent(query, chat_id, message_id, username)
    assert isinstance(response, str)
    assert response.strip() == "Dummy LLM Response"
