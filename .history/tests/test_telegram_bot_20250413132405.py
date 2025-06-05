import pytest
from telegram_bot import agent
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# A fake response class to simulate llm.invoke return value
class FakeResponse:
    def __init__(self, content):
        self.content = content

# A fake LLM class with an invoke method
class FakeLLM:
    def invoke(self, messages):
        return FakeResponse("This is a fake response.")

# Fixture providing a sample state for the agent with minimal required structure.
@pytest.fixture
def sample_state():
    return {
        "messages": [
            SystemMessage(content="System: initial state"),
            HumanMessage(content="Hello!")
        ],
        "tool_calls": [],
        "requires_tool": False,
        "current_tool": None,
        "chat_id": "test_chat",
        "username": "tester"
    }

# Fixture that patches the global `llm` in telegram_bot with our FakeLLM
@pytest.fixture(autouse=True)
def patch_llm(monkeypatch):
    from telegram_bot import llm
    fake_llm = FakeLLM()
    monkeypatch.setattr("telegram_bot.llm", fake_llm)

def test_agent_response(sample_state):
    # Call the agent function and verify that an AIMessage was appended.
    new_state = agent(sample_state)
    messages = new_state["messages"]
    assert len(messages) == len(sample_state["messages"]) + 1
    last_message = messages[-1]
    assert isinstance(last_message, AIMessage)
    assert "This is a fake response." in last_message.content