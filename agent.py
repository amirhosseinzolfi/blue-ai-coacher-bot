import time
import logging
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from llm_setup import llm
from memory import optimize_memory, message_counter
from prompts_loader import prompt_template_text
from utils.helpers import strip_thinking_tags
from utils.rich_logger import log_comprehensive_interaction, log_user_business_data

logger = logging.getLogger(__name__)

def agent(state):
    """
    Calls the LLM to generate a response from the formatted conversation.
    Handles both text-only and multimodal inputs.
    """
    chat_id = state["chat_id"]
    username = state.get("username", "")
    logger.info(f"🧠 Processing agent node for chat: {chat_id}")
    logger.info(f"Username: {username if username else 'Not provided'}")

    ai_tone = state.get("ai_tone", "دوستانه")
    business_info = state.get("business_info", "")
    messages = state["messages"]

    try:
        # Build the single system instruction
        formatted_system_prompt = prompt_template_text.format(
            business_info=business_info,
            user_name=username or "کاربر"
        )
        system_instruction_content = (
            formatted_system_prompt
            + (f"\n\nADDRESS USER: Always address the user as {username}" if username else "")
            + f"\n\nTONE: {ai_tone}"
        )
        system_message = SystemMessage(content=system_instruction_content)

        # Prepare messages for LLM
        messages_for_llm = [system_message] + list(messages)

        # Capture final prompts for logging
        final_system_instruction = system_instruction_content
        final_user_prompt = messages_for_llm[-1].content if messages_for_llm else ""

        start_time = time.time()
        response = llm.invoke(messages_for_llm)
        duration = time.time() - start_time
        ai_message = AIMessage(content=response.content)

        log_comprehensive_interaction(
            logger, chat_id, state.get("session_id", "N/A"),
            final_system_instruction,
            str(final_user_prompt),
            ai_message.content, ai_tone, llm.model_name
        )

        if username:
            for placeholder in ["[نام کاربر]", "[name]", "[نام]"]:
                ai_message.content = ai_message.content.replace(placeholder, username)
        ai_message.content = strip_thinking_tags(ai_message.content)

        logger.info(f"📤 Response: {ai_message.content[:50]}...")
        new_state = state.copy()
        new_state["messages"] = list(state["messages"]) + [ai_message]
        if message_counter.increment_and_check(chat_id):
            new_state = optimize_memory(new_state)
        return new_state
    except Exception as e:
        logger.error(f"❌ Error during LLM call: {e}", exc_info=True)
        error_message = AIMessage(content="متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.")
        new_state = state.copy()
        new_state["messages"] = list(state["messages"]) + [error_message]
        return new_state

def route_tool(state):
    """
    Routes the agent to the optimize_memory function.
    """
    return "optimize_memory"