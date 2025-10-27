import os
import threading
import logging
import traceback

# --------------------------
# Setup Logging
# --------------------------
LOG_FORMAT = "%(asctime)s [%(threadName)s] %(levelname)s: %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        # optionally also FileHandler if you want logs in a file
        # logging.FileHandler("chatbot.log"),
    ]
)
logger = logging.getLogger("main")

# If g4f has its own internal logger or debug switch:
try:
    import g4f
    # enable debug logging in g4f, if available
    if hasattr(g4f, "debug"):
        try:
            g4f.debug.logging = True
            logger.info("Enabled g4f internal logging.")
        except Exception as e:
            logger.warning(f"Could not enable g4f.debug.logging: {e}")
    else:
        logger.debug("g4f.debug attribute not found; cannot enable internal g4f logging.")
except ImportError:
    g4f = None
    logger.info("g4f module not installed.")

# --------------------------
# G4F API Server Bootstrap
# --------------------------
try:
    from g4f.api import run_api
except ImportError:
    run_api = None
    logger.info("g4f.api module not found. API server won't start.")

if run_api:
    def _start_g4f():
        try:
            logger.info("Starting G4F API server on http://0.0.0.0:1555/v1 …")
            run_api(bind="0.0.0.0:1555")
            logger.info("G4F API server has stopped (exited run_api).")
        except Exception as e:
            logger.error(f"Error during G4F API server startup: {e}")
            logger.debug(traceback.format_exc())

    api_thread = threading.Thread(target=_start_g4f, daemon=True, name="G4F-API-Thread")
    api_thread.start()
else:
    logger.warning("G4F API server not available (run_api is None).")

# --------------------------
# Terminal Chatbot Script
# --------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ANSI escape codes for colors (optional)
class Colors:
    USER  = '\033[92m'  # Green
    BOT   = '\033[94m'  # Blue
    ERROR = '\033[91m'  # Red
    INFO  = '\033[93m'  # Yellow
    ENDC  = '\033[0m'   # Reset

AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking",
    "gemini-1.5-pro",
    "gemini-2.0-pro",
    "gpt-4o",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-4o-mini",
    "o1",
    "o3-mini",
    "llama-3.3-70b",
    "deepseek-r1",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
]

def select_model():
    logger.debug("Displaying model selection menu.")
    print(f"{Colors.INFO}Please select an LLM model to use:{Colors.ENDC}")
    for i, model_name in enumerate(AVAILABLE_MODELS):
        print(f"{Colors.INFO}{i + 1}. {model_name}{Colors.ENDC}")

    while True:
        try:
            choice = input(f"{Colors.USER}Enter the number of the model: {Colors.ENDC}").strip()
            model_index = int(choice) - 1
            if 0 <= model_index < len(AVAILABLE_MODELS):
                logger.info(f"Model selected: {AVAILABLE_MODELS[model_index]} (index {model_index})")
                return model_index
            else:
                logger.warning(f"Invalid model number: {choice}")
                print(f"{Colors.ERROR}Invalid selection. Please enter a number between 1 and {len(AVAILABLE_MODELS)}.{Colors.ENDC}")
        except ValueError:
            logger.warning(f"Non-integer input during model select: {choice}")
            print(f"{Colors.ERROR}Invalid input. Please enter a number.{Colors.ENDC}")
        except Exception as e:
            logger.error(f"Error in select_model: {e}")
            logger.debug(traceback.format_exc())

def main_chat_loop():
    model_index = select_model()
    if model_index is None:
        logger.warning("No model selected; exiting.")
        return

    selected_model_name = AVAILABLE_MODELS[model_index]
    logger.info(f"Initializing ChatBot LLM with model: {selected_model_name}")
    try:
        llm = ChatOpenAI(
            base_url="http://localhost:1555/v1",
            model_name=selected_model_name,
            temperature=0.5,
            api_key="11"  # replace with a valid key if needed
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{user_input}")
        ])
        parser = StrOutputParser()
        chain = prompt | llm | parser

        logger.info("ChatBot ready. Type 'quit' or 'exit' to exit.")
        print(f"{Colors.INFO}ChatBot ready. Type 'quit' to exit.{Colors.ENDC}")
        print(f"{Colors.INFO}{'-'*30}{Colors.ENDC}")

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.debug(traceback.format_exc())
        return

    while True:
        try:
            user_input = input(f"{Colors.USER}You: {Colors.ENDC}").strip()
            if user_input.lower() in ("quit", "exit"):
                logger.info("User requested exit; shutting down main loop.")
                print(f"{Colors.INFO}Goodbye!{Colors.ENDC}")
                break
            if not user_input:
                continue

            logger.debug(f"User input: {user_input}")
            response = chain.invoke({"user_input": user_input})
            logger.debug(f"LLM response: {response}")
            print(f"{Colors.BOT}Bot: {response}{Colors.ENDC}")

        except ConnectionError as ce:
            logger.error(f"Connection error: {ce}")
            logger.debug(traceback.format_exc())
            print(f"{Colors.ERROR}Connection error: {ce}{Colors.ENDC}")
        except Exception as e:
            logger.error(f"General error in chat loop: {e}")
            logger.debug(traceback.format_exc())
            print(f"{Colors.ERROR}Error: {e}{Colors.ENDC}")

if __name__ == "__main__":
    logger.info("Script started.")
    main_chat_loop()
    logger.info("Script finished.")
