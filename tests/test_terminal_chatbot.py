import os
import threading

# --------------------------
# G4F API Server Bootstrap
# --------------------------
try:
    from g4f.api import run_api
except ImportError:
    run_api = None

if run_api:
    def _start_g4f():
        print("Starting G4F API server on http://localhost:1555/v1 …")
        run_api(bind="0.0.0.0:1555")
    threading.Thread(target=_start_g4f, daemon=True, name="G4F-API-Thread").start()
else:
    print("g4f.api module not found. Install the 'g4f' package to run the local API server.")

# --------------------------
# Terminal Chatbot Script
# --------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ANSI escape codes for colors
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
    "gpt-4o-mini",
    "o1",
    "o3-mini",
    "llama-3.3-70b",
    "deepseek-r1",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
]

def select_model():
    print(f"{Colors.INFO}Please select an LLM model to use:{Colors.ENDC}")
    for i, model_name in enumerate(AVAILABLE_MODELS):
        print(f"{Colors.INFO}{i + 1}. {model_name}{Colors.ENDC}")

    while True:
        try:
            choice = input(f"{Colors.USER}Enter the number of the model: {Colors.ENDC}").strip()
            model_index = int(choice) - 1
            if 0 <= model_index < len(AVAILABLE_MODELS):
                return model_index
            else:
                print(f"{Colors.ERROR}Invalid selection. Please enter a number between 1 and {len(AVAILABLE_MODELS)}.{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.ERROR}Invalid input. Please enter a number.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.ERROR}An error occurred: {e}{Colors.ENDC}")


def main_chat_loop():
    selected_model_index = select_model()
    if selected_model_index is None:
        return

    selected_model_name = AVAILABLE_MODELS[selected_model_index]
    print(f"{Colors.INFO}Initializing ChatBot LLM with model: {selected_model_name}…{Colors.ENDC}")
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

        print(f"{Colors.INFO}ChatBot ready. Type 'quit' to exit.{Colors.ENDC}")
        print(f"{Colors.INFO}{'-'*30}{Colors.ENDC}")

    except Exception as e:
        print(f"{Colors.ERROR}Initialization error: {e}{Colors.ENDC}")
        return

    while True:
        try:
            user_input = input(f"{Colors.USER}You: {Colors.ENDC}").strip()
            if user_input.lower() in ("quit", "exit"):
                print(f"{Colors.INFO}Goodbye!{Colors.ENDC}")
                break
            if not user_input:
                continue

            response = chain.invoke({"user_input": user_input})
            print(f"{Colors.BOT}Bot: {response}{Colors.ENDC}")

        except ConnectionError as ce:
            print(f"{Colors.ERROR}Connection error: {ce}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.ERROR}Error: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main_chat_loop()
