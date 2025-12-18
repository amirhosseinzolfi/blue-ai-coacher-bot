
# Blue AI Coacher Bot (بلو)

![GitHub last commit](https://img.shields.io/github/last-commit/amirhosseinzolfi/blue-ai-coacher-bot)
![GitHub issues](https://img.shields.io/github/issues/amirhosseinzolfi/blue-ai-coacher-bot)
![GitHub pull requests](https://img.shields.io/github/issues-pr/amirhosseinzolfi/blue-ai-coacher-bot)
![MIT License](https://img.shields.io/github/license/amirhosseinzolfi/blue-ai-coacher-bot)


**Blue (بلو)** is a personalized Telegram group business coaching bot designed to assist business teams and members by leveraging business and team member data. It aims to provide professional, friendly, and engaging coaching through AI.

## 🚀 Features

*   **Personalized Coaching:** Aware of business and team member data to provide tailored advice.
*   **Persian Language Support:** All interactions are primarily in Persian.
*   **Task Management:**
    *   Generation of daily task lists.
    *   Tracking user tasks and activities.
*   **Reporting:**
    *   Daily user activity reports (analyzing chat history, tasks, in/out times, efficiency).
    *   Conversation summarization.
*   **Content Generation:**
    *   Creative Instagram story and post ideas based on business information.
    *   Image generation based on user prompts.
*   **Business Insights:**
    *   Summarization of business information (domain, goals, team members, roles, skills).
    *   Analysis of images for insights.
*   **Team Engagement:**
    *   Daily coaching tips and productivity guides.
    *   Leaderboard generation based on user activity and efficiency.
*   **Jira Integration (MCP):**
    *   Real-time access to Jira tasks, sprints, and issues.
    *   Automatic detection of Jira-related queries.
    *   Task management and sprint tracking through natural language.
*   **Customizable Interaction:**
    *   Adjustable AI tone (Friendly, Professional, Creative).
    *   Interaction via direct commands, mentions (`بلو` or `@Blue`), or inline/menu buttons.
*   **Contextual Awareness:** Utilizes chat history, business info, and user reports for relevant responses.
*   **Structured Output:** Uses Markdown, titles, bullet points, and emojis for clear and engaging communication.

## 🛠️ Core Technologies

*   **Python 3.x**
*   **Telegram Bot API:** via `pyTelegramBotAPI` library for Telegram integration.
*   **LangChain & LangGraph:** For building sophisticated LLM-powered applications and agentic workflows.
*   **OpenAI & Gemini Models:** Leverages models like GPT-4o and Gemini-2.5-Flash via G4F and direct API calls for AI responses.
*   **G4F (GPT4Free):** Provides access to various LLM models.
*   **SQLite:** For database management (chat history, business info, user reports, LangGraph checkpoints).
*   **Rich:** For enhanced terminal logging and display.
*   **Schedule:** For daily automated tasks like resets.
*   **MCP (Model Context Protocol):** For Jira integration via LangChain adapters.

## ⚙️ Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher.
    *   Access to a Telegram Bot Token.
    *   API keys for OpenAI/Google AI Studio if using proprietary models directly (though G4F is also used).


2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/amirhosseinzolfi/blue-ai-coacher-bot.git
    cd blue-ai-coacher-bot
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```


4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # For Jira integration (optional):
    pip install -r requirements_jira.txt
    ```


5.  **Configuration:**
    *   Copy `config.py.example` to `config.py`:
        ```bash
        cp config.py.example config.py
        ```
    *   Edit `config.py` and fill in your credentials:
        *   `TELEGRAM_BOT_TOKEN` - Your Telegram bot token from @BotFather
        *   `OPENAI_API_KEY` - Your OpenAI API key (optional, G4F can be used)
        *   `GOOGLE_API_KEY` - Your Google AI Studio API key for Gemini models
        *   `JIRA_ENABLED` - Set to `true` to enable Jira integration (default: true)
        *   `JIRA_MCP_URL` - Jira MCP server URL (default: http://localhost:9000/mcp)
        *   `JIRA_DEFAULT_PROJECT` - Default Jira project key (default: BAP)
    *   For Jira integration:
        *   Ensure Jira MCP server is running on the configured URL
        *   Duplicate `mcp-atlassian.env` to `mcp-atlassian.env.local`, populate Jira MCP credentials, and keep the `.local` file private.
    *   **Environment Variables:**
        *   Copy `.env.example` to `.env` and fill in all required secrets (never commit `.env` to git).

6.  **Initialize Database:**
    The bot uses SQLite and will create database files automatically on first run.


7.  **Run the Bot:**
    ```bash
    python bot.py
    ```
    This will also attempt to start the G4F API server locally.

## 🚀 Quickstart

```bash
git clone https://github.com/amirhosseinzolfi/blue-ai-coacher-bot.git
cd blue-ai-coacher-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config.py.example config.py
# Edit config.py and .env as needed
python bot.py
```


## 📂 Project Structure

```
/root/blue-ai-coacher-bot/
├── bot.py                     # Main bot application, Telebot setup, G4F API server start
├── langgraph_code.py          # LangChain & LangGraph core logic, agent execution
├── graph_definition.py        # Defines the LangGraph agent structure
├── command_handlers.py        # Handles Telegram command inputs (e.g., /start, /settings)
├── callback_handlers.py       # Handles inline button callback queries
├── message_handlers.py        # Handles regular text, photo, and document messages
├── jira_integration.py        # Jira MCP integration and tool management
├── jira_agent.py              # Jira ReAct agent for task queries
├── config.py.example          # Example configuration file (copy to config.py)
├── daily_reset.py             # Handles daily reset tasks
├── db_manager.py              # SQLite database interaction logic
├── prompts/
│   └── prompts.py             # Contains all LLM prompt templates (including Jira prompts)
├── utils/
│   ├── helpers.py             # Utility functions (e.g., Markdown escaping)
│   ├── rich_logger.py         # Custom logging setup using Rich
│   ├── date_helpers.py        # Date utility functions
│   └── persian_date.py        # Persian date conversion utilities
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```


## 🤖 Usage

Interact with Blue (بلو) in a Telegram group or direct chat:

*   **Initial Setup:** Use `/settings` to register business information and set the AI's tone.
*   **Commands:**
    *   `/start`: Initiates interaction and shows a welcome message.
    *   `/help`: Displays available commands.
    *   `/settings`: Allows configuration of business info and AI tone.
    *   `/options`: Shows additional actions like daily tasks, Instagram ideas, reports.
    *   `/new_chat`: Starts a new conversation session, clearing previous context for some operations.
    *   `/about`: Shows information about the bot.
*   **Mentions:** In group chats, mention "بلو" or tag `@Blue` (replace with actual bot username if different) in your message to get a response.
*   **Inline Buttons:** Many commands and options present inline buttons for quick actions.
*   **Menu Buttons:** Predefined menu buttons for common actions like "➕ افزودن تسک" (Add Task), "📊 گزارش امروز" (Today's Report).

### Jira Integration Usage

When Jira integration is enabled, you can ask questions like:
- "تسکهای من در جیرا چیه؟" (What are my tasks in Jira?)
- "اسپرینت فعلی چه وضعیتی داره؟" (What's the current sprint status?)
- "show me BAP-123 issue details"
- "لیست مسائل باز" (List open issues)

The bot automatically detects Jira-related queries and uses MCP tools to fetch real-time data.


## 🔒 Security Notes

*   Never commit your `config.py` file with real API keys to version control.
*   Use environment variables in production environments.
*   The `.gitignore` file is configured to exclude sensitive files.
*   Keep `mcp-atlassian.env.local` private and never commit it.


## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a Pull Request.

Please ensure your code adheres to the project's coding style and includes relevant tests if applicable.


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
