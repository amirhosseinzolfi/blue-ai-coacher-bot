
# Blue AI Coacher Bot

Blue AI Coacher Bot is a powerful Telegram chatbot that serves as your friendly and expert business coach. Using OpenAI’s language models integrated through LangChain, this bot delivers personalized coaching exclusively in Persian. It manages chat history via an SQLite database, supports customizable settings (like business information and AI tone), and offers interactive commands for generating daily reports, analyzing member activities, and listing tasks.

> **Note:**  
> The bot runs using continuous polling and the G4F Interference API. For production deployments on serverless platforms (e.g., Vercel), consider switching to a webhook-based architecture.

---

## Features

- **Personalized Business Coaching:**  
  Acts as “Blue,” a professional, knowledgeable, and friendly business coach who responds only in Persian.

- **Chat History Management:**  
  Stores and retrieves conversation history using SQLite and LangChain’s SQLChatMessageHistory.

- **Customizable Settings:**  
  - **Business Information:** Easily configure and update details about your business and team.  
  - **AI Tone:** Choose from three tones—**رسمی**, **دوستانه** (default), and **حرفه‌ای**—to match your desired style.

- **Interactive Options:**  
  Access interactive commands via inline keyboards:
  - **گزارش روزانه:** Get a complete daily report of the chat.
  - **آنالیز امروز اعضا:** Receive an analysis report on group member activities.
  - **تسک‌های امروز:** Generate a list of individual tasks based on the conversation.

- **Detailed Logging & Debugging:**  
  Comprehensive logs (session IDs, history counts, user prompts, and AI responses) help you monitor the bot’s performance and troubleshoot issues.

- **Deployment Ready:**  
  Easily deployable to cloud platforms like Vercel (with minimal adjustments). A minimal `requirements.txt` keeps the package size within limits.

---

## Project Structure

```
blue-ai-coacher-bot/
├── bot.py           # Main bot code containing all the logic
├── requirements.txt # List of required Python libraries (minimal)
├── vercel.json      # Vercel deployment configuration file
├── .gitignore       # Git ignore file to exclude virtual environment and cache files
└── README.md        # This file – comprehensive project documentation
```

---

## Prerequisites

- **Python 3.7+**  
- **Telegram Bot Token:**  
  Obtain your token from [BotFather](https://core.telegram.org/bots#6-botfather).
- **OpenAI API Key:**  
  Required for using OpenAI models.
- **Google API Key:**  
  If applicable to your project.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/amirhosseinzolfi/blue-ai-coacher-bot.git
cd blue-ai-coacher-bot
```

### 2. Create and Activate a Virtual Environment (Recommended)

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (or set these variables in your deployment platform) with the following keys:

```env
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
```

> **Important:** Do **not** commit your `.env` file. Make sure it is added to your `.gitignore`.

---

## Running Locally

To start the bot locally:

```bash
python bot.py
```

You should see log messages indicating that the G4F Interference API server, SQLite database, and Telegram bot have been initialized.

---

## Deployment on Vercel

Blue AI Coacher Bot can be deployed on Vercel with a minimal configuration.

### 1. Create a `vercel.json` File

Ensure that your `vercel.json` (in the project root) contains:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "bot.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "bot.py"
    }
  ]
}
```

### 2. Import the Repository into Vercel

- Log in to [Vercel](https://vercel.com) and click **"New Project"**.
- Select your GitHub repository: `amirhosseinzolfi/blue-ai-coacher-bot`.
- Choose **Other** as the framework preset.
- Set the **Root Directory** to `./`.

### 3. Add Environment Variables

In your Vercel project dashboard (Settings > Environment Variables), add:

- `TELEGRAM_BOT_TOKEN`
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`

### 4. Deploy

Click **"Deploy"** and monitor the build logs. Once deployed, Vercel will provide you with a URL for your project.

> **Note:**  
> Vercel’s serverless functions are meant for short-lived tasks. Since this bot uses continuous polling, you may need to refactor to use webhooks for a more production-ready deployment.

---

## Minimal Requirements

Below is an example of a minimal `requirements.txt` to keep your deployment package lean:

```
pyTelegramBotAPI==4.26.0
SQLAlchemy==2.0.37
langchain-openai==0.3.3
langchain-core==0.3.33
langchain-community==0.3.16
g4f[all]==0.4.5.4
```

Adjust version numbers as necessary. Exclude any packages that are not directly used by your bot.

---

## Usage

Once the bot is running (locally or deployed):

- **Interact via Telegram:**  
  - `/start` — Display bot welcome message and chat details.
  - `/help` — Show all available commands.
  - `/getchatid` — Retrieve the current chat’s ID and type.
  - `/show_history` — Generate a full report of the current chat history.
  - `/refresh_history` — Clear the current session’s history and start anew.
  - `/show_sessions` — List all stored chat sessions.
  - `/about` — Learn more about the bot.
  - `/settings` — Configure business information and AI tone.
  - `/options` — Access interactive options (daily report, member analysis, tasks).

- **Interactive Menus:**  
  Inline keyboards allow you to quickly set business information and adjust the AI tone.

---

## Logging & Debugging

The bot produces detailed logs in the terminal, including:

- **Initialization:**  
  Confirmation of API server start, SQLite database setup, and bot initialization.
- **Command Processing:**  
  Session IDs, number of messages loaded, user prompts, and AI responses.
- **Error Handling:**  
  Clear error messages for any issues encountered during execution.

These logs are invaluable for troubleshooting and monitoring the bot’s performance.

---

## Contributing

Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request on [GitHub Issues](https://github.com/amirhosseinzolfi/blue-ai-coacher-bot/issues).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
