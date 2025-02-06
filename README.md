# Blue AI Coacher Bot

Blue AI Coacher Bot is a Telegram chatbot that acts as your friendly, expert business coach. Powered by OpenAI’s language models and LangChain, the bot provides personalized coaching in Persian, manages conversation history, and offers configurable settings for business information and AI tone. It also comes with interactive options to generate daily reports, analyze member activities, and list daily tasks.

## Features

- **Personalized Business Coaching:**  
  Acts as “Blue,” a professional business coach who responds exclusively in Persian.
  
- **Chat History Management:**  
  Utilizes SQLite and LangChain’s SQLChatMessageHistory to load, store, and manage conversation history.
  
- **Customizable Settings:**  
  - **Business Information:** Configure and save details about your business and team.  
  - **AI Tone:** Choose from three tones—**رسمی**, **دوستانه** (default), and **حرفه ای**.
  
- **Interactive Options:**  
  Access commands via inline keyboards:
  - **گزارش روزانه:** Get a complete report of the chat.
  - **آنالیز امروز اعضا:** Receive an analysis report on member activities.
  - **تسک های امروز:** List individual tasks based on the chat conversation.
  
- **Detailed Logging:**  
  Extensive logging at every step—session IDs, chat history counts, user prompts, and AI responses—to aid in debugging and monitoring.

- **Deployment Ready:**  
  Easily deployable to cloud platforms (e.g., Vercel). *(Note: For persistent processes, consider using webhooks instead of polling.)*

## Project Structure

```

blue-ai-coacher-bot/ ├── bot.py # Main Python file containing the Telegram bot code ├── requirements.txt # Python dependencies ├── vercel.json # Vercel deployment configuration └── README.md # Project documentation (this file)

```

## Prerequisites

- **Python 3.7+**  
- **Telegram Bot Token:** Obtain from [BotFather](https://core.telegram.org/bots#6-botfather).
- **OpenAI API Key:** Required for using OpenAI’s models.
- **Google API Key:** (If applicable for your project)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/amirhosseinzolfi/blue-ai-coacher-bot.git
   cd blue-ai-coacher-bot
```

2. **Create a Virtual Environment (Recommended):**
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
3. **Install Dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
    

## Configuration

### Environment Variables

Create a `.env` file in your project root (or set these variables in your deployment environment) with the following keys:

```
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
```

Alternatively, configure these environment variables in your hosting platform (e.g., Vercel).

### Vercel Deployment Configuration

The included `vercel.json` file tells Vercel how to build and deploy the project:

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

## Running Locally

To run the bot on your local machine:

```bash
python bot.py
```

> **Note:** The bot currently uses `bot.polling(none_stop=True)`. For production deployments, consider switching to Telegram webhooks to better suit serverless environments.

## Deployment

### Deploying on Vercel

4. **Push Your Code to GitHub:**  
    Ensure your repository is up-to-date at [github.com/amirhosseinzolfi/blue-ai-coacher-bot](https://github.com/amirhosseinzolfi/blue-ai-coacher-bot).
    
5. **Import the Repository into Vercel:**
    
    - Log in to [Vercel](https://vercel.com/) and click **"New Project"**.
    - Select the **blue-ai-coacher-bot** repository from your GitHub account.
    - Vercel will auto-detect the `vercel.json` file and set up the Python builder.
6. **Configure Environment Variables on Vercel:**  
    In your project settings on Vercel, add the required environment variables (`TELEGRAM_BOT_TOKEN`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, etc.) and trigger a redeployment.
    
7. **Deploy:**  
    Click the **"Deploy"** button and monitor the logs. Once deployed, you will receive a URL for your project.
    

### Alternative Hosting Options

If you encounter issues with Vercel’s serverless functions (due to the long-running nature of polling), consider deploying on platforms that support persistent processes (e.g., Heroku, DigitalOcean, or AWS EC2).

## Usage

Interact with the bot via Telegram:

- **Commands:**
    
    - `/start` — Initialize the bot and display chat details.
    - `/help` — Show available commands.
    - `/getchatid` — Retrieve current chat ID and type.
    - `/show_history` — Generate a summary of the conversation.
    - `/refresh_history` — Clear and start a new chat session.
    - `/show_sessions` — List all saved sessions.
    - `/about` — Information about the bot.
    - `/settings` — Configure business info and AI tone.
    - `/options` — Access interactive options (daily report, member analysis, tasks).
- **Interactive Menus:**  
    Inline keyboards guide you through setting business information and AI tone, and let you choose from several pre-defined report options.
    

## Logging and Debugging

The bot prints detailed logs to the terminal, including:

- Session IDs and chat history counts.
- User prompts and corresponding AI responses.
- Status messages for each step in command and callback processing.

These logs help in troubleshooting and monitoring the internal processes.

## Contributing

Contributions, issues, and feature requests are welcome!  
Please check the [issues](https://github.com/amirhosseinzolfi/blue-ai-coacher-bot/issues) section or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://chatgpt.com/c/LICENSE) file for details.

## Acknowledgments

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [LangChain](https://langchain.readthedocs.io/)
- [OpenAI](https://openai.com/)
- Many thanks to the contributors and open-source community for their invaluable support.

