# NBA Analytics Telegram Bot

## Setup

1. Clone repo
2. Copy `config.py.template` → `config.py`, then fill in API keys.
3. Install dependencies:
pip install -r requirements.txt

4. Run locally:
python bot.py

5. (Optional) Deploy via Heroku/Render:
- Push to GitHub
- Set config vars (`API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, etc.)
- Use `Procfile` to deploy

## Commands (/commands supported):
- `/scores`: fetch last 30 days of NBA games and save CSV
- `/reddit`: sentiment analysis on recent Reddit posts
- `/yt`: fetch NBA YouTube highlights
- `/ai`: GPT‑4 sentiment + theme summary

Enjoy your bot! 🏀
