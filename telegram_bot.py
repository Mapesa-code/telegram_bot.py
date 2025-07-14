from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from nba.fetch_scores import fetch_last_30_days_scores
from utils.config import TELEGRAM_BOT_TOKEN

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello! I'm your NBA bot. Use /fetch_scores to get NBA results.")

def fetch_scores(update: Update, context: CallbackContext):
    try:
        games = fetch_last_30_days_scores()
        update.message.reply_text(f"Fetched {len(games)} NBA games.")
    except Exception as e:
        update.message.reply_text(f"Error: {str(e)}")

def run_bot():
    updater = Updater(TELEGRAM_BOT_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("fetch_scores", fetch_scores))
    updater.start_polling()
    updater.idle()
