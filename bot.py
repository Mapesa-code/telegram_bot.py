from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from config import TELEGRAM_BOT_TOKEN
from scores import fetch_last_30_days_scores, save_to_csv
from reddit import fetch_reddit_posts, analyze_sentiments

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Welcome to the NBA Bot! Use /fetch_scores or /reddit_sentiment.")

def fetch_scores(update: Update, context: CallbackContext):
    games = fetch_last_30_days_scores()
    save_to_csv(games)
    update.message.reply_text("Scores saved to CSV.")

def reddit_sentiment(update: Update, context: CallbackContext):
    posts = fetch_reddit_posts()
    results, sentiment = analyze_sentiments(posts)
    msg = "\n".join(f"{r['title']} (score={r['score']:.2f})" for r in results)
    msg += f"\n\nOverall sentiment: {sentiment}"
    update.message.reply_text(msg[:4096])  # Telegram msg limit

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("fetch_scores", fetch_scores))
    dp.add_handler(CommandHandler("reddit_sentiment", reddit_sentiment))
    updater.start_polling()
    updater.idle()
if __name__ == "__main__":
    main()

