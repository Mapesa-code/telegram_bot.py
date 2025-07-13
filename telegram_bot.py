import os
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)
from src.model import train_model, make_prediction
from src.sentiment import get_team_sentiment
from src.youtube import get_highlight_video
from src.utils import load_team_stats

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Load model and stats
model, label_encoder = train_model()
team_stats = load_team_stats()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the NBA Bot! Use /predict <home_team> vs <away_team>")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = " ".join(context.args)
        if " vs " not in text:
            await update.message.reply_text("Use the format: /predict <home_team> vs <away_team>")
            return

        home_team, away_team = map(str.strip, text.split("vs"))
        prediction, prob = make_prediction(model, label_encoder, home_team, away_team, team_stats)
        sentiment = get_team_sentiment(home_team, away_team)
        video = get_highlight_video(home_team, away_team)

        response = (
            f"🏀 Prediction: {home_team} vs {away_team}\n"
            f"✅ Winner: {prediction} (Confidence: {prob:.2f})\n\n"
            f"🧠 Reddit Sentiment:\n{sentiment}\n\n"
            f"🎥 Highlights: {video}"
        )
        await update.message.reply_text(response)

    except Exception as e:
        logging.error(str(e))
        await update.message.reply_text("There was an error processing your request.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Use /predict <home_team> vs <away_team> to get predictions, sentiment, and highlights.")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("help", help_command))

    app.run_polling()
