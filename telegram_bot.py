

import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "861c2d372b8a409b9db663cb0bc08dbc"
BASE_URL = "https://api.sportsdata.io/v3/nba/scores/json/GamesByDateFinal/2025-01-31?"

def fetch_scores_for_date(date_str):
    url = f"{BASE_URL}/{date_str}"
    params = {"key": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {date_str}: {response.status_code}")
        return []

def get_last_30_days_dates():
    today = datetime.utcnow()
    dates = []
    for i in range(30):
        day = today - timedelta(days=i)
        # Format: 2025-APR-17
        date_str = day.strftime("%Y-%b-%d").upper()
        dates.append(date_str)
    return dates

def fetch_last_30_days_scores():
    all_games = []
    for date_str in get_last_30_days_dates():
        games = fetch_scores_for_date(date_str)
        if games:
            all_games.extend(games)
    return all_games

def save_to_csv(games, filename="nba_scores_last_30_days.csv"):
    if games:
        df = pd.DataFrame(games)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} games to {filename}")
    else:
        print("No game data to save.")

if __name__ == "__main__":
    games = fetch_last_30_days_scores()
    save_to_csv(games)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('nba_scores_last_30_days.csv')

# --- Feature Engineering ---
# Example: Use only games with valid scores
df = df[(df['HomeTeamScore'].notnull()) & (df['AwayTeamScore'].notnull())]

# Create target variable: 1 if home team won, 0 otherwise
df['HomeWin'] = (df['HomeTeamScore'] > df['AwayTeamScore']).astype(int)

# Example features (you can add more, e.g., team stats, recent records, etc.)
# Here, we use the difference in team scores as a simple feature
df['ScoreDiff'] = df['HomeTeamScore'] - df['AwayTeamScore']

# Possible features to use (feel free to expand)
features = [
    'HomeTeamScore',
    'AwayTeamScore',
    'ScoreDiff'
]

X = df[features]
y = df['HomeWin']
import train_test_split
# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train logistic regression model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Check model coefficients
coef_df = pd.DataFrame({'feature': features, 'coefficient': model.coef_[0]})
print("\nModel coefficients:\n", coef_df)

import requests
import pandas as pd

API_KEY = "861c2d372b8a409b9db663cb0bc08dbc"
BASE_URL = "https://api.sportsdata.io/v3/nba/scores/json/AllTeams"

def fetch_all_teams(api_key):
    params = {"key": api_key}
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def save_teams_to_csv(teams, filename="nba_teams.csv"):
    df = pd.DataFrame(teams)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} teams to {filename}")

if __name__ == "__main__":
    teams = fetch_all_teams(API_KEY)
    save_teams_to_csv(teams)
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

API_KEY = "861c2d372b8a409b9db663cb0bc08dbc"
SEASON = "2025"
ORIGINAL_CSV = "nba_scores_last_30_days.csv"

# 1. Fetch all NBA team profiles via API
teams_url = f"https://api.sportsdata.io/v3/nba/scores/json/AllTeams"
teams_params = {"key": API_KEY}
teams_response = requests.get(teams_url, params=teams_params)
teams_response.raise_for_status()
teams = teams_response.json()
teams_df = pd.DataFrame(teams)
teams_df = teams_df[teams_df["Active"] == True]
teams_df = teams_df[teams_df["Key"].str.len() <= 3]
teams_df["FullTeamName"] = teams_df["City"] + " " + teams_df["Name"]
teams_df.to_csv("nba_teams.csv", index=False)
print("Saved nba_teams.csv")

# 2. Fetch player season stats for each team and aggregate
stats_base_url = "https://api.sportsdata.io/v3/nba/stats/json/PlayerSeasonStatsByTeam"
team_stats_list = []

for idx, row in teams_df.iterrows():
    abbr = row["Key"]
    url = f"{stats_base_url}/{SEASON}/{abbr}"
    try:
        r = requests.get(url, params={"key": API_KEY})
        if r.status_code == 200:
            players = r.json()
            print(f"Fetching stats for team {abbr} got {len(players)} players.")
            if players:
                df = pd.DataFrame(players)
                agg = {
                    "TeamAbbr": abbr,
                    "AvgPoints": df["Points"].mean(),
                    "AvgAssists": df["Assists"].mean(),
                    "AvgRebounds": df["Rebounds"].mean(),
                    "AvgSteals": df["Steals"].mean(),
                    "AvgBlocks": df["Blocks"].mean(),
                    "AvgTurnovers": df["Turnovers"].mean(),
                }
                team_stats_list.append(agg)
        else:
            print(f"Failed to fetch stats for {abbr}: {r.status_code}")
    except Exception as e:
        print(f"Error for {abbr}: {e}")

print("team_stats_list:", team_stats_list)
if team_stats_list:
    print("First dict keys:", team_stats_list[0].keys())

team_stats_df = pd.DataFrame(team_stats_list)
print("team_stats_df columns:", team_stats_df.columns)
team_stats_df.to_csv("nba_team_season_stats.csv", index=False)
print("Saved nba_team_season_stats.csv")

# 3. Merge with original games CSV
games = pd.read_csv(ORIGINAL_CSV)
team_name_to_abbr = dict(zip(teams_df["FullTeamName"], teams_df["Key"]))

if "HomeTeam" in games.columns and "AwayTeam" in games.columns:
    games["HomeTeamAbbr"] = games["HomeTeam"].map(team_name_to_abbr)
    games["AwayTeamAbbr"] = games["AwayTeam"].map(team_name_to_abbr)
else:
    raise ValueError("Your games CSV must have HomeTeam and AwayTeam columns.")

# Only merge if 'TeamAbbr' exists!
if "TeamAbbr" in team_stats_df.columns:
    games = games.merge(
        team_stats_df, left_on="HomeTeamAbbr", right_on="TeamAbbr", how="left", suffixes=('', '_Home')
    )
    for col in ['AvgPoints', 'AvgAssists', 'AvgRebounds', 'AvgSteals', 'AvgBlocks', 'AvgTurnovers']:
        games.rename(columns={col: f'Home_{col}'}, inplace=True)
    games.drop(columns=['TeamAbbr'], inplace=True)

    games = games.merge(
        team_stats_df, left_on="AwayTeamAbbr", right_on="TeamAbbr", how="left", suffixes=('', '_Away')
    )
    for col in ['AvgPoints', 'AvgAssists', 'AvgRebounds', 'AvgSteals', 'AvgBlocks', 'AvgTurnovers']:
        games.rename(columns={col: f'Away_{col}'}, inplace=True)
    games.drop(columns=['TeamAbbr'], inplace=True)
else:
    print("No TeamAbbr column found in team_stats_df. Cannot merge team stats.")

games.to_csv("nba_scores_last_30_days_with_stats.csv", index=False)
print("Saved nba_scores_last_30_days_with_stats.csv")

# 4. Train logistic regression model
games = games[(games['HomeTeamScore'].notnull()) & (games['AwayTeamScore'].notnull())]
games['HomeWin'] = (games['HomeTeamScore'] > games['AwayTeamScore']).astype(int)
features = [
    'Home_AvgPoints', 'Home_AvgAssists', 'Home_AvgRebounds', 'Home_AvgSteals', 'Home_AvgBlocks', 'Home_AvgTurnovers',
    'Away_AvgPoints', 'Away_AvgAssists', 'Away_AvgRebounds', 'Away_AvgSteals', 'Away_AvgBlocks', 'Away_AvgTurnovers',
]
X = games[features].fillna(0)
y = games['HomeWin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

import requests

# Query parameters
query = "Oklahoma City Thunder"
limit = 10
url = f"https://api.pushshift.io/reddit/search/submission/?q={query}&size={limit}&sort=desc"

# Get data
response = requests.get(url)
data = response.json()

# Get posts
posts = data.get('data', [])

print(f"Showing {len(posts)} recent Reddit posts about '{query}':\n")
for i, post in enumerate(posts, 1):
    title = post.get('title', '[No Title]')
    subreddit = post.get('subreddit', '')
    link = post.get('full_link', '')
    print(f"{i}. {title} (r/{subreddit})")
    print(f"   {link}\n")

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Step 1: Get Reddit posts about Oklahoma City Thunder
query = "Oklahoma City Thunder"
limit = 10
url = f"https://api.pushshift.io/reddit/search/submission/?q={query}&size={limit}&sort=desc"
response = requests.get(url)
posts = response.json().get('data', [])

# Step 2: Analyze sentiment using VADER
analyzer = SentimentIntensityAnalyzer()
results = []
aggregate_score = 0

for post in posts:
    title = post.get('title', '[No Title]')
    sentiment = analyzer.polarity_scores(title)
    score = sentiment['compound']
    aggregate_score += score
    results.append({'title': title, 'score': score})

# Step 3: Print posts and their sentiment
print("10 Recent Reddit Posts about the Oklahoma City Thunder and their sentiment:")
for i, res in enumerate(results, 1):
    tag = "positive" if res['score'] > 0 else "negative" if res['score'] < 0 else "neutral"
    print(f"{i}. {res['title']} [{tag}, score={res['score']:.3f}]")

# Step 4: Aggregate sentiment
average_score = aggregate_score / len(results) if results else 0
if average_score > 0:
    summary = "positive"
elif average_score < 0:
    summary = "negative"
else:
    summary = "neutral"
print("\nAggregate sentiment:", summary)

import os
from googleapiclient.discovery import build
from pytube import YouTube
def main():
    api_key = 'AIzaSyAbTk3dh42inDAQZ7-OYsWXiMRfLcbCKmM'  # Your YouTube API key
    query = "NBA highlights"

    videos = get_youtube_videos(api_key, query)
    if videos:
        print("Latest NBA Highlights:")
        for i, video in enumerate(videos):
            print(f"{i + 1}. {video['title']} (https://www.youtube.com/watch?v={video['video_id']})")

        # Download the first video
        download_video(videos[0]['video_id'])
    else:
        print("No videos found.")

def download_video(video_id):
    url = f'https://www.youtube.com/watch?v={video_id}'
    yt = YouTube(url)

    if yt.streams.filter(file_extension='mp4'):
        stream = yt.streams.filter(file_extension='mp4').first()
        stream.download(filename=f"{video_id}.mp4")
        print(f"Downloaded: {video_id}.mp4")
    else:
        print("No MP4 stream found")


import requests
import openai

# --- CONFIGURATION ---
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

REDDIT_API_KEY = "yMDHFEMGGDJjcgiFgdOC3G4Ax6um-Q"  # Your Reddit API Key (if applicable)

# --- 1. Fetch Reddit posts about Oklahoma City Thunder ---
query = "Oklahoma City Thunder"
limit = 10
url = f"https://api.pushshift.io/reddit/search/submission/?q={query}&size={limit}&sort=desc"

# Optional: Include the Reddit API key in the headers if the endpoint requires it
headers = {
    "Authorization": f"Bearer {REDDIT_API_KEY}"
}

response = requests.get(url, headers=headers)

# Print the response for debugging
print(response.json())  # Added for debugging purposes

posts = response.json().get("data", [])

# --- 2. Prepare the content for GPT-4 ---
if not posts:
    print("No Reddit posts found. Please try a different query or check the API.")
else:
    post_texts = []
    for post in posts:
        title = post.get("title", "[No Title]")
        selftext = post.get("selftext", "")
        # Use title + selftext if available, else just title
        if selftext and selftext.lower() != "[removed]":
            post_texts.append(f"Title: {title}\nText: {selftext}")
        else:
            post_texts.append(f"Title: {title}")

    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Analyze the following Reddit posts about the Oklahoma City Thunder. Summarize the overall sentiment in one word: positive, negative, or neutral. Also briefly summarize the main themes of the posts."
        },
        {
            "role": "user",
            "content": "\n\n".join(post_texts)
        }
    ]

    # --- 3. Send to GPT-4 for sentiment and summary ---
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation,
        max_tokens=300,
        temperature=0.2
    )

    reply = response.choices[0].message.content

    print("\nGPT-4 Analysis and Sentiment Summary:")
    print(reply)

import os
import requests
import pandas as pd
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai

# --- CONFIGURATION ---
OPENAI_API_KEY = "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"
API_KEY = "861c2d372b8a409b9db663cb0bc08dbc"  # SportsData API Key
TELEGRAM_BOT_TOKEN = "7972586424:AAH7YaQ2c2K8GhD_FThpYku3VjmcX_mZkJI"  # Telegram Bot Token

openai.api_key = OPENAI_API_KEY

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Hello! I am your NBA stats bot. Use /fetch_scores to get NBA scores.")

def fetch_scores(update: Update, context: CallbackContext) -> None:
    # Implement your logic to fetch NBA scores and handle exceptions
    try:
        all_games = fetch_last_30_days_scores()  # Define this function as originally done
        save_to_csv(all_games, "nba_scores_last_30_days.csv")
        update.message.reply_text("NBA scores fetched successfully! Check the CSV.")
    except Exception as e:
        update.message.reply_text(f"Error fetching scores: {e}")

def fetch_reddit_posts(update: Update, context: CallbackContext) -> None:
    query = "Oklahoma City Thunder"
    url = f"https://api.pushshift.io/reddit/search/submission/?q={query}&size=10&sort=desc"
    response = requests.get(url)

    if response.status_code != 200:
        update.message.reply_text("Failed to fetch Reddit posts.")
        return

    posts = response.json().get("data", [])
    analyzer = SentimentIntensityAnalyzer()
    results = []

    for post in posts:
        title = post.get('title', '[No Title]')
        sentiment = analyzer.polarity_scores(title)
        results.append({'title': title, 'score': sentiment['compound']})

    summary = analyze_sentiments(results)  # Define this function as per your logic

    response_text = "\n".join([f"{res['title']} (score={res['score']})" for res in results])
    response_text += f"\nAggregate sentiment: {summary}"

    update.message.reply_text(response_text)

def analyze_sentiments(results):
    aggregate_score = sum(res['score'] for res in results)
    average_score = aggregate_score / len(results) if results else 0
    if average_score > 0:
        return "positive"
    elif average_score < 0:
        return "negative"
    else:
        return "neutral"

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("fetch_scores", fetch_scores))
    dispatcher.add_handler(CommandHandler("fetch_reddit_posts", fetch_reddit_posts))

    updater.start_polling()
    updater.idle()
