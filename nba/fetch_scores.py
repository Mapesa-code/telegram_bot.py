import requests
from datetime import datetime, timedelta
from utils.config import SPORTS_API_KEY

BASE_URL = "https://api.sportsdata.io/v3/nba/scores/json/GamesByDate"

def get_last_30_days_dates():
    today = datetime.utcnow()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]

def fetch_scores_for_date(date_str):
    url = f"{BASE_URL}/{date_str}"
    response = requests.get(url, params={"key": SPORTS_API_KEY})
    if response.status_code == 200:
        return response.json()
    print(f"Failed for {date_str}: {response.status_code}")
    return []

def fetch_last_30_days_scores():
    all_games = []
    for date in get_last_30_days_dates():
        all_games.extend(fetch_scores_for_date(date))
    return all_games
