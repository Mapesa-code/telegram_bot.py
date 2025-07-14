import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_reddit_posts(query="Oklahoma City Thunder", limit=10):
    url = f"https://api.pushshift.io/reddit/search/submission/?q={query}&size={limit}&sort=desc"
    response = requests.get(url)
    return response.json().get("data", [])

def analyze_sentiments(posts):
    analyzer = SentimentIntensityAnalyzer()
    results = [{"title": p.get("title", "[No Title]"), "score": analyzer.polarity_scores(p.get("title", ""))["compound"]} for p in posts]
    avg_score = sum(r["score"] for r in results) / len(results)
    summary = "positive" if avg_score > 0 else "negative" if avg_score < 0 else "neutral"
    return results, summary
