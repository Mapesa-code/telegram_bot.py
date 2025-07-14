from googleapiclient.discovery import build

def get_youtube_videos(api_key, query="NBA highlights", max_results=5):
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.search().list(q=query, part="snippet", maxResults=max_results, type="video").execute()
    return [{"title": item["snippet"]["title"], "video_id": item["id"]["videoId"]} for item in response["items"]]
