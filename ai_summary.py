import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def summarize_posts_with_gpt(posts):
    texts = [f"Title: {p.get('title', '[No Title]')}\nText: {p.get('selftext', '')}" for p in posts]
    chat = [
        {"role": "system", "content": "Summarize overall sentiment (positive, negative, neutral) and themes."},
        {"role": "user", "content": "\n\n".join(texts)}
    ]
    res = openai.ChatCompletion.create(model="gpt-4", messages=chat)
    return res.choices[0].message.content
