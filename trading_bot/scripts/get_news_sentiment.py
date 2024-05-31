import requests
from textblob import TextBlob

def get_news_sentiment(keyword):
    url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey=487753535a0b4309942040c85e1b7886"
    response = requests.get(url)
    articles = response.json()['articles']
    
    sentiments = []
    for article in articles:
        analysis = TextBlob(article['description'])
        sentiments.append(analysis.sentiment.polarity)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment

if __name__ == "__main__":
    bitcoin_sentiment = get_news_sentiment("bitcoin")
    print(f"Bitcoin Sentiment: {bitcoin_sentiment}")