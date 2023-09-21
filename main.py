import requests
from bs4 import BeautifulSoup
from transformers import pipeline

class NewsAggregator:
    def __init__(self):
        self.news_sources = [
            "https://www.bbc.com/news/",
            "https://www.cnn.com/",
            "https://www.nytimes.com/",
        ]

    def scrape_articles(self):
        articles = []
        for source in self.news_sources:
            try:
                response = requests.get(source)
                soup = BeautifulSoup(response.text, "html.parser")
                articles.extend(soup.find_all("article"))
            except requests.exceptions.RequestException as e:
                print(f"Error scraping data from {source}: {e}")
        return articles


class NewsSummarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization")

    def generate_summary(self, article):
        text = article.text.strip()
        summary = self.summarizer(text, max_length=100, min_length=50, do_sample=False)[0]['summary_text']
        return summary


class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)[0]
        return result["label"]


class NewsApp:
    def __init__(self):
        self.news_aggregator = NewsAggregator()
        self.news_summarizer = NewsSummarizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_articles = []
        self.user_preferences = []

    def run(self):
        self.news_articles = self.news_aggregator.scrape_articles()
        self.show_news_feed()

    def show_news_feed(self):
        for article in self.news_articles:
            summary = self.news_summarizer.generate_summary(article)
            sentiment = self.sentiment_analyzer.analyze_sentiment(summary)

            if self.user_preferences:
                if not any(pref.lower() in summary.lower() for pref in self.user_preferences):
                    continue

            print("Headline:", article.get("headline"))
            print("Summary:", summary)
            print("Sentiment:", sentiment)
            print("-" * 50)


if __name__ == "__main__":
    news_app = NewsApp()
    news_app.run()