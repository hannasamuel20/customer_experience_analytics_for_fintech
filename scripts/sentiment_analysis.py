from textblob import TextBlob
import pandas as pd
from langdetect import detect
import emoji
import matplotlib.pyplot as plt
import spacy
import re

# Load English language model
nlp = spacy.load("en_core_web_sm")
THEME_KEYWORDS = {
    "Account Access Issues": ["login", "password", "account lock", "access denied"],
    "Transaction Performance": ["transfer slow", "transaction failed", "delay"],
    "User Interface": ["app crash", "interface", "navigation", "design"],
    "Customer Support": ["support", "agent", "response time", "service"],
    "Feature Requests": ["feature", "request", "add", "should have"]
}

class SentimentAnalysis:
    def __init__(self,df):
        self.df = df
        # self.convert_datetime()

    
    def remove_amharic_rows(self):
        self.df = self.df[self.df['language'] == 'en']
    
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop 
            and not token.is_punct 
            and token.pos_ in ["NOUN", "VERB", "ADJ"]
        ]
        return " ".join(tokens)
    
    # Keyword extraction function
    def extract_keywords(text):
        doc = nlp(text)
        keywords = [
            chunk.text.lower() 
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) <= 3  # Include 1-3 word phrases
        ]
        return list(set(keywords))
    
    def apply_preprocessing(self):
        self.df['processed_review'] = self.df['review'].astype(str).apply(preprocess_text)
        
    def apply_keyword_extraction(self):
        self.df["keywords"] = self.df["review_text"].apply(extract_keywords)

    def emoji_to_text(self):
        def convert_emoji_to_text(text):
            return emoji.demojize(text, language='en')
        self.df['review_clean'] = self.df['review_text'].apply(convert_emoji_to_text)

        
    def convert_datetime(self):
        self.df['date'] = pd.to_datetime(self.df['date'])

    def sentiment_analysis(self):
        def get_sentiment(text):
            polarity = TextBlob(text).sentiment.polarity
            return polarity

        # Apply the function to the headline column
        self.df['sentiment'] = self.df['review_text'].apply(get_sentiment)

    def plot_sentiment_by_rating(self):
        # Group by rating and calculate mean sentiment
        mean_sentiment_per_rating = self.df.groupby('rating')['sentiment'].mean().reset_index()
        plt.figure(figsize=(8, 5))
        plt.bar(mean_sentiment_per_rating['rating'], mean_sentiment_per_rating['sentiment'], color='skyblue')
        plt.xlabel('Rating')
        plt.ylabel('Average Sentiment Score')
        plt.title('Average Sentiment Score per Rating')
        plt.xticks(mean_sentiment_per_rating['rating'])  # Show all rating values
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
