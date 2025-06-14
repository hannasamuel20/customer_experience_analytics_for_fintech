from textblob import TextBlob
import pandas as pd
from langdetect import detect
import emoji
import matplotlib.pyplot as plt
import re
import seaborn as sns

from transformers import pipeline
from tqdm import tqdm
tqdm.pandas() 


# Load sentiment-analysis pipeline with DistilBERT



class SentimentAnalysis:
    def __init__(self,df):
        self.df = df
        self.convert_datetime()
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        # self.convert_datetime()

    
    def remove_amharic_rows(self):
        self.df = self.df[self.df['language'] == 'en']


    def emoji_to_text(self):
        def convert_emoji_to_text(text):
            return emoji.demojize(text, language='en')
        self.df['review_clean'] = self.df['review_text'].apply(convert_emoji_to_text)

        
    def convert_datetime(self):
        self.df['date'] = pd.to_datetime(self.df['date'])

    def sentiment_analysis_text_blob(self):
        def get_sentiment(text):
            polarity = TextBlob(text).sentiment.polarity
            return polarity

        # Apply the function to the headline column
        self.df['sentiment'] = self.df['review_text'].apply(get_sentiment)
    
    def sentiment_analysis_distilbert(self):
        
        def get_sentiment_label_score(text):
            try:
                result = self.sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens
                return pd.Series([result['label'], result['score']])
            except:
                return pd.Series([None, None])

        # Apply to all reviews
        self.df[['sentiment_label', 'sentiment_score']] = self.df['review_text'].progress_apply(get_sentiment_label_score)
        # Map labels to numeric values
        self.df['sentiment_value'] = self.df['sentiment_label'].map({'POSITIVE': 1, 'NEGATIVE': -1})

        # Multiply to get scaled sentiment score
        self.df['scaled_sentiment'] = self.df['sentiment_value'] * self.df['sentiment_score']


    def plot_sentiment_by_rating(self):
        banks = self.df['bank_name'].unique()

        
        for bank in banks:
            bank_df = self.df[self.df['bank_name'] == bank]
            mean_sentiment_per_rating = bank_df.groupby('rating')['scaled_sentiment'].mean().reset_index()

            plt.figure(figsize=(8, 5))
            plt.bar(mean_sentiment_per_rating['rating'], mean_sentiment_per_rating['scaled_sentiment'], color='skyblue')
            plt.xlabel('Rating')
            plt.ylabel('Average Sentiment Score')
            plt.title(f'Average Sentiment Score per Rating - {bank}')
            plt.xticks(mean_sentiment_per_rating['rating'])  # Show all rating values
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def plot_sentiment_trend(self):
        self.df['date_month'] = self.df['date'].dt.to_period('M').dt.to_timestamp()

        banks = self.df['bank_name'].unique()

        # Create subplots for sentiment trends
        fig, axes = plt.subplots(1, len(banks), figsize=(18, 5), sharey=True)
        fig.suptitle('Sentiment Trends Over Time', fontsize=16)

        for i, bank in enumerate(banks):
            bank_data = self.df[self.df['bank_name'] == bank]
            sentiment_trend = (
                bank_data.groupby('date_month')['scaled_sentiment']
                .mean()
                .reset_index()
            )
            axes[i].plot(sentiment_trend['date_month'], sentiment_trend['scaled_sentiment'], marker='o', color='teal')
            axes[i].set_title(f'{bank}')
            axes[i].set_xlabel('Month')
            axes[i].tick_params(axis='x', rotation=45)

        axes[0].set_ylabel('Average Sentiment Score')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    def plot_rating_distribution_plot(self):
        banks = self.df['bank_name'].unique()
        fig, axes = plt.subplots(1, len(banks), figsize=(18, 5), sharey=True)
        fig.suptitle('Rating Distribution per Bank', fontsize=16)

        for i, bank in enumerate(banks):
            bank_data = self.df[self.df['bank_name'] == bank]
            sns.countplot(data=bank_data, x='rating', ax=axes[i], palette='Blues')
            axes[i].set_title(f'{bank}')
            axes[i].set_xlabel('Rating')

        axes[0].set_ylabel('Number of Reviews')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


