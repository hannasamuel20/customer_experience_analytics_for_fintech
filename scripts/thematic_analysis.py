
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load English language model

theme_keywords = {
    "Account Access Issues": ["login", "logout", "access", "password", "signin", "crash","password", "account lock", "access denied"],
    "Transaction Performance": ["transfer", "delay", "slow", "load", "transaction", "network","transaction failed"],
    "User Interface & Experience": ["interface", "ui", "design", "easy", "simple", "navigate"],
    "Customer Support": ["support", "help", "respond", "contact", "service"],
    "Feature Requests": ["feature", "add", "option", "notification", "update", "function"],
    "General Satisfaction": ["great", "nice", "good", "love", "amazing", "awesome", "perfect", "excellent", "satisfied"]
}



class ThematicAnalysis:
    def __init__(self,df):
        self.df = df
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self,text):
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        return tokens

    # Keyword extraction function
    def extract_keywords(self,text):
        doc = self.nlp(text)
        keywords = [
            chunk.text.lower() 
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) <= 3  # Include 1-3 word phrases
        ]
        return list(set(keywords))
    
    def assign_themes(self,tokens):
        matched_themes = set()
        for theme, keywords in theme_keywords.items():
            if any(token in keywords for token in tokens):
                matched_themes.add(theme)
        return list(matched_themes) if matched_themes else ["Uncategorized"]
    
    
    def apply_preprocessing(self):
        self.df['tokens'] = self.df['review_text'].astype(str).apply(self.preprocess_text)
        
    def apply_keyword_extraction(self):
        self.df["keywords"] = self.df["review_text"].apply(self.extract_keywords)

    def apply_theme_assignment(self):
        self.df['themes'] = self.df['tokens'].apply(self.assign_themes)
    
    def save_as_csv(self):
        self.df[['review_text','rating', 'sentiment_label', 'sentiment_score', 'themes','sentiment','scaled_sentiment']].to_csv("../data/thematic_analysis_results.csv", index=False)

    def plot_themes_frequency_per_bank(self):

        # Explode theme list so each row has one theme
        df_exploded = self.df.explode('themes')

        # Get list of unique banks
        banks = df_exploded['bank_name'].unique()

        # Set up subplots: 1 row, len(banks) columns
        fig, axes = plt.subplots(1, len(banks), figsize=(6 * len(banks), 6), sharey=True)

        for i, bank in enumerate(banks):
            bank_data = df_exploded[df_exploded['bank_name'] == bank]
            sns.countplot(ax=axes[i], data=bank_data, x='themes',
                        order=bank_data['themes'].value_counts().index,
                        palette='pastel')
            axes[i].set_title(f'Theme Frequency - {bank}')
            axes[i].set_xlabel('Theme')
            axes[i].set_ylabel('Number of Reviews' if i == 0 else '')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_heatmap(self):
        df_exploded = self.df.explode('themes')
        theme_counts = df_exploded.groupby(['bank_name', 'themes']).size().unstack(fill_value=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(theme_counts, annot=True, fmt='d', cmap='Blues')
        plt.title('Review Theme Distribution per Bank')
        plt.xlabel('Theme')
        plt.ylabel('Bank')
        plt.tight_layout()
        plt.show()

    def plot_stacked_chart(self):
        df_exploded = self.df.explode('themes')
        theme_dist = df_exploded.groupby(['bank_name', 'themes']).size().unstack(fill_value=0)
        theme_dist.T.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Stacked Theme Distribution per Bank')
        plt.xlabel('Theme')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.legend(title='Bank', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

