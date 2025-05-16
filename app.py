import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import re
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

nltk.download('vader_lexicon')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    filtered = [word for word in tokens if word not in stopwords]
    return ' '.join(filtered)

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        overall = 'Positive 😊'
    elif compound <= -0.05:
        overall = 'Negative 😞'
    else:
        overall = 'Neutral 😐'

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return {
        'overall': overall,
        'vader_scores': scores,
        'textblob_polarity': polarity,
        'textblob_subjectivity': subjectivity
    }

def sentence_level_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentences = text.split('.')
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            score = sia.polarity_scores(sentence)['compound']
            if score >= 0.05:
                sentiment = 'Positive 😊'
            elif score <= -0.05:
                sentiment = 'Negative 😞'
            else:
                sentiment = 'Neutral 😐'
            results.append((sentence, sentiment, score))
    return results

st.title("Advanced Sentiment Analysis")

user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze") and user_input.strip():
    processed = preprocess_text(user_input)
    st.write(f"**Processed Text:** {processed}")

    result = analyze_sentiment(user_input)
    st.write(f"**Overall Sentiment:** {result['overall']}")
    st.write(f"VADER Scores: {result['vader_scores']}")
    st.write(f"TextBlob Polarity: {result['textblob_polarity']:.2f}")
    st.write(f"TextBlob Subjectivity: {result['textblob_subjectivity']:.2f}")

    st.write("### Sentence-Level Sentiment")
    sentence_results = sentence_level_sentiment(user_input)
    for sent, sent_label, score in sentence_results:
        st.write(f"{sent_label} ({score:.2f}): {sent}")

    # Bar plot for VADER scores
    scores = result['vader_scores']
    categories = ['Positive', 'Neutral', 'Negative']
    values = [scores['pos'], scores['neu'], scores['neg']]
    fig, ax = plt.subplots()
    ax.bar(categories, values, color=['green', 'gray', 'red'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Prepare CSV data
    data = {
        "Original Text": [user_input],
        "Processed Text": [processed],
        "Overall Sentiment": [result['overall']],
        "VADER Positive": [scores['pos']],
        "VADER Neutral": [scores['neu']],
        "VADER Negative": [scores['neg']],
        "TextBlob Polarity": [result['textblob_polarity']],
        "TextBlob Subjectivity": [result['textblob_subjectivity']],
    }
    df = pd.DataFrame(data)

    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Analysis Report as CSV",
        data=csv,
        file_name='sentiment_analysis_report.csv',
        mime='text/csv'
    )
