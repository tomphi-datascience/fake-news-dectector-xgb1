import streamlit as st
import joblib
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# === Load trained model ===
model = joblib.load("fakenews_model.pkl")

# === Load TF-IDF configuration files ===
with open("tfidf_params.json") as f:
    tfidf_params = json.load(f)

with open("tfidf_vocab.json") as f:
    tfidf_vocab = json.load(f)

with open("tfidf_idf.json") as f:
    idf_values = json.load(f)

with open("tfidf_stopwords.json") as f:
    stop_words = json.load(f)

# === Rebuild TfidfVectorizer and restore internals ===
tfidf = TfidfVectorizer(**tfidf_params)
tfidf.vocabulary_ = tfidf_vocab
tfidf._tfidf.idf_ = np.array(idf_values)
tfidf.stop_words_ = set(stop_words)

# === Streamlit UI ===
st.set_page_config(page_title="Fake News Classifier", layout="centered")
st.title("📰 Fake News Classifier")
st.write("Paste a news article below and find out whether it's **real** or **fake**.")

# === Text cleaning ===
def clean_text_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# === User input ===
