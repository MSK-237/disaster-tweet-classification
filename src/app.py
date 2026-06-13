# Streamlit uygulaması

import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

st.set_page_config(
    page_title="Disaster Tweet Classification",
    page_icon="🚨",
    layout="centered"
)

stop_words = set(stopwords.words("english"))

def temizle(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&amp;", " ", text)
    text = re.sub(r"rt", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    kelimeler = [
        kelime for kelime in text.split()
        if kelime not in stop_words
    ]

    return " ".join(kelimeler)

model = joblib.load("disaster_tweet_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("🚨 Afet Tweet Sınıflandırma Uygulaması")

st.write(
    "Bu uygulama, girilen bir tweet metninin gerçek bir afet olayıyla ilgili olup olmadığını tahmin eder."
)

tweet = st.text_area(
    "Tweet metnini giriniz:",
    placeholder="Örnek: Forest fire near La Ronge Sask. Canada"
)

if st.button("Tahmin Et"):
    if tweet.strip() == "":
        st.warning("Lütfen bir tweet metni giriniz.")
    else:
        clean_tweet = temizle(tweet)
        tweet_tfidf = tfidf.transform([clean_tweet])
        prediction = model.predict(tweet_tfidf)[0]

        if prediction == 1:
            st.error("Sonuç: Bu tweet gerçek bir afetle ilgili olabilir.")
        else:
            st.success("Sonuç: Bu tweet gerçek bir afetle ilgili görünmüyor.")

        st.write("Temizlenmiş metin:")
        st.info(clean_tweet)