# Disaster Tweet Classification

## Project Overview

This project was developed using the Kaggle competition **Natural Language Processing with Disaster Tweets**. The objective is to classify whether a tweet is related to a real disaster event or not based on its textual content.

Various Natural Language Processing (NLP) techniques, feature engineering approaches, and machine learning algorithms were evaluated throughout the project. The final solution uses **TF-IDF Vectorization** and **Logistic Regression**, which achieved the best overall performance.

**Kaggle Notebook:** [View Notebook](https://www.kaggle.com/code/mhskaya/natural-language-processing-with-disaster-tweets)

**Live Demo:** [Try Application](https://huggingface.co/spaces/MSK34/disaster-tweet-classification)

---

## Dataset

Source:

* Kaggle: Natural Language Processing with Disaster Tweets

Dataset Features:

* id
* keyword
* location
* text
* target

Target Variable:

* 0 → Not a real disaster
* 1 → Real disaster

---

## Exploratory Data Analysis

The following analyses were performed:

* Missing value analysis
* Class distribution analysis
* Tweet length analysis
* Word frequency analysis
* Disaster keyword analysis
* WordCloud visualization
* Disaster keyword effectiveness analysis

---

## Text Preprocessing

The tweet texts were cleaned before model training:

* Lowercase conversion
* URL removal
* Mention removal
* Special character removal
* Stopword removal

---

## Feature Engineering

Additional numerical features were generated:

* tweet_length
* word_count
* unique_word_count
* hashtag_count
* mention_count
* url_count
* exclamation_count

Keyword information was also incorporated into tweet text to improve contextual understanding.

---

## Models Evaluated

| Model                     | Accuracy | F1 Score |
| ------------------------- | -------- | -------- |
| Logistic Regression       | 0.8247   | 0.7781   |
| Logistic Regression (C=2) | 0.8201   | 0.7769   |
| Multinomial Naive Bayes   | 0.8221   | 0.7698   |
| Ridge Classifier          | 0.8109   | 0.7696   |
| Linear SVC                | 0.7997   | 0.7597   |
| Bernoulli NB              | 0.8102   | 0.7417   |
| Passive Aggressive        | 0.7315   | 0.6913   |

---

## Final Model

### TF-IDF + Logistic Regression

TF-IDF Parameters:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)
```

Model Parameters:

```python
LogisticRegression(
    max_iter=1000
)
```

---

## Results

| Metric              | Score   |
| ------------------- | ------- |
| Validation Accuracy | 0.8247  |
| Validation F1 Score | 0.7781  |
| Kaggle Public Score | 0.79987 |

The final submission achieved a **0.79987 Kaggle Public Score**, outperforming the baseline solution provided in the competition.

---

## Streamlit Application

A Streamlit web application was developed for interactive tweet classification.

Users can:

* Enter any tweet text
* Instantly receive a prediction
* Determine whether the tweet is likely related to a real disaster event

The application loads the trained Logistic Regression model and TF-IDF vectorizer to generate real-time predictions.

---

## Project Structure

```text
disaster-tweet-classification
│
├── src
│   ├── app.py
│   ├── disaster_tweet_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── natural_language_processing_with_disaster_tweets.ipynb
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TF-IDF
* Logistic Regression
* NLP
* Streamlit
* Hugging Face Spaces

---

## Conclusion

This project demonstrates that traditional NLP techniques combined with classical machine learning algorithms remain highly effective for text classification problems. Despite testing multiple algorithms and additional engineered features, TF-IDF combined with Logistic Regression consistently provided the strongest performance, achieving a competitive Kaggle leaderboard score while maintaining a lightweight and deployable solution.
