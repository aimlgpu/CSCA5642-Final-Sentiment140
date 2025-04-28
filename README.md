# Assignment: CSCA 5642 Introduction to Deep Learning Final Project

# Introduction: Sentiment Analysis on the Sentiment140 Dataset Using Deep Learning

## Project Overview

### Problem Statement
This project focuses on sentiment analysis of tweets from the Sentiment140 dataset, a key task for understanding public opinion in social media contexts. The objective is to classify tweets into positive or negative sentiments, which is a binary classification problem. The type of learning employed is supervised learning, utilizing labeled data (target: 0 = Negative, 4 = Positive, mapped to 0 and 1) to train deep learning models that predict sentiment based on tweet text.

### Importance and Goal
Sentiment analysis is crucial for companies and organizations, such as social media platforms like Twitter (now X) or marketing firms, to monitor public sentiment, manage brand reputation, and inform decision-making. Accurate sentiment classification can help identify trends, detect crises, and personalize user experiences, potentially impacting business strategies and customer engagement. The goal of this project is to develop and compare deep learning models—LSTM, CNN, and GRU—to accurately classify tweet sentiments. The project emphasizes performance metrics (e.g., accuracy, precision, recall, F1-score) and provides insights through visualizations (e.g., training history plots, confusion matrices). Additionally, this project leverages GPU-accelerated training (using Kaggle’s P100 GPU) to enhance my skills in NLP and deep learning model development.

## Dataset Description

### Data Source
The dataset used is the Sentiment140 dataset, sourced from Kaggle (link to dataset). This public dataset is cited as:
- Kazanova, A. (2010). Sentiment140 [Data set]. Kaggle. https://www.kaggle.com/datasets/kazanova/sentiment140

### Data Characteristics
- **Size:** The dataset contains 1,600,000 samples (rows) and 6 features (columns) initially, reduced to 2 features (text, target) for modeling after dropping irrelevant metadata columns (ids, date, flag, user).
- **Feature Types:** There is 1 text feature (text: the tweet content) and 1 categorical target feature (target: sentiment label, 0 = Negative, 4 = Positive, mapped to 0 and 1). After preprocessing, the text feature is transformed into tokenized sequences for deep learning models.
- **Key Features:**
    text: Text, the raw tweet content (e.g., "@switchfoot http://twitpic.com/2y1zl - Awww, t...").
 - target: Categorical, binary sentiment label (0 = Negative, 1 = Positive after mapping).

### Preprocessing Details:
- Tweets were cleaned by removing URLs, mentions, hashtags, punctuation, and converting to lowercase, reducing tweet length by ~50% on average.
- Initially, stopwords were removed (except 'not', 'no', 'never') to preserve sentiment-relevant context, but this was later adjusted to retain all stopwords for sequential models (LSTM, GRU).
- Outliers (tweets <10 characters after cleaning) were removed, resulting in 1,517,394 samples.
- Single Source: The data is from a single tabulated CSV file, not multi-table or gathered from multiple sources.
