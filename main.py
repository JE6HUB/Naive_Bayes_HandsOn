import pandas as pd
import EvalMetrics

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

"""
This script implements a Naive Bayes classifier for sentiment analysis on the IMDB movie reviews dataset.
It includes data preprocessing, vectorization methods, model training, prediction, and evaluation.
"""

# Load the dataset
splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])

"""
Data Preprocessing
- The dataset is already in a clean format, but we will perform some text normalization.
- Convert text to lowercase, remove newlines, and extra spaces.
- The 'text' column contains the movie reviews, and the 'label' column contains the sentiment labels (0 for negative, 1 for positive).
"""

#text normalization
def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = ' '.join(text.split())  # Remove extra spaces
    return text
# Apply normalization to the text columns
df_train['text'] = df_train['text'].apply(normalize_text)
df_test['text'] = df_test['text'].apply(normalize_text)


"""
Vectorization methods
- CountVectorizer: Converts a collection of text documents to a matrix of token counts.
- TfidfVectorizer: Converts a collection of text documents to a matrix of TF-IDF features.
- Both methods return the training and test sets in the form of sparse matrices.
- CountVec() uses CountVectorizer.
- TfidfVec() uses TfidfVectorizer.
"""

#CountVectorizer
def CountVec():
    vectorizer = CountVectorizer() #create a 'vectorizer' object

    X_train = vectorizer.fit_transform(df_train['text'])
    Y_train = df_train['label']
    X_test = vectorizer.transform(df_test['text'])
    Y_test = df_test['label']
    return X_train, Y_train, X_test, Y_test
#TfidfVectorizer
def TfidfVec():
    vectorizer = TfidfVectorizer() #create a 'vectorizer' object

    X_train = vectorizer.fit_transform(df_train['text'])
    Y_train = df_train['label']
    X_test = vectorizer.transform(df_test['text'])
    Y_test = df_test['label']
    return X_train, Y_train, X_test, Y_test

# Choose the vectorization method
X_train, Y_train, X_test, Y_test = CountVec()  # or TfidfVec()


""" 
Model Training and Prediction
- We will use the Multinomial Naive Bayes model from sklearn.
- The model is trained on the training set and then used to predict labels for the test set.
- The model can also be implemented manually using the SparseMultinomialNB class.
"""
model = MultinomialNB(alpha=1.0) # set a model object
model.fit(X_train, Y_train) # train the model
Y_pred = model.predict(X_test) # predict labels for a test set

#if you want to use manual implementation, comment the above lines and uncomment the following lines
# from ManualImplement import SparseMultinomialNB
# model = SparseMultinomialNB(alpha=1.0)
# model.fit(X_train, Y_train)
# Y_pred = model.predict(X_test)


"""
Evaluation
- We will evaluate the model using accuracy, precision, recall, and F1-score.
- The EvalMetrics module provides functions to calculate these metrics and print a classification report.
"""
# Evaluate the model
EvalMetrics.print_classification_report(Y_test, Y_pred)