import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def spam_classification(data_file):
    # 1. Data Preparation
    data = pd.read_csv(data_file)

    # Drop rows where 'email' is NaN
    data = data.dropna(subset=['email'])

    X = data['email']
    y = data['label']

    # Text preprocessing can be enhanced further
    X = X.str.lower().str.replace('[^a-z\s]', '')

    # Convert email content into numerical features using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    X_vectorized = tfidf.fit_transform(X)

    # 2. Model Creation and Training
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 3. Evaluation
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))