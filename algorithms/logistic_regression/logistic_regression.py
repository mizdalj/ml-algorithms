import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from algorithms.logistic_regression.my_logistic_regression import MyLogisticRegression
from algorithms.utils import my_train_test_split


def spam_classification(data_file, implementation='scikit-learn'):
    # 1. Data Preparation
    data = pd.read_csv(data_file, low_memory=False)
    data = data.dropna(subset=['email'])

    X = data['email']
    y = data['label'].reset_index(drop=True)  # Reset indices of y

    # Text preprocessing can be enhanced further
    X = X.str.lower().str.replace('[^a-z\s]', '')

    # Convert email content into numerical features using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    X_vectorized = tfidf.fit_transform(X).toarray()

    # 2. Model Creation and Training
    if implementation == 'scikit-learn':
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
    else:
        X_train, X_test, y_train, y_test = my_train_test_split(X_vectorized, y, test_size=0.2, random_seed=42)

        # Convert y_train and y_test to numpy arrays for compatibility with custom implementation
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        model = MyLogisticRegression(learning_rate=0.2, n_iters=1000)

    model.fit(X_train, y_train)

    # 3. Evaluation
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
