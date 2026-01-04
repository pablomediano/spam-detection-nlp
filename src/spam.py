# Install required packages (only needed if not already installed)
# !pip install pandas scikit-learn nltk matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("data/youtube04_eminem.csv")
df = df[["content", "class"]]
df = df.rename(columns={"content": "message", "class": "label"})
df["label"] = df["label"].map({True: "spam", False: "ham"})

print(df.head())
print(df["label"].value_counts())


# Count spam vs ham
print(df["label"].value_counts())
sns.countplot(x="label", data=df)
plt.title("Distribution of Spam vs Ham")
plt.show()

X = df["message"]
y = df["label"]

df["label"] = df["label"].map({True: "spam", False: "ham"})



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Bag-of-Words
cv = CountVectorizer(stop_words="english")
X_train_bow = cv.fit_transform(X_train)
X_test_bow = cv.transform(X_test)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_bow, y_train)
y_pred_nb = nb.predict(X_test_bow)
print("Naive Bayes (BoW)")
print(classification_report(y_test, y_pred_nb))

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
print("Logistic Regression (TF-IDF)")
print(classification_report(y_test, y_pred_lr))

svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
print("Support Vector Machine (TF-IDF)")
print(classification_report(y_test, y_pred_svm))

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"],
                yticklabels=["ham", "spam"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

plot_cm(y_test, y_pred_nb, "Naive Bayes (BoW)")
plot_cm(y_test, y_pred_lr, "Logistic Regression (TF-IDF)")
plot_cm(y_test, y_pred_svm, "SVM (TF-IDF)")

