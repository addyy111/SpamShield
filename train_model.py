
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_PATH = BASE_DIR / "spam_model.pkl"
RESULTS_PATH = BASE_DIR / "model_comparison.csv"

# Load the dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

df = df.dropna(subset=["label", "message"])

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

X = df["message"]
y = df["label"]

# Use the same test data for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Naive Bayes": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        ("classifier", MultinomialNB())
    ]),

    "Logistic Regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Spam Precision": precision,
        "Spam Recall": recall,
        "Spam F1-score": f1
    })

    trained_models[name] = model

    print(f"\n{name} classification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Spam", "Spam"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# Display and save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# Select the model with the highest spam F1-score
best_name = results_df.loc[
    results_df["Spam F1-score"].idxmax(),
    "Model"
]

best_model = trained_models[best_name]
joblib.dump(best_model, MODEL_PATH)

print(f"\nSelected model: {best_name}")
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved comparison to: {RESULTS_PATH}")
