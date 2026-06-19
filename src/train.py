import os
import sys

# ensure src/ is on the path whether running locally or on Azure
sys.path.insert(0, os.path.dirname(__file__))
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import argparse

# ── Constants ─────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "sentimentops-tfidf-logreg"
MODEL_PATH      = "data/model.joblib"
TRAIN_PATH      = "data/train.csv"
TEST_PATH       = "data/test.csv"


def load_data(train_path: str, test_path: str):
    """Load preprocessed train and test CSVs."""
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline(max_features: int, ngram_range: tuple, C: float):
    """
    Build a sklearn Pipeline.
    Parameters are now explicit arguments so MLflow can log them.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=C,
            solver="lbfgs",
        ))
    ])
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series):
    """Run predictions and return metrics."""
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "report"   : classification_report(
                        y_test, y_pred,
                        target_names=["negative", "positive"]
                     )
    }
    return metrics


def save_model(pipeline: Pipeline, path: str = MODEL_PATH):
    """Save the trained pipeline to disk."""
    joblib.dump(pipeline, path)
    print(f"Model saved → {path}")


def train(
    max_features : int   = 50000,
    ngram_range  : tuple = (1, 2),
    C            : float = 1.0,
) -> None:
    """
    Full training run wrapped in an MLflow context.
    All params and metrics are logged automatically.
    """

    # ── 1. Setup MLflow ───────────────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        # ── 2. Log Parameters ─────────────────────────────────────────
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range",  str(ngram_range))
        mlflow.log_param("C",            C)

        # ── 3. Load Data ──────────────────────────────────────────────
        print("Loading data...")
        train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

        X_train, y_train = train_df["text"], train_df["label"]
        X_test,  y_test  = test_df["text"],  test_df["label"]

        # ── 4. Build and Train ────────────────────────────────────────
        print("Training...")
        pipeline = build_pipeline(max_features, ngram_range, C)
        pipeline.fit(X_train, y_train)

        # ── 5. Evaluate ───────────────────────────────────────────────
        print("Evaluating...")
        metrics = evaluate(pipeline, X_test, y_test)

        print(f"\nAccuracy : {metrics['accuracy']:.4f}")
        print(f"\n{metrics['report']}")

        # ── 6. Log Metrics to MLflow ──────────────────────────────────
        mlflow.log_metric("accuracy", metrics["accuracy"])

        # ── 7. Log Model to MLflow ────────────────────────────────────────────
# Azure ML has its own model registry — skip mlflow model logging there
        is_azure = os.environ.get("AZUREML_RUN_ID") is not None

        if not is_azure:
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name="sentimentops-tfidf"
            )
        else:
            print("Running on Azure ML — skipping mlflow.log_model (Azure handles this)")

        # ── 8. Save locally ───────────────────────────────────────────────────
        save_model(pipeline)

        

        print(f"\nMLflow run logged under experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int,   default=50000)
    parser.add_argument("--ngram_range",  type=str,   default="1,2")
    parser.add_argument("--C",            type=float, default=1.0)
    args = parser.parse_args()

    ngram = tuple(int(x) for x in args.ngram_range.split(","))

    # ── When running on Azure, data/ folder doesn't exist ─────────────
    # ── So we generate it fresh from HuggingFace ──────────────────────
    if not os.path.exists("data/train.csv"):
        print("data/train.csv not found — downloading from HuggingFace...")
        from preprocess import load_imdb, preprocess, save_data
        os.makedirs("data", exist_ok=True)
        train_df, test_df = load_imdb()
        train_df = preprocess(train_df)
        test_df  = preprocess(test_df)
        save_data(train_df, test_df)
        print("Data ready.")

    train(
        max_features = args.max_features,
        ngram_range  = ngram,
        C            = args.C,
    )