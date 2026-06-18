import os, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

def load_data(train_path:str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def build_pipeline():
    """
    build sklearn pipeline with 2steps:
    1: TF-IDF vectorizer
    2: Logistic regression
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features = 50000,
            ngram_range=(1,2),
            sublinear_tf = True,
        )),
        ('clf', LogisticRegression(
            max_iter = 1000,
            C=1.0,
            solver = 'lbfgs',
        ))
    ])
    return pipeline

def evaluate(pipeline: Pipeline, x_test: pd.Series, y_test: pd.Series):
    y_pred = pipeline.predict(x_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test,y_pred, target_names=['negative','positive'])
    }
    return  metrics

def save_model(pipeline: Pipeline, path = 'data/model.joblib'):
    """
    Save the pretrained model to disk
    """
    joblib.dump(pipeline, path)
    print(f"Model Saved to -> {path}")

if __name__ == "__main__":
    print("Loading Data...")
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    x_train = train_df['text']
    y_trian = train_df['label']
    x_test = test_df['text'] 
    y_test = test_df['label']

    print(f"train samples: {len(x_train)}")
    print(f"test samples {len(x_test)}")

    #Building Pipeline
    print("Building pipeline")
    pipeline = build_pipeline()
    print("pipeline built")

    #train the model
    print("Training...")
    pipeline.fit(x_train, y_trian)
    print("training complete")

    # Evaluating the model
    print("evaluating the model")
    metrics = evaluate(pipeline, x_test, y_test)
    print(f"\nAccurace: {metrics['accuracy']: .4f}")
    print(f"\nClassification report: {metrics['report']}")

    #saving the model
    save_model(pipeline)

