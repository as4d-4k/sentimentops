import os, sys, click, joblib
# at the top of cli.py — update imports
sys.path.insert(0, os.path.dirname(__file__))
from predict import load_model, predict_sentiment, predict_batch
from preprocess import clean_text, load_imdb, preprocess, save_data
from train import build_pipeline, evaluate, save_model, train, TRAIN_PATH, TEST_PATH

#CLI Group
@click.group()
@click.version_option(version="1.0.0", prog_name='sentimentops')
def cli():
    """
    SentimentOps -- End-to-End sentiment analysis MLOps Pipeline
    Train models, make predictions, evaluate performance,
    and submit jobs to Azure ML - All from command line
    """
#command 1: preprocess
@cli.command
def preprocess_data():
    """ Download and preprocess the IMDB dataset."""
    click.echo("Loading IMDB dataset from HuggingFace...")
    train_df, test_df = load_imdb()

    click.echo("Preprocess...")
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    os.makedirs('data', exist_ok=True)
    save_data(train_df, test_df)

    click.secho(f"Train Samples: {len(train_df)}", fg = 'green')
    click.secho(f"Test Samples: {test_df}", fg= 'green')
    click.secho("Data saved to data/", fg="green")

#Command 2: Train
@cli.command()
@click.option("--max-features", default=50000,  show_default=True, help="TF-IDF vocabulary size")
@click.option("--ngram-range",  default="1,2",  show_default=True, help="TF-IDF ngram range e.g. 1,2")
@click.option("--c",            default=1.0,    show_default=True, help="Logistic Regression regularization")
@click.option("--track/--no-track", default=True, help="Enable or disable MLflow tracking")
def train_model(max_features, ngram_range, c, track):
    """Train the TF-IDF + Logistic Regression model."""

    # check data exists
    if not os.path.exists(TRAIN_PATH):
        click.secho(
            "data/train.csv not found. Run: sentimentops preprocess-data first.",
            fg="red"
        )
        sys.exit(1)

    click.echo(f"Training with max_features={max_features}, C={c}, ngram_range={ngram_range}")

    ngram = tuple(int(x) for x in ngram_range.split(","))

    if track:
        click.echo("MLflow tracking enabled.")
        train(max_features=max_features, ngram_range=ngram, C=c)
    else:
        # train without MLflow
        click.echo("MLflow tracking disabled.")
        import pandas as pd
        train_df = pd.read_csv(TRAIN_PATH)
        test_df  = pd.read_csv(TEST_PATH)

        pipeline = build_pipeline(max_features, ngram, c)
        pipeline.fit(train_df["text"], train_df["label"])
        metrics  = evaluate(pipeline, test_df["text"], test_df["label"])

        click.secho(f"\nAccuracy: {metrics['accuracy']:.4f}", fg="green")
        click.echo(metrics["report"])
        save_model(pipeline)

    click.secho("Training complete.", fg="green")


#  Command 3: predict 

@cli.command()
@click.argument("text")
@click.option("--model-path", default="data/model.joblib", show_default=True)
@click.option("--source", type=click.Choice(["local", "azure"]), default="local")
def predict(text, model_path, source):
    """Predict sentiment of a movie review."""

    # source flag overrides model-path only if explicitly different from default
    paths = {
        "local" : "data/model.joblib",
        "azure" : "data/azure_outputs/model.joblib",
    }

    # if user passed --model-path explicitly, use that
    # otherwise use the source-based path
    final_path = model_path if model_path != "data/model.joblib" else paths[source]

    if not os.path.exists(final_path):
        click.secho(
            f"Model not found at '{final_path}'. "
            f"Run 'sentimentops train-model' first.",
            fg="red"
        )
        sys.exit(1)

    try:
        pipeline = load_model(final_path)
        result   = predict_sentiment(text, pipeline)
    except FileNotFoundError as e:
        click.secho(str(e), fg="red")
        sys.exit(1)
    except ValueError as e:
        click.secho(str(e), fg="red")
        sys.exit(1)

    color = "green" if result["sentiment"] == "positive" else "red"

    click.echo(f"\nModel       : {source.upper()}")
    click.echo(f"Input       : {text[:80]}...")
    click.secho(f"Sentiment   : {result['sentiment'].upper()}", fg=color, bold=True)
    click.echo(f"Confidence  : {result['confidence']:.2%}")


#Command 4: batch evaluate
@cli.command()
@click.option("--model-path", default="data/model.joblib", show_default=True)
@click.option("--source", type=click.Choice(["local", "azure"]), default="local")
def batch_predict(model_path, source):
    """
    Predict sentiment for multiple reviews from stdin.

    Usage:

        echo "great movie\\nterrible film" | sentimentops batch-predict
    """
    paths = {
        "local" : "data/model.joblib",
        "azure" : "data/azure_outputs/model.joblib",
    }

    path = paths[source] if source else model_path

    try:
        pipeline = load_model(path)
    except FileNotFoundError as e:
        click.secho(str(e), fg="red")
        sys.exit(1)

    texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        click.secho("No input provided via stdin.", fg="red")
        sys.exit(1)

    results = predict_batch(texts, pipeline)

    click.echo(f"\nBatch results ({len(results)} reviews):\n")
    for i, (text, result) in enumerate(zip(texts, results), 1):
        color = "green" if result["sentiment"] == "positive" else "red"
        click.echo(f"{i}. {text[:50]}...")
        click.secho(
            f"   → {result['sentiment'].upper()} ({result['confidence']:.2%})",
            fg=color
        )


#Command 5: evaluate
@cli.command()
@click.option("--model-path", default="data/model.joblib", show_default=True, help="Path to model file")
@click.option("--test-path",  default="data/test.csv",     show_default=True, help="Path to test CSV")
def evaluate_model(model_path, test_path):
    """Evaluate model performance on the test set."""
    import pandas as pd

    if not os.path.exists(model_path):
        click.secho(f"Model not found at {model_path}.", fg="red")
        sys.exit(1)

    if not os.path.exists(test_path):
        click.secho(f"Test data not found at {test_path}.", fg="red")
        sys.exit(1)

    click.echo("Loading model and test data...")
    pipeline = joblib.load(model_path)
    test_df  = pd.read_csv(test_path)

    metrics = evaluate(pipeline, test_df["text"], test_df["label"])

    click.secho(f"\nAccuracy: {metrics['accuracy']:.4f}", fg="green", bold=True)
    click.echo(f"\nClassification Report:\n{metrics['report']}")


# Command 5: submit-job 

@cli.command()
@click.option("--max-features", default=50000, show_default=True, help="TF-IDF vocabulary size")
@click.option("--ngram-range",  default="1,2", show_default=True, help="ngram range e.g. 1,2")
@click.option("--c",            default=1.0,   show_default=True, help="Logistic Regression C")
def cloud_train(max_features, ngram_range, c):
    """Submit a training job to Azure ML cloud."""
    import subprocess

    # use sys.executable to get the current venv's python
    # this guarantees the subprocess uses the same venv with all packages
    python = sys.executable

    project_root = os.path.join(os.path.dirname(__file__), "..")

    result = subprocess.run(
        [
            python, "azure/submit_job.py",
            "--max_features", str(max_features),
            "--ngram_range",  ngram_range,
            "--C",            str(c),
        ],
        cwd=os.path.abspath(project_root)
    )
    sys.exit(result.returncode)


# Entry Point 
    cli()