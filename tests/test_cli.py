import pytest
from click.testing import CliRunner
from src.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    """CLI should show help without errors."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "SentimentOps" in result.output


def test_cli_version(runner):
    """CLI should return version."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "1.0.0" in result.output


def test_predict_command(runner):
    """Predict command should return sentiment."""
    result = runner.invoke(cli, [
        "predict",
        "This movie was absolutely fantastic"
    ])
    assert result.exit_code == 0
    assert "Sentiment" in result.output
    assert "Confidence" in result.output


def test_predict_missing_model(runner, tmp_path):
    """Predict should fail gracefully if model doesn't exist."""
    result = runner.invoke(cli, [
        "predict",
        "--model-path", "nonexistent/model.joblib",
        "great movie"
    ])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_evaluate_missing_model(runner):
    """Evaluate should fail gracefully if model doesn't exist."""
    result = runner.invoke(cli, [
        "evaluate-model",
        "--model-path", "nonexistent/model.joblib"
    ])
    assert result.exit_code == 1