from setuptools import setup, find_packages

setup(
    name    = "sentimentops",
    version = "0.1.0",
    author  = "Asad Ullah",
    description = "End-to-end MLOps pipeline for sentiment analysis",
    packages    = find_packages(),
    python_requires = ">=3.8",
    install_requires = open("requirements.txt").read().splitlines(),
    entry_points = {
        "console_scripts": [
            "sentimentops=src.cli:cli",
        ],
    },
)