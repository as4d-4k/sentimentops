import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler,
)
from datasets import load_dataset
from torch.optim import AdamW
import mlflow

sys.path.insert(0, os.path.dirname(__file__))

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME      = "distilbert-base-uncased"
EXPERIMENT_NAME = "sentimentops-distilbert"
MAX_LENGTH      = 512
BATCH_SIZE      = 16
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-5
MODEL_SAVE_PATH = "data/distilbert_model"


# ── Dataset ───────────────────────────────────────────────────────────────────

class IMDBDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for IMDB HuggingFace dataset.
    Tokenizes text on the fly.
    """
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── Training ──────────────────────────────────────────────────────────────────

def train_distilbert(
    num_epochs    : int   = NUM_EPOCHS,
    batch_size    : int   = BATCH_SIZE,
    learning_rate : float = LEARNING_RATE,
    max_length    : int   = MAX_LENGTH,
    sample_size   : int   = None,   # None = full dataset, int = subset for testing
):
    """
    Fine-tune DistilBERT on IMDB sentiment dataset.

    Args:
        num_epochs    : number of training epochs
        batch_size    : samples per batch
        learning_rate : AdamW learning rate
        max_length    : max token length (DistilBERT limit is 512)
        sample_size   : use subset of data for local testing
    """

    # ── 1. Device Setup ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 2. Load Dataset ───────────────────────────────────────────────
    print("Loading IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts  = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts   = dataset["test"]["text"]
    test_labels  = dataset["test"]["label"]

    # use subset for local testing
    if sample_size:
        print(f"Using subset of {sample_size} samples for local testing")
        train_texts  = train_texts[:sample_size]
        train_labels = train_labels[:sample_size]
        test_texts   = test_texts[:sample_size // 5]
        test_labels  = test_labels[:sample_size // 5]

    print(f"Train samples : {len(train_texts)}")
    print(f"Test samples  : {len(test_texts)}")

    # ── 3. Tokenize ───────────────────────────────────────────────────
    print("Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        train_texts,
        truncation  = True,
        padding     = True,
        max_length  = max_length,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation  = True,
        padding     = True,
        max_length  = max_length,
    )

    # ── 4. Create Datasets and DataLoaders ────────────────────────────
    train_dataset = IMDBDataset(train_encodings, train_labels)
    test_dataset  = IMDBDataset(test_encodings,  test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ── 5. Load Model ─────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = 2,
    )
    model.to(device)

    # ── 6. Optimizer and Scheduler ────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name               = "linear",
        optimizer          = optimizer,
        num_warmup_steps   = num_training_steps // 10,
        num_training_steps = num_training_steps,
    )

    # ── 7. MLflow Setup ───────────────────────────────────────────────
    is_azure = os.environ.get("AZUREML_RUN_ID") is not None
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        mlflow.log_param("model_name",    MODEL_NAME)
        mlflow.log_param("num_epochs",    num_epochs)
        mlflow.log_param("batch_size",    batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_length",    max_length)
        mlflow.log_param("train_samples", len(train_texts))

        # ── 8. Training Loop ──────────────────────────────────────────
        print("\nStarting training...")

        for epoch in range(num_epochs):
            model.train()
            total_loss    = 0
            correct       = 0
            total         = 0

            for step, batch in enumerate(train_loader):
                # move batch to device
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                # forward pass
                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels,
                )

                loss = outputs.loss

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # track metrics
                total_loss += loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                correct     += (predictions == labels).sum().item()
                total       += labels.size(0)

                # print progress every 50 steps
                if (step + 1) % 50 == 0:
                    avg_loss = total_loss / (step + 1)
                    accuracy = correct / total
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Step {step+1}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Acc: {accuracy:.4f}"
                    )

            # log epoch metrics to MLflow
            epoch_loss = total_loss / len(train_loader)
            epoch_acc  = correct / total
            mlflow.log_metric("train_loss",     epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc,  step=epoch)

            print(f"\nEpoch {epoch+1} complete | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # ── 9. Evaluation after each epoch ────────────────────────
            val_accuracy = evaluate_distilbert(model, test_loader, device)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            print(f"Validation accuracy: {val_accuracy:.4f}\n")

        # ── 10. Save Model ────────────────────────────────────────────
        print(f"Saving model to {MODEL_SAVE_PATH}...")
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print("Model saved.")

        if not is_azure:
            mlflow.log_artifacts(MODEL_SAVE_PATH, artifact_path="distilbert_model")


def evaluate_distilbert(model, test_loader, device):
    """Evaluate model on test set, return accuracy."""
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total   += labels.size(0)

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs",    type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length",    type=int,   default=MAX_LENGTH)
    parser.add_argument("--sample_size",   type=int,   default=None,
                        help="Use subset of data for local testing e.g. 200")
    args = parser.parse_args()

    train_distilbert(
        num_epochs    = args.num_epochs,
        batch_size    = args.batch_size,
        learning_rate = args.learning_rate,
        max_length    = args.max_length,
        sample_size   = args.sample_size,
    )