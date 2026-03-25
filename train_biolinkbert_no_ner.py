import pandas as pd
import torch
import numpy as np
from pathlib import Path

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

print("=== BASELINE 1: BIOLINKBERT WITHOUT NER ===")

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path.home() / "medical_nlp"

TRAIN_PATH = ROOT / "data/train/train_with_notes.csv"
VAL_PATH = ROOT / "data/val/val_with_notes.csv"
TEST_PATH = ROOT / "data/test/test_with_notes.csv"

RESULTS = ROOT / "results_baseline_no_ner"

MODEL_DIR = RESULTS / "model"
PLOTS_DIR = RESULTS / "plots"
METRICS_DIR = RESULTS / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "michiyasunaga/BioLinkBERT-base"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# --------------------------------------------------
# BUILD PREMISE / HYPOTHESIS
# --------------------------------------------------

def build_inputs(df):

    premises = []
    hypotheses = []

    for _, row in df.iterrows():

        note = str(row["original_note"])
        claim = str(row["claim_natural_text"])

        premises.append(note)
        hypotheses.append(claim)

    df["premise"] = premises
    df["hypothesis"] = hypotheses

    return df

train_df = build_inputs(train_df)
val_df = build_inputs(val_df)
test_df = build_inputs(test_df)

# --------------------------------------------------
# CONVERT TO HF DATASETS
# --------------------------------------------------

train_dataset = Dataset.from_pandas(
    train_df[["premise","hypothesis","label"]]
)

val_dataset = Dataset.from_pandas(
    val_df[["premise","hypothesis","label"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["premise","hypothesis","label"]]
)

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):

    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    "torch",
    columns=["input_ids","attention_mask","label"]
)

val_dataset.set_format(
    "torch",
    columns=["input_ids","attention_mask","label"]
)

test_dataset.set_format(
    "torch",
    columns=["input_ids","attention_mask","label"]
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("Loading BioLinkBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1
    }

# --------------------------------------------------
# TRAINING ARGUMENTS
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir=str(MODEL_DIR),

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    load_best_model_at_end=True,
)

# --------------------------------------------------
# TRAINER
# --------------------------------------------------

trainer = Trainer(

    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("Training model...")

trainer.train()

trainer.save_model(MODEL_DIR)

# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

print("Evaluating on test set...")

predictions = trainer.predict(test_dataset)

logits = predictions.predictions
labels = predictions.label_ids

preds = np.argmax(logits, axis=1)

probs = torch.softmax(
    torch.tensor(logits),
    dim=1
)[:,1].numpy()

# --------------------------------------------------
# METRICS
# --------------------------------------------------

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)
auc = roc_auc_score(labels, probs)

report = classification_report(labels, preds)

print(report)

with open(METRICS_DIR / "classification_report.txt","w") as f:
    f.write(report)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(PLOTS_DIR / "confusion_matrix.png")

# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

fpr, tpr, _ = roc_curve(labels, probs)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(PLOTS_DIR / "roc_curve.png")

# --------------------------------------------------
# PRECISION RECALL CURVE
# --------------------------------------------------

precision, recall, _ = precision_recall_curve(labels, probs)

plt.figure()

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.savefig(PLOTS_DIR / "precision_recall_curve.png")

print("Baseline training complete.")