import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

ROOT = Path.home() / "medical_nlp"
MODEL_PATH = ROOT / "results/model"

print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded.\n")

while True:

    print("\nEnter patient diagnoses (comma separated):")
    diagnoses = input("> ")

    print("\nEnter procedures performed (comma separated):")
    procedures = input("> ")

    print("\nEnter claim to verify:")
    claim = input("> ")

    premise = f"""
NOTE_DIAGNOSES:
{diagnoses}

NOTE_PROCEDURES:
{procedures}
"""

    inputs = tokenizer(
        premise,
        claim,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()

    confidence = probs[0][pred].item()

    label = "SUPPORTED" if pred == 1 else "NOT SUPPORTED"

    print("\nPrediction:", label)
    print("Confidence:", round(confidence, 3))