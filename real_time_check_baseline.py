import torch
import re
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path.home() / "medical_nlp"

MODEL_PATH = ROOT / "results_baseline_no_ner/model"
DATA_PATH = ROOT / "data"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading Baseline BioLinkBERT (NO NER)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded.\n")

# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    return text.strip()

# --------------------------------------------------
# CLEANING (VERY SIMPLE - SAME AS TRAINING)
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

print("=== BASELINE (NO NER) REAL-TIME SYSTEM ===")

while True:

    try:
        note_file = input("\nEnter NOTE PDF filename: ")
        claim_file = input("Enter CLAIM PDF filename: ")

        note_path = DATA_PATH / note_file
        claim_path = DATA_PATH / claim_file

        # -----------------------------
        # EXTRACT TEXT
        # -----------------------------

        note_text = extract_text_from_pdf(note_path)
        claim_text = extract_text_from_pdf(claim_path)

        print("\n--- RAW NOTE (first 300 chars) ---")
        print(note_text[:300])

        print("\n--- RAW CLAIM ---")
        print(claim_text)

        # -----------------------------
        # CLEAN TEXT
        # -----------------------------

        note_text = clean_text(note_text)
        claim_text = clean_text(claim_text)

        # -----------------------------
        # MODEL INPUT (NO PROCESSING)
        # -----------------------------

        inputs = tokenizer(
            note_text,
            claim_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # -----------------------------
        # PREDICTION
        # -----------------------------

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

        label = "SUPPORTED ✅" if pred == 1 else "NOT SUPPORTED ❌"

        print("\n==============================")
        print("BASELINE PREDICTION:", label)
        print("Confidence:", round(confidence, 3))
        print("==============================")

    except KeyboardInterrupt:
        print("\nExiting...")
        break

    except Exception as e:
        print("\nError:", e)
        continue