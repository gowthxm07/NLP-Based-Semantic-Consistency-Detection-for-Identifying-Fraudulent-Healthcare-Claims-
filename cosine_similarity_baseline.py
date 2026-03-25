from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np
from pathlib import Path

print("=== COSINE SIMILARITY BASELINE ===")

# --------------------------------------------------
# PATH
# --------------------------------------------------

ROOT = Path.home() / "medical_nlp"
TEST_PATH = ROOT / "data/test/test_with_notes.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(TEST_PATH)

print("Loaded rows:", len(df))

notes = df["original_note"].fillna("").tolist()
claims = df["claim_natural_text"].fillna("").tolist()
labels = df["label"].values

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

# --------------------------------------------------
# GENERATE EMBEDDINGS
# --------------------------------------------------

print("Generating embeddings...")

note_embeddings = model.encode(notes, batch_size=64, show_progress_bar=True)
claim_embeddings = model.encode(claims, batch_size=64, show_progress_bar=True)

# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

scores = cosine_similarity(note_embeddings, claim_embeddings).diagonal()

print("Similarity score stats:")
print("Min:", scores.min())
print("Max:", scores.max())
print("Mean:", scores.mean())

# --------------------------------------------------
# THRESHOLD CLASSIFICATION
# --------------------------------------------------

threshold = 0.88

predictions = (scores > threshold).astype(int)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions)

print("\nAccuracy:", accuracy)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(labels, predictions))