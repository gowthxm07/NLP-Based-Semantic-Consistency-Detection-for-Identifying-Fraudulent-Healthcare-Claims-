import pandas as pd
import numpy as np
import ast
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

print("=== HARD NEGATIVE GENERATION USING FAISS ===")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

ROOT = Path.home() / "medical_nlp"

CLAIMS_PATH = ROOT / "data/claims_natural_language_dedup.csv"
NOTES_PATH = ROOT / "data/note_entities.csv"

TRAIN_PATH = ROOT / "data/train/train.csv"
VAL_PATH = ROOT / "data/val/val.csv"
TEST_PATH = ROOT / "data/test/test.csv"

OUTPUT_DIR = ROOT / "pairs"
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_K = 50

TARGETS = {
    "train":3000,
    "val":1000,
    "test":5000
}

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

claims_df = pd.read_csv(CLAIMS_PATH)
notes_df = pd.read_csv(NOTES_PATH)

notes_df["diagnoses"] = notes_df["diagnoses"].apply(ast.literal_eval)
notes_df["procedures"] = notes_df["procedures"].apply(ast.literal_eval)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

datasets = {
    "train":train_df,
    "val":val_df,
    "test":test_df
}

print("Claims:", len(claims_df))
print("Notes:", len(notes_df))

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

torch.set_grad_enabled(False)

model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO",
    device="cuda"
)

# --------------------------------------------------
# BUILD NOTE TEXT
# --------------------------------------------------

note_texts = []
note_ids = []

for idx,row in notes_df.iterrows():

    diag = row["diagnoses"] if isinstance(row["diagnoses"],list) else []
    proc = row["procedures"] if isinstance(row["procedures"],list) else []

    text = " ".join(diag + proc)

    note_texts.append(text)
    note_ids.append(row["note_id"])

# --------------------------------------------------
# EMBED NOTES
# --------------------------------------------------

print("Embedding notes...")

note_embeddings = model.encode(
    note_texts,
    batch_size=256,
    convert_to_numpy=True,
    show_progress_bar=True
)

# normalize
faiss.normalize_L2(note_embeddings)

dim = note_embeddings.shape[1]

# --------------------------------------------------
# FAISS INDEX
# --------------------------------------------------

index = faiss.IndexFlatIP(dim)

index.add(note_embeddings)

print("FAISS index built")

# --------------------------------------------------
# EMBED CLAIMS
# --------------------------------------------------

print("Embedding claims...")

claim_texts = claims_df["claim_natural_text"].fillna("").tolist()

claim_embeddings = model.encode(
    claim_texts,
    batch_size=256,
    convert_to_numpy=True,
    show_progress_bar=True
)

faiss.normalize_L2(claim_embeddings)

# --------------------------------------------------
# RETRIEVE CANDIDATES
# --------------------------------------------------

print("Searching FAISS index...")

scores, indices = index.search(claim_embeddings, TOP_K)

# --------------------------------------------------
# GENERATE HARD NEGATIVES
# --------------------------------------------------

for split,df in datasets.items():

    print(f"\nGenerating hard negatives for {split}")

    target = TARGETS[split]

    claims = set(df["claim_index"].unique())

    existing_pairs = set(zip(df.claim_index,df.note_id))

    hard_rows = []
    used_pairs = set()

    for claim_idx in tqdm(claims):

        claim_scores = scores[claim_idx]
        claim_notes = indices[claim_idx]

        # iterate from lowest similarity
        for score,note_pos in zip(claim_scores[::-1],claim_notes[::-1]):

            note_id = note_ids[note_pos]

            pair = (claim_idx,note_id)

            if pair in existing_pairs:
                continue

            if pair in used_pairs:
                continue

            if score < 0.80:  # low similarity threshold

                hard_rows.append({
                    "claim_index":claim_idx,
                    "note_id":note_id,
                    "final_score":float(score),
                    "label":0
                })

                used_pairs.add(pair)

                if len(hard_rows) >= target:
                    break

        if len(hard_rows) >= target:
            break

    out_path = OUTPUT_DIR / f"hard_inconsistent_{split}.csv"

    pd.DataFrame(hard_rows).to_csv(out_path,index=False)

    print("Saved:",out_path)
    print("Total:",len(hard_rows))

print("\n=== HARD NEGATIVE GENERATION COMPLETE ===")