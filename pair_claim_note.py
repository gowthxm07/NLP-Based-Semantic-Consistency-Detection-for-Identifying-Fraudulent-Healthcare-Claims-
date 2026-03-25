import pandas as pd
import numpy as np
import ast
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=== FAISS CLAIM-NOTE PAIRING (SAMPLE MODE) ===")

# -----------------------------
# CONFIG
# -----------------------------
CLAIMS_PATH = "claims_natural_language.csv"
NOTE_ENTITIES_PATH = "data/note_entities.csv"
STRUCTURED_NOTES_PATH = "data/structured_notes.csv"

TOP_K = 50
SIM_THRESHOLD = 0.75

# -----------------------------
# LOAD DATA
# -----------------------------
claims_df = pd.read_csv(CLAIMS_PATH)
notes_df = pd.read_csv(NOTE_ENTITIES_PATH)
structured_df = pd.read_csv(STRUCTURED_NOTES_PATH)

# Convert string lists to Python lists
notes_df["diagnoses"] = notes_df["diagnoses"].apply(ast.literal_eval)
notes_df["procedures"] = notes_df["procedures"].apply(ast.literal_eval)

# Merge raw note text
notes_df = notes_df.merge(
    structured_df,
    left_on="note_id",
    right_index=True,
    how="left"
)

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# BUILD DIAGNOSIS EMBEDDING INDEX
# -----------------------------
print("Embedding note diagnoses...")

all_diag_texts = []
diag_to_note_map = []

for idx, row in notes_df.iterrows():
    for diag in row["diagnoses"]:
        all_diag_texts.append(diag)
        diag_to_note_map.append(idx)

diag_embeddings = model.encode(all_diag_texts, convert_to_numpy=True)

# Normalize for cosine similarity
faiss.normalize_L2(diag_embeddings)

dimension = diag_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(diag_embeddings)

print("FAISS index built with", len(all_diag_texts), "diagnosis entities")

# -----------------------------
# PAIR GENERATION
# -----------------------------
consistent_pairs = []
inconsistent_pairs = []

for _, claim in claims_df.iterrows():

    claim_diag = claim["diagnosis_text"]
    claim_proc = claim["procedure_text"]

    claim_diag_emb = model.encode([claim_diag], convert_to_numpy=True)
    faiss.normalize_L2(claim_diag_emb)

    # Retrieve top-K similar diagnosis entities
    scores, indices = index.search(claim_diag_emb, TOP_K)

    candidate_note_indices = set()
    for idx in indices[0]:
        candidate_note_indices.add(diag_to_note_map[idx])

    for note_idx in candidate_note_indices:

        note = notes_df.loc[note_idx]

        if len(note["procedures"]) == 0:
            continue

        # Compute diagnosis similarity
        note_diag_emb = model.encode(note["diagnoses"], convert_to_numpy=True)
        diag_sim = np.max(cosine_similarity(claim_diag_emb, note_diag_emb))

        # Compute procedure similarity
        claim_proc_emb = model.encode([claim_proc], convert_to_numpy=True)
        note_proc_emb = model.encode(note["procedures"], convert_to_numpy=True)
        proc_sim = np.max(cosine_similarity(claim_proc_emb, note_proc_emb))

        final_score = 0.6 * diag_sim + 0.4 * proc_sim

        row = {
            **claim.to_dict(),
            "note_id": note["note_id"],
            "patient_id": note["patient_id"],
            "note_diagnoses": note["diagnoses"],
            "note_procedures": note["procedures"],
            "note_full_text": note["note_text"],
            "diagnosis_similarity": float(diag_sim),
            "procedure_similarity": float(proc_sim),
            "final_score": float(final_score)
        }

        if diag_sim >= SIM_THRESHOLD and proc_sim >= SIM_THRESHOLD:
            row["label"] = 1
            consistent_pairs.append(row)
        else:
            row["label"] = 0
            inconsistent_pairs.append(row)

        # Stop after 5 + 5
        if len(consistent_pairs) >= 5 and len(inconsistent_pairs) >= 5:
            break

    if len(consistent_pairs) >= 5 and len(inconsistent_pairs) >= 5:
        break

# -----------------------------
# SAVE
# -----------------------------
pd.DataFrame(consistent_pairs).to_csv("data/consistent_sample.csv", index=False)
pd.DataFrame(inconsistent_pairs).to_csv("data/inconsistent_sample.csv", index=False)

print("Saved:")
print("  consistent_sample.csv")
print("  inconsistent_sample.csv")