import pandas as pd
import numpy as np
import ast
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print("=== STAGE 1: TOP-5 CANDIDATE GENERATION (OPTIMIZED) ===")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CLAIMS_PATH = "data/claims_natural_language_dedup.csv"
NOTE_ENTITIES_PATH = "data/note_entities.csv"
OUTPUT_PATH = "data/all_candidate_pairs_top5.csv"

TOP_K = 20
DIAG_WEIGHT = 0.6
PROC_WEIGHT = 0.4
CHUNK_SIZE = 5000

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

claims_df = pd.read_csv(CLAIMS_PATH, low_memory=False)
notes_df = pd.read_csv(NOTE_ENTITIES_PATH)

notes_df["diagnoses"] = notes_df["diagnoses"].apply(ast.literal_eval)
notes_df["procedures"] = notes_df["procedures"].apply(ast.literal_eval)

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
# BUILD DIAGNOSIS INDEX
# --------------------------------------------------

print("Embedding note diagnoses...")

all_diag_texts = []
diag_to_note = []
note_to_diag_indices = {}

idx = 0

for note_idx, row in notes_df.iterrows():
    note_to_diag_indices[note_idx] = []
    for diag in row["diagnoses"]:
        all_diag_texts.append(diag)
        diag_to_note.append(note_idx)
        note_to_diag_indices[note_idx].append(idx)
        idx += 1

diag_embeddings = model.encode(
    all_diag_texts,
    batch_size=256,
    convert_to_numpy=True,
    show_progress_bar=True
)

faiss.normalize_L2(diag_embeddings)

dim = diag_embeddings.shape[1]

index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200

index.add(diag_embeddings)

print("Diagnosis index built")

# --------------------------------------------------
# PROCEDURE EMBEDDINGS (BATCHED)
# --------------------------------------------------

print("Embedding note procedures (batched)...")

all_proc_texts = []
proc_to_note = []

for note_idx, row in notes_df.iterrows():
    for proc in row["procedures"]:
        all_proc_texts.append(proc)
        proc_to_note.append(note_idx)

proc_embeddings = model.encode(
    all_proc_texts,
    batch_size=256,
    convert_to_numpy=True,
    show_progress_bar=True
)

faiss.normalize_L2(proc_embeddings)

note_proc_embeddings = {}

cursor = 0

for note_idx, row in notes_df.iterrows():

    count = len(row["procedures"])

    if count == 0:
        note_proc_embeddings[note_idx] = None
    else:
        note_proc_embeddings[note_idx] = proc_embeddings[cursor:cursor+count]
        cursor += count

# --------------------------------------------------
# EMBED CLAIMS
# --------------------------------------------------

print("Embedding claim texts...")

claim_texts = claims_df["claim_natural_text"].fillna("").tolist()

claim_embeddings = model.encode(
    claim_texts,
    batch_size=256,
    convert_to_numpy=True,
    show_progress_bar=True
)

faiss.normalize_L2(claim_embeddings)

# --------------------------------------------------
# FAISS SEARCH
# --------------------------------------------------

print("Searching index...")

scores_matrix, indices_matrix = index.search(claim_embeddings, TOP_K)

# --------------------------------------------------
# STREAM WRITE RESULTS
# --------------------------------------------------

print("Generating candidate pairs...")

first_write = True
buffer = []

for claim_idx in tqdm(range(len(claims_df))):

    claim_emb = claim_embeddings[claim_idx].reshape(1,-1)
    claim_row = claims_df.iloc[claim_idx]

    candidate_diag_indices = indices_matrix[claim_idx]

    candidate_notes = set(diag_to_note[i] for i in candidate_diag_indices)

    for note_idx in candidate_notes:

        diag_indices = note_to_diag_indices[note_idx]
        if not diag_indices:
            continue

        note_diag_emb = diag_embeddings[diag_indices]
        diag_sim = np.max(np.dot(claim_emb, note_diag_emb.T))

        note_proc_emb = note_proc_embeddings[note_idx]
        if note_proc_emb is None:
            continue

        proc_sim = np.max(np.dot(claim_emb, note_proc_emb.T))

        final_score = DIAG_WEIGHT * diag_sim + PROC_WEIGHT * proc_sim

        buffer.append({
            "claim_index": claim_idx,
            "ClaimID": claim_row.get("ClaimID", None),
            "note_id": note_idx,
            "diagnosis_similarity": float(diag_sim),
            "procedure_similarity": float(proc_sim),
            "final_score": float(final_score)
        })

    if claim_idx % CHUNK_SIZE == 0 and claim_idx > 0:

        pd.DataFrame(buffer).to_csv(
            OUTPUT_PATH,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        buffer = []
        first_write = False

# write remaining rows
if buffer:

    pd.DataFrame(buffer).to_csv(
        OUTPUT_PATH,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )

print("Stage 1 complete")