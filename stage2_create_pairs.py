import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import ast
from rapidfuzz import fuzz

print("=== STAGE 2: COVERAGE-AWARE CLAIM–NOTE PAIRING ===")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MAX_CLAIMS_PER_NOTE = 20

SCORE_THRESHOLD = 0.88
COVERAGE_THRESHOLD = 0.40
FUZZ_THRESHOLD = 85

BUFFER_SIZE = 20000

ROOT = Path.home() / "medical_nlp"

CANDIDATES_PATH = ROOT / "data" / "all_candidate_pairs_top5.csv"
CLAIMS_PATH = ROOT / "data" / "claims_natural_language_dedup.csv"
NOTES_PATH = ROOT / "data" / "note_entities.csv"

PAIRS_DIR = ROOT / "pairs"
PAIRS_DIR.mkdir(exist_ok=True)

CONSISTENT_PATH = PAIRS_DIR / "consistent.csv"
INCONSISTENT_PATH = PAIRS_DIR / "inconsistent.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

pairs_df = pd.read_csv(CANDIDATES_PATH)

claims_df = pd.read_csv(CLAIMS_PATH, low_memory=False)
claims_df["claim_index"] = claims_df.index

notes_df = pd.read_csv(NOTES_PATH)

notes_df["diagnoses"] = notes_df["diagnoses"].apply(ast.literal_eval)
notes_df["procedures"] = notes_df["procedures"].apply(ast.literal_eval)

print("Candidate pairs:", len(pairs_df))
print("Claims:", len(claims_df))
print("Notes:", len(notes_df))

# --------------------------------------------------
# CREATE FAST LOOKUPS
# --------------------------------------------------

claims_map = claims_df.set_index("claim_index").to_dict("index")
notes_map = notes_df.set_index("note_id").to_dict("index")

note_diag_map = {}
note_proc_map = {}

for note_id, row in notes_map.items():

    diag = [str(x).lower() for x in row["diagnoses"]] if isinstance(row["diagnoses"], list) else []
    proc = [str(x).lower() for x in row["procedures"]] if isinstance(row["procedures"], list) else []

    note_diag_map[note_id] = diag
    note_proc_map[note_id] = proc

# --------------------------------------------------
# SORT PAIRS
# --------------------------------------------------

pairs_df = pairs_df.sort_values(
    ["claim_index","final_score"],
    ascending=[True,False]
)

grouped = pairs_df.groupby("claim_index")

# --------------------------------------------------
# CLAIM ENTITY COLUMNS
# --------------------------------------------------

diag_cols = [f"ClmDiagnosisCode_{i}_TEXT" for i in range(1,11)]
proc_cols = [f"ClmProcedureCode_{i}_TEXT" for i in range(1,7)]

# --------------------------------------------------
# ENTITY FUNCTIONS
# --------------------------------------------------

def get_claim_entities(row):

    diag = []
    proc = []

    for c in diag_cols:
        val = row.get(c)
        if pd.notna(val):
            diag.append(str(val).lower())

    for c in proc_cols:
        val = row.get(c)
        if pd.notna(val):
            proc.append(str(val).lower())

    return diag, proc


def match_ratio(claim_terms, note_terms):

    if len(claim_terms) == 0:
        return 0

    matches = 0

    for c in claim_terms:

        best = 0

        for n in note_terms:
            score = fuzz.token_set_ratio(c,n)
            if score > best:
                best = score

        if best >= FUZZ_THRESHOLD:
            matches += 1

    return matches / len(claim_terms)

# --------------------------------------------------
# STREAM BUFFERS
# --------------------------------------------------

consistent_buffer=[]
inconsistent_buffer=[]

first_write_consistent=True
first_write_inconsistent=True

note_usage=defaultdict(int)

consistent_count=0
inconsistent_count=0
skipped_claims=0

# --------------------------------------------------
# PROCESS
# --------------------------------------------------

print("\nSelecting best candidate for each claim...")

for claim_idx,group in tqdm(grouped,total=len(grouped)):

    claim_row = claims_map[claim_idx]

    claim_diag,claim_proc = get_claim_entities(claim_row)

    best_candidate=None
    best_score=-1

    for _,row in group.iterrows():

        note_id=row["note_id"]
        score=row["final_score"]

        note_diag=note_diag_map.get(note_id,[])
        note_proc=note_proc_map.get(note_id,[])

        diag_cov=match_ratio(claim_diag,note_diag)
        proc_cov=match_ratio(claim_proc,note_proc)

        coverage=max(diag_cov,proc_cov)

        # build full row
        row_data={
            **row.to_dict(),
            **claim_row,
            **notes_map.get(note_id,{})
        }

        row_data["coverage"]=coverage
        row_data["diagnosis_coverage"]=diag_cov
        row_data["procedure_coverage"]=proc_cov

        if coverage < COVERAGE_THRESHOLD or score < SCORE_THRESHOLD:

            row_data["label"]=0
            inconsistent_buffer.append(row_data)
            inconsistent_count+=1

        else:

            if note_usage[note_id] < MAX_CLAIMS_PER_NOTE:

                if score>best_score:
                    best_score=score
                    best_candidate=row_data

    if best_candidate is not None:

        note_usage[best_candidate["note_id"]]+=1
        best_candidate["label"]=1

        consistent_buffer.append(best_candidate)
        consistent_count+=1

    else:
        skipped_claims+=1

    # write buffers
    if len(consistent_buffer)>=BUFFER_SIZE:

        pd.DataFrame(consistent_buffer).to_csv(
            CONSISTENT_PATH,
            mode="w" if first_write_consistent else "a",
            header=first_write_consistent,
            index=False
        )

        consistent_buffer=[]
        first_write_consistent=False

    if len(inconsistent_buffer)>=BUFFER_SIZE:

        pd.DataFrame(inconsistent_buffer).to_csv(
            INCONSISTENT_PATH,
            mode="w" if first_write_inconsistent else "a",
            header=first_write_inconsistent,
            index=False
        )

        inconsistent_buffer=[]
        first_write_inconsistent=False

# --------------------------------------------------
# WRITE REMAINING
# --------------------------------------------------

if consistent_buffer:
    pd.DataFrame(consistent_buffer).to_csv(
        CONSISTENT_PATH,
        mode="w" if first_write_consistent else "a",
        header=first_write_consistent,
        index=False
    )

if inconsistent_buffer:
    pd.DataFrame(inconsistent_buffer).to_csv(
        INCONSISTENT_PATH,
        mode="w" if first_write_inconsistent else "a",
        header=first_write_inconsistent,
        index=False
    )

print("\n=== DATASET STATS ===")

print("Consistent pairs:",consistent_count)
print("Inconsistent pairs:",inconsistent_count)
print("Unique notes used:",len(note_usage))
print("Skipped claims:",skipped_claims)