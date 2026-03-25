import pandas as pd
import re
from pathlib import Path

print("=== CLEANING & STRUCTURING NOTEEVENTS (NOTE-LEVEL) ===")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path.home() / "medical_nlp"
notes_path = ROOT / "NOTEEVENTS.csv"
output_path = ROOT / "data" / "structured_notes.csv"

(ROOT / "data").mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD NOTES
# --------------------------------------------------
print("Loading NOTEEVENTS...")
notes = pd.read_csv(notes_path, low_memory=False)

print("Total notes:", len(notes))

# --------------------------------------------------
# FILTER NOTE TYPES (KEEP CLINICAL NOTES ONLY)
# --------------------------------------------------
if "CATEGORY" in notes.columns:
    allowed_categories = [
        "Discharge summary",
        "Physician",
        "Physician ",
        "Progress Note",
        "Operative Report"
    ]
    notes = notes[notes["CATEGORY"].isin(allowed_categories)]

print("After category filtering:", len(notes))

# --------------------------------------------------
# ENSURE REQUIRED COLUMNS
# --------------------------------------------------
if "SUBJECT_ID" not in notes.columns:
    raise ValueError("SUBJECT_ID column not found in NOTEEVENTS")

if "TEXT" not in notes.columns:
    raise ValueError("TEXT column not found in NOTEEVENTS")

# --------------------------------------------------
# BASIC TEXT CLEANING
# --------------------------------------------------
def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove common headers
    text = re.sub(r"Discharge Date:.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Dictated by:.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Signed by:.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Addendum:.*?\n", "", text, flags=re.IGNORECASE)

    # Remove excessive whitespace
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

print("Cleaning notes...")
notes["clean_text"] = notes["TEXT"].apply(clean_text)

# Remove very small notes
notes = notes[notes["clean_text"].str.len() > 50]

print("After removing empty/short notes:", len(notes))

# --------------------------------------------------
# KEEP ONE NOTE PER ROW
# --------------------------------------------------
structured_notes = notes[[
    "SUBJECT_ID",
    "clean_text"
]].copy()

structured_notes.columns = [
    "patient_id",
    "note_text"
]

print("Structured note-level rows:", len(structured_notes))
print("Unique patients:", structured_notes["patient_id"].nunique())

# --------------------------------------------------
# SAVE
# --------------------------------------------------
structured_notes.to_csv(output_path, index=False)

print("Saved structured notes to:", output_path)
print("=== DONE ===")