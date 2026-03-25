import pandas as pd
import re
from pathlib import Path

print("=== EXPANDING ABBREVIATIONS + CONVERTING CLAIMS TO NATURAL LANGUAGE ===")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path.home() / "medical_nlp"

input_path = ROOT / "claims_mapped_ready.csv"
output_path = ROOT / "claims_natural_language.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(input_path, low_memory=False)

# --------------------------------------------------
# ABBREVIATION DICTIONARY
# (You can expand this over time)
# --------------------------------------------------
ABBR_MAP = {
    "nos": "",
    "nec": "",
    "w/o": "without",
    "w/": "with",
    "hx": "history of",
    "s/p": "status post",
    "chr": "chronic",
    "crbl": "cerebral",
    "crny": "coronary",
    "athrslc": "atherosclerotic",
    "htn": "hypertension",
    "dmii": "type 2 diabetes mellitus",
    "dmi": "type 1 diabetes mellitus",
    "dmit": "type 1 diabetes mellitus",
    "dml": "diabetes mellitus",
    "cmp": "complications",
    "uncntrld": "uncontrolled",
    "nt st": "not stated",
    "mal neo": "malignant neoplasm",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "afib": "atrial fibrillation",
    "uti": "urinary tract infection",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury"
}

# --------------------------------------------------
# NORMALIZATION FUNCTION
# --------------------------------------------------
def expand_abbreviations(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Replace abbreviations
    for abbr, full in ABBR_MAP.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full, text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# COLLECT DIAGNOSIS + PROCEDURE TEXT COLUMNS
# --------------------------------------------------
diag_cols = [c for c in df.columns if "DiagnosisCode" in c and "_TEXT" in c]
proc_cols = [c for c in df.columns if "ProcedureCode" in c and "_TEXT" in c]

print("Diagnosis columns:", len(diag_cols))
print("Procedure columns:", len(proc_cols))


# --------------------------------------------------
# BUILD NATURAL LANGUAGE CLAIM
# --------------------------------------------------
def build_claim_sentence(row):

    # Expand diagnosis texts
    diagnoses = []
    for col in diag_cols:
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            cleaned = expand_abbreviations(val)
            diagnoses.append(cleaned)

    # Expand procedure texts
    procedures = []
    for col in proc_cols:
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            cleaned = expand_abbreviations(val)
            procedures.append(cleaned)

    # Remove duplicates
    diagnoses = list(set(diagnoses))
    procedures = list(set(procedures))

    # Construct sentence
    sentence = ""

    if diagnoses:
        sentence += "The patient was diagnosed with "
        sentence += ", ".join(diagnoses)

    if procedures:
        if diagnoses:
            sentence += ", and "
        else:
            sentence += "The patient "

        sentence += "underwent "
        sentence += ", ".join(procedures)

    if sentence == "":
        sentence = "No diagnosis or procedure information available."

    return sentence.strip()


# --------------------------------------------------
# APPLY TRANSFORMATION
# --------------------------------------------------
print("Converting claims to natural language...")
df["claim_natural_text"] = df.apply(build_claim_sentence, axis=1)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
df.to_csv(output_path, index=False)

print("Saved natural language claims to:", output_path)
print("=== DONE ===")