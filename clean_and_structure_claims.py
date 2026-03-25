import pandas as pd
import re
from pathlib import Path
import ast

print("=== CLEANING & STRUCTURING CLAIMS ===")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path.home() / "medical_nlp"
claims_path = ROOT / "claims_mapped_ready.csv"
output_path = ROOT / "data" / "structured_claims_cleaned.csv"

(ROOT / "data").mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD CLAIMS
# --------------------------------------------------
claims = pd.read_csv(claims_path, low_memory=False)

print("Total claims:", len(claims))

# --------------------------------------------------
# IDENTIFY DIAGNOSIS & PROCEDURE COLUMNS
# --------------------------------------------------
diag_code_cols = [c for c in claims.columns if "ClmDiagnosisCode_" in c and "_TEXT" not in c]
diag_text_cols = [c for c in claims.columns if "ClmDiagnosisCode_" in c and "_TEXT" in c]

proc_code_cols = [c for c in claims.columns if "ClmProcedureCode_" in c and "_TEXT" not in c]
proc_text_cols = [c for c in claims.columns if "ClmProcedureCode_" in c and "_TEXT" in c]

print("Diagnosis code columns:", len(diag_code_cols))
print("Diagnosis text columns:", len(diag_text_cols))
print("Procedure code columns:", len(proc_code_cols))
print("Procedure text columns:", len(proc_text_cols))

# --------------------------------------------------
# BASIC NORMALIZATION FUNCTION
# --------------------------------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # normalize spaces
    return text.strip()

# --------------------------------------------------
# FILL NA
# --------------------------------------------------
claims[diag_text_cols] = claims[diag_text_cols].fillna("")
claims[proc_text_cols] = claims[proc_text_cols].fillna("")
claims[diag_code_cols] = claims[diag_code_cols].fillna("")
claims[proc_code_cols] = claims[proc_code_cols].fillna("")

# --------------------------------------------------
# EXTRACT LISTS
# --------------------------------------------------
def extract_clean_list(row, text_cols, code_cols):
    text_list = []
    code_list = []

    for t_col, c_col in zip(text_cols, code_cols):
        text_val = normalize_text(row[t_col])
        code_val = str(row[c_col]).strip()

        if text_val != "":
            text_list.append(text_val)

        if code_val != "" and code_val.lower() != "nan":
            code_list.append(code_val)

    # Remove duplicates
    text_list = list(set(text_list))
    code_list = list(set(code_list))

    return text_list, code_list

# --------------------------------------------------
# APPLY EXTRACTION
# --------------------------------------------------
diagnosis_texts = []
diagnosis_codes = []
procedure_texts = []
procedure_codes = []

for _, row in claims.iterrows():
    d_texts, d_codes = extract_clean_list(row, diag_text_cols, diag_code_cols)
    p_texts, p_codes = extract_clean_list(row, proc_text_cols, proc_code_cols)

    diagnosis_texts.append(d_texts)
    diagnosis_codes.append(d_codes)
    procedure_texts.append(p_texts)
    procedure_codes.append(p_codes)

claims["diagnosis_texts"] = diagnosis_texts
claims["diagnosis_codes"] = diagnosis_codes
claims["procedure_texts"] = procedure_texts
claims["procedure_codes"] = procedure_codes

# --------------------------------------------------
# MAP FRAUD LABEL TO BINARY
# --------------------------------------------------
claims["fraud_label"] = claims["PotentialFraud"].map({
    "Yes": 1,
    "No": 0,
    1: 1,
    0: 0
})

# --------------------------------------------------
# BUILD FINAL STRUCTURED DATAFRAME
# --------------------------------------------------
structured_claims = claims[[
    "ClaimID",
    "Provider",
    "InscClaimAmtReimbursed",
    "fraud_label",
    "diagnosis_codes",
    "diagnosis_texts",
    "procedure_codes",
    "procedure_texts"
]].copy()

structured_claims.columns = [
    "claim_id",
    "provider_id",
    "claim_amount",
    "fraud_label",
    "diagnosis_codes",
    "diagnosis_texts",
    "procedure_codes",
    "procedure_texts"
]

print("Structured claims created:", len(structured_claims))

# --------------------------------------------------
# SAVE
# --------------------------------------------------
structured_claims.to_csv(output_path, index=False)

print("Saved cleaned structured claims to:", output_path)
print("=== DONE ===")