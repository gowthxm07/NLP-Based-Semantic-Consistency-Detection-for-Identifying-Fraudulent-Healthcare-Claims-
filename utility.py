import pandas as pd

print("=== DEDUPLICATING CLAIMS (BY NATURAL LANGUAGE) ===")

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "claims_natural_language.csv"
OUTPUT_PATH = "data/claims_natural_language_dedup.csv"

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Original rows:", len(df))

# -----------------------------
# CLEAN NATURAL LANGUAGE
# -----------------------------
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = " ".join(x.split())
    return x

df["claim_natural_text_clean"] = df["claim_natural_text"].apply(clean_text)

# -----------------------------
# DROP DUPLICATES
# -----------------------------
dedup_df = df.drop_duplicates(subset=["claim_natural_text_clean"])

print("After deduplication:", len(dedup_df))
print("Removed duplicates:", len(df) - len(dedup_df))

# Drop helper column
dedup_df = dedup_df.drop(columns=["claim_natural_text_clean"])

# -----------------------------
# SAVE
# -----------------------------
dedup_df.to_csv(OUTPUT_PATH, index=False)

print("Saved to:", OUTPUT_PATH)
print("Done.")