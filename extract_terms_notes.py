import pandas as pd
import spacy
import re
from tqdm import tqdm

print("=== 3-LAYER CLINICAL EXTRACTION PIPELINE (BATCH OPTIMIZED) ===")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_PATH = "data/structured_notes.csv"
OUTPUT_PATH = "data/note_entities.csv"
BATCH_SIZE = 32   # adjust if needed

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_PATH)

print("Loaded notes:", len(df))
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# IDENTIFY TEXT COLUMN
# --------------------------------------------------
if "note_text" in df.columns:
    TEXT_COL = "note_text"
elif "TEXT" in df.columns:
    TEXT_COL = "TEXT"
else:
    raise ValueError("No note text column found.")

# --------------------------------------------------
# LOAD BIOMEDICAL NER MODEL
# --------------------------------------------------
print("Loading biomedical NER model...")
nlp = spacy.load("en_ner_bc5cdr_md")

# --------------------------------------------------
# LAYER 1 — SECTION FILTERING
# --------------------------------------------------
IMPORTANT_SECTIONS = [
    "discharge diagnosis",
    "final diagnosis",
    "assessment",
    "hospital course",
    "procedures"
]

def extract_relevant_sections(text):
    text_lower = text.lower()
    extracted_text = ""

    for section in IMPORTANT_SECTIONS:
        if section in text_lower:
            start = text_lower.find(section)
            extracted_text += text[start:start+4000]

    return extracted_text if extracted_text.strip() else text


# --------------------------------------------------
# LAYER 2 — NEGATION + HISTORY FILTERING
# --------------------------------------------------
NEGATION_WORDS = [
    "no",
    "not",
    "without",
    "denies",
    "negative for",
    "no evidence of"
]

HISTORICAL_PATTERNS = [
    "family history",
    "past medical history",
    "prior history"
]

def is_negated(entity, sentence):
    sentence_lower = sentence.lower()
    entity_lower = entity.lower()

    for neg in NEGATION_WORDS:
        if neg in sentence_lower:
            if sentence_lower.find(neg) < sentence_lower.find(entity_lower):
                return True
    return False

def is_historical(sentence):
    sentence_lower = sentence.lower()
    return any(pattern in sentence_lower for pattern in HISTORICAL_PATTERNS)


# --------------------------------------------------
# LAYER 3 — PROCEDURE EXTRACTION
# --------------------------------------------------
PROCEDURE_PATTERNS = [
    r"underwent ([a-zA-Z\s\-]+)",
    r"performed ([a-zA-Z\s\-]+)",
    r"status post ([a-zA-Z\s\-]+)",
    r"s/p ([a-zA-Z\s\-]+)",
    r"taken to the operating room for ([a-zA-Z\s\-]+)"
]

COMMON_PROCEDURE_WORDS = [
    "surgery", "biopsy", "angioplasty",
    "bypass", "graft", "resection",
    "intubation", "dialysis",
    "transplant", "replacement"
]

def extract_procedures(text):
    procedures = []
    text_lower = text.lower()

    for pattern in PROCEDURE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        procedures.extend(matches)

    for word in COMMON_PROCEDURE_WORDS:
        if word in text_lower:
            procedures.append(word)

    return list(set([p.strip() for p in procedures]))


# --------------------------------------------------
# PREPROCESS TEXTS (SECTION FILTER FIRST)
# --------------------------------------------------
print("Preprocessing sections...")
texts = [
    extract_relevant_sections(str(text))
    for text in df[TEXT_COL].fillna("")
]

# --------------------------------------------------
# BATCH PROCESS WITH PROGRESS BAR
# --------------------------------------------------
results = []

print("Running NER in batches...")

for idx, doc in enumerate(
    tqdm(nlp.pipe(texts, batch_size=BATCH_SIZE), total=len(texts))
):

    diagnoses = []
    drugs = []

    for ent in doc.ents:

        if ent.label_ not in ["DISEASE", "CHEMICAL"]:
            continue

        sentence = ent.sent.text

        if is_historical(sentence):
            continue

        if is_negated(ent.text, sentence):
            continue

        if ent.label_ == "DISEASE":
            diagnoses.append(ent.text.lower())

        if ent.label_ == "CHEMICAL":
            drugs.append(ent.text.lower())

    procedures = extract_procedures(texts[idx])

    results.append({
        "note_id": idx,
        "diagnoses": list(set(diagnoses)),
        "procedures": procedures,
        "drugs": list(set(drugs))
    })

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved extracted entities to:", OUTPUT_PATH)
print("Done.")