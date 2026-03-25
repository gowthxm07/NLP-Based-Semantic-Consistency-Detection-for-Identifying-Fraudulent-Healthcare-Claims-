import torch
import re
import spacy
import pdfplumber
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path.home() / "medical_nlp"
MODEL_PATH = ROOT / "results/model"
DATA_PATH = ROOT / "data"

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("Loading models...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

nlp = spacy.load("en_ner_bc5cdr_md")
embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

print("All models loaded.\n")

# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()

# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --------------------------------------------------
# SECTION FILTERING (NOTES)
# --------------------------------------------------

IMPORTANT_SECTIONS = ["diagnosis", "history", "treatment", "procedure"]

def extract_sections(text):
    t = text.lower()
    result = ""

    for sec in IMPORTANT_SECTIONS:
        if sec in t:
            idx = t.find(sec)
            result += t[idx:idx+1000]

    return result if result else text

# --------------------------------------------------
# NEGATION
# --------------------------------------------------

NEG_WORDS = ["no", "not", "without", "denies", "negative for"]

def is_negated(entity, sentence):
    sentence = sentence.lower()
    for neg in NEG_WORDS:
        if neg in sentence and sentence.find(neg) < sentence.find(entity):
            return True
    return False

# --------------------------------------------------
# NOTE ENTITY EXTRACTION
# --------------------------------------------------

def extract_note_entities(text):

    doc = nlp(text)

    diagnoses = []
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            if not is_negated(ent.text, ent.sent.text):
                diagnoses.append(ent.text.lower())

    # fallback keywords
    fallback = ["diabetes", "hypertension", "migraine", "kidney", "coronary"]
    for f in fallback:
        if f in text:
            diagnoses.append(f)

    # procedures
    procedures = []
    sentences = text.split(".")

    for sent in sentences:
        sent = sent.lower()

        if any(n in sent for n in NEG_WORDS):
            continue

        if "underwent" in sent or "performed" in sent:
            procedures.append(sent.strip())

        if "surgery" in sent or "dialysis" in sent or "angioplasty" in sent:
            procedures.append(sent.strip())

    return list(set(diagnoses)), list(set(procedures))

# --------------------------------------------------
# CLAIM EXTRACTION (NO HARDCODING)
# --------------------------------------------------

def extract_claim_terms(claim):

    doc = nlp(claim)

    diagnoses = []
    procedures = []

    # ----------- DIAGNOSIS EXTRACTION -----------

    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            diagnoses.append(ent.text.lower())

    # pattern: "diagnosed with X"
    pattern_diag = re.findall(r"diagnosed with ([a-zA-Z0-9\s\-]+)", claim.lower())
    diagnoses.extend(pattern_diag)

    # ----------- PROCEDURE EXTRACTION -----------

    pattern_proc = re.findall(
        r"(underwent|received|performed)\s([a-zA-Z0-9\s\-]+)",
        claim.lower()
    )

    for _, proc in pattern_proc:
        procedures.append(proc.strip())

    # also capture standalone procedure words
    doc_tokens = [token.text.lower() for token in doc]

    for word in doc_tokens:
        if word in ["angioplasty", "bypass", "dialysis", "surgery"]:
            procedures.append(word)

    return list(set(diagnoses)), list(set(procedures))

# --------------------------------------------------
# COVERAGE
# --------------------------------------------------

def compute_coverage(claim_terms, note_terms):

    if not claim_terms:
        return 0

    match = 0

    for c in claim_terms:
        for n in note_terms:
            if fuzz.token_set_ratio(c, n) > 70:
                match += 1
                break

    return match / len(claim_terms)

# --------------------------------------------------
# SIMILARITY
# --------------------------------------------------

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarity(claim_terms, note_terms):

    if not claim_terms or not note_terms:
        return 0

    c_emb = embed_model.encode(claim_terms)
    n_emb = embed_model.encode(note_terms)

    sims = []

    for c in c_emb:
        for n in n_emb:
            sims.append(cosine_sim(c, n))

    return max(sims)

# --------------------------------------------------
# BUILD PREMISE
# --------------------------------------------------

def build_premise(diagnoses, procedures):
    return f"""
NOTE_DIAGNOSES:
{"; ".join(diagnoses)}

NOTE_PROCEDURES:
{"; ".join(procedures)}
"""

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

print("\n=== FINAL ROBUST REAL-TIME SYSTEM ===\n")

while True:

    note_file = input("\nEnter NOTE PDF: ")
    claim_file = input("Enter CLAIM PDF: ")

    note_text = extract_text_from_pdf(DATA_PATH / note_file)
    claim_text = extract_text_from_pdf(DATA_PATH / claim_file)

    print("\n--- RAW NOTE ---")
    print(note_text[:300])

    print("\n--- RAW CLAIM ---")
    print(claim_text)

    # preprocessing
    note_text = extract_sections(clean_text(note_text))
    claim_text = clean_text(claim_text)

    # extraction
    note_diag, note_proc = extract_note_entities(note_text)
    claim_diag, claim_proc = extract_claim_terms(claim_text)

    print("\nNOTE DIAG:", note_diag)
    print("NOTE PROC:", note_proc)

    print("\nCLAIM DIAG:", claim_diag)
    print("CLAIM PROC:", claim_proc)

    # coverage
    coverage = compute_coverage(
        claim_diag + claim_proc,
        note_diag + note_proc
    )

    # similarity
    diag_sim = compute_similarity(claim_diag, note_diag)
    proc_sim = compute_similarity(claim_proc, note_proc)

    final_score = 0.6 * diag_sim + 0.4 * proc_sim

    print("\nCoverage:", round(coverage, 3))
    print("Final score:", round(final_score, 3))

    # filtering
    if coverage < 0.2:
        print("❌ LOW COVERAGE")
        continue

    if final_score < 0.3:
        print("❌ LOW SIMILARITY")
        continue

    # model
    premise = build_premise(note_diag, note_proc)

    inputs = tokenizer(
        premise,
        claim_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits).item()

    print("\nFINAL:", "SUPPORTED ✅" if pred == 1 else "NOT SUPPORTED ❌")