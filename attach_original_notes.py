import pandas as pd
from pathlib import Path

print("=== ATTACHING ORIGINAL NOTES TO TRAIN/VAL/TEST ===")

ROOT = Path.home() / "medical_nlp"

NOTES_PATH = ROOT / "data" / "structured_notes.csv"

TRAIN_PATH = ROOT / "data/train/train.csv"
VAL_PATH = ROOT / "data/val/val.csv"
TEST_PATH = ROOT / "data/test/test.csv"

OUTPUT_TRAIN = ROOT / "data/train/train_with_notes.csv"
OUTPUT_VAL = ROOT / "data/val/val_with_notes.csv"
OUTPUT_TEST = ROOT / "data/test/test_with_notes.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

notes = pd.read_csv(NOTES_PATH)
train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print("Notes:", len(notes))
print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

# --------------------------------------------------
# MERGE USING NOTE_ID
# --------------------------------------------------

notes = notes.rename(columns={"patient_id":"patient_id","note_text":"original_note"})

train = train.merge(
    notes[["patient_id","original_note"]],
    how="left",
    left_on="note_id",
    right_index=True
)

val = val.merge(
    notes[["patient_id","original_note"]],
    how="left",
    left_on="note_id",
    right_index=True
)

test = test.merge(
    notes[["patient_id","original_note"]],
    how="left",
    left_on="note_id",
    right_index=True
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

train.to_csv(OUTPUT_TRAIN,index=False)
val.to_csv(OUTPUT_VAL,index=False)
test.to_csv(OUTPUT_TEST,index=False)

print("Saved datasets with original notes.")