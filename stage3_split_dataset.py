import pandas as pd
from pathlib import Path
import numpy as np

print("=== STAGE 3: TRAIN / VAL / TEST SPLIT ===")

ROOT = Path.home() / "medical_nlp"

CONSISTENT_PATH = ROOT / "pairs" / "consistent.csv"
INCONSISTENT_PATH = ROOT / "pairs" / "inconsistent.csv"

DATA_DIR = ROOT / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

TRAIN_DIR.mkdir(parents=True,exist_ok=True)
VAL_DIR.mkdir(parents=True,exist_ok=True)
TEST_DIR.mkdir(parents=True,exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

consistent=pd.read_csv(CONSISTENT_PATH)
inconsistent=pd.read_csv(INCONSISTENT_PATH)

print("Consistent rows:",len(consistent))
print("Inconsistent rows:",len(inconsistent))

# --------------------------------------------------
# SPLIT CLAIMS
# --------------------------------------------------

claims=consistent["claim_index"].unique()

np.random.seed(42)
np.random.shuffle(claims)

TRAIN_CLAIMS=12000
VAL_CLAIMS=3500

train_claims=claims[:TRAIN_CLAIMS]
val_claims=claims[TRAIN_CLAIMS:TRAIN_CLAIMS+VAL_CLAIMS]
test_claims=claims[TRAIN_CLAIMS+VAL_CLAIMS:]

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

train_pos=consistent[consistent["claim_index"].isin(train_claims)]
train_neg=inconsistent[inconsistent["claim_index"].isin(train_claims)]

train_neg=train_neg.sample(
    n=min(len(train_neg),len(train_pos)),
    random_state=42
)

train=pd.concat([train_pos,train_neg]).sample(frac=1,random_state=42)

# --------------------------------------------------
# VAL
# --------------------------------------------------

val_pos=consistent[consistent["claim_index"].isin(val_claims)]
val_neg=inconsistent[inconsistent["claim_index"].isin(val_claims)]

val_neg=val_neg.sample(
    n=min(len(val_neg),len(val_pos)),
    random_state=42
)

val=pd.concat([val_pos,val_neg]).sample(frac=1,random_state=42)

# --------------------------------------------------
# TEST (natural distribution)
# --------------------------------------------------

test_pos=consistent[consistent["claim_index"].isin(test_claims)]
test_neg=inconsistent[inconsistent["claim_index"].isin(test_claims)]

test=pd.concat([test_pos,test_neg]).sample(frac=1,random_state=42)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

train.to_csv(TRAIN_DIR/"train.csv",index=False)
val.to_csv(VAL_DIR/"val.csv",index=False)
test.to_csv(TEST_DIR/"test.csv",index=False)

print("\nTRAIN:",len(train))
print("VAL:",len(val))
print("TEST:",len(test))