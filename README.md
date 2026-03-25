# NLP-Based-Semantic-Consistency-Detection-for-Identifying-Fraudulent-Healthcare-Claims-


📌 Overview

This project presents an automated healthcare claim verification system that determines whether an insurance claim is supported or not supported by clinical notes.

The system combines:

Biomedical Named Entity Recognition (NER)
Semantic similarity search (FAISS + embeddings)
Natural Language Inference (BioLinkBERT)

to build an intelligent pipeline for verifying claim consistency against medical records.

🎯 Objective

Healthcare systems process thousands of claims daily. Manual verification is:

❌ Time-consuming
❌ Error-prone
❌ Difficult to scale

This project aims to:

✔ Automate claim verification
✔ Detect inconsistencies between claims and clinical notes
✔ Reduce fraud and improve efficiency

⚙️ System Pipeline
Claim + Clinical Note
        ↓
Data Preprocessing
        ↓
Entity Extraction (NER + Rules)
        ↓
Semantic Embeddings
        ↓
FAISS Similarity Search
        ↓
Coverage & Similarity Filtering
        ↓
BioLinkBERT (NLI Model)
        ↓
Prediction (Supported / Not Supported)
🧠 Key Features
🔬 Biomedical NER using spaCy (en_ner_bc5cdr_md)
🔍 Semantic similarity using Sentence Transformers
⚡ Fast retrieval using FAISS
🤖 Transformer-based reasoning using BioLinkBERT
📄 Real-time PDF-based claim verification
📊 Baseline comparison (cosine similarity + no NER model)
📈 Evaluation using Accuracy, F1, ROC-AUC, BERTScore
📂 Dataset
1. MIMIC-III Clinical Notes
Source: ICU patient records
Table used: NOTEEVENTS
Size: ~2M+ records
Key columns:
SUBJECT_ID
HADM_ID
CATEGORY
TEXT

Contains:

Discharge summaries
Physician notes
Treatment details
2. Healthcare Claims Dataset
Size: ~558K records
Contains:
Diagnosis codes
Procedure codes
Claim amounts
Fraud labels

Converted into:
👉 Natural language claim statements

🔄 Data Preprocessing
**Claims**
Code → text conversion
Abbreviation expansion
Text normalization
Natural language transformation
Clinical Notes
Cleaning (remove noise, headers)
Section filtering:
**Diagnosis**
Treatment
Procedures
NER extraction:
Diseases
Procedures
Negation removal
🔗 Pair Generation
Sentence embeddings generated
FAISS used for similarity search
Top-K candidate notes retrieved
Filtering using:
Coverage score
Similarity threshold


🤖 Model
BioLinkBERT (NLI)
Pretrained biomedical transformer
Fine-tuned for binary classification:
1 → Supported
0 → Not Supported

Project Structure
medical_nlp/
│
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── note1.pdf ...
│   ├── claim1.pdf ...
│
├── scripts/
│   ├── train.py
│   ├── train_biolinkbert_no_ner.py
│   ├── taking_bert_score.py
│   ├── real_time_pipeline.py
│   ├── baseline_real_time.py
│
├── results/
│   ├── model/
│   ├── plots/
│   ├── metrics/
│
└── README.md


Requirements

Install dependencies:
pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn
pip install spacy sentence-transformers faiss-cpu rapidfuzz pdfplumber bert-score
python -m spacy download en_ner_bc5cdr_md


▶️ How to Run (Step-by-Step)
1. Clone the repository
git clone https://github.com/your-username/medical-claim-verification.git
cd medical-claim-verification
2. Create a virtual environment (recommended)
python -m venv venv
Activate:
Linux/Mac: source venv/bin/activate
Windows: venv\Scripts\activate
3. Install dependencies
pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn
pip install spacy sentence-transformers faiss-cpu rapidfuzz pdfplumber bert-score
4. Download spaCy biomedical model
python -m spacy download en_ner_bc5cdr_md
5. Prepare dataset
Place files inside medical_nlp/data/
Train/val/test CSV files
PDF files (e.g., note1.pdf, claim1.pdf)
6. Train proposed model (with NER pipeline)
python scripts/train.py
Output saved in: results/model/
7. Train baseline model (without NER)
python scripts/train_biolinkbert_no_ner.py
Output saved in: results_baseline_no_ner/model/
8. Compute BERTScore
python data/taking_bert_score.py
Outputs semantic similarity scores
9. Run main real-time system (recommended)
python scripts/real_time_pipeline.py
Input:
NOTE PDF filename
CLAIM PDF filename
Output:
Prediction (Supported / Not Supported)
Confidence score
10. Run baseline real-time system
python scripts/baseline_real_time.py
Runs without preprocessing (for comparison)
11. View results
Check:
results/plots/ → graphs (ROC, PR, confusion matrix)
results/metrics/ → evaluation reports
12. Quick run (if model already trained)
python scripts/real_time_pipeline.py
