from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf(text, filename):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = [Paragraph(text.replace("\n", "<br/>"), styles["Normal"])]
    doc.build(story)


data = {
    "note1.pdf": """Discharge Summary:

Patient Name: John Doe
Age: 68 years
Gender: Male

Chief Complaint:
The patient presented with chest pain and shortness of breath for the past 2 days.

History of Present Illness:
The patient is a known case of hypertension and coronary artery disease. He reported 
increasing chest discomfort radiating to the left arm. No history of syncope. No history of diabetes mellitus.

Past Medical History:
Hypertension (10 years), Coronary artery disease (5 years)

Investigations:
ECG showed ST depression. Troponin levels were mildly elevated.

Hospital Course:
The patient was taken to the cardiac catheterization lab and underwent coronary angioplasty 
with stent placement. The procedure was successful with no immediate complications.

Medications:
Aspirin, Clopidogrel, Beta-blockers

Discharge Condition:
Stable. Advised follow-up in cardiology OPD.""",
    "claim1.pdf": """The patient was diagnosed with hypertension and coronary artery disease and underwent angioplasty.""",

    "note2.pdf": """Clinical Note:

Patient Name: Jane Smith
Age: 55 years
Gender: Female

Chief Complaint:
Frequent urination, increased thirst, and fatigue.

History of Present Illness:
The patient presented with symptoms suggestive of metabolic imbalance. Blood glucose 
levels were significantly elevated on admission.

Diagnosis:
Type 2 Diabetes Mellitus

Past History:
No history of cardiac disease or hypertension.

Treatment:
The patient was started on insulin therapy. Dietary modifications and lifestyle 
changes were advised. Blood glucose was monitored regularly.

Hospital Course:
The patient responded well to insulin therapy with improved glucose levels.

Discharge Plan:
Continue insulin therapy and follow up with endocrinology.""",
    "claim2.pdf": """The patient was diagnosed with type 2 diabetes mellitus and received insulin therapy.""",

    "note3.pdf": """Discharge Summary:

Patient Name: Robert Lee
Age: 60 years
Gender: Male

Chief Complaint:
Generalized weakness and swelling of lower limbs.

History of Present Illness:
The patient was diagnosed with chronic kidney disease. He reported fatigue and fluid retention.
No complaints of chest pain or cardiovascular symptoms.

Past Medical History:
Chronic kidney disease (Stage 3)

Investigations:
Elevated creatinine levels. Ultrasound showed reduced kidney size.

Treatment:
The patient underwent dialysis sessions and received supportive therapy.

Hospital Course:
The patient improved with dialysis. No cardiac procedures were performed.

Important Notes:
No evidence of hypertension or coronary artery disease was found.

Discharge Condition:
Stable.""",
    "claim3.pdf": """The patient was diagnosed with hypertension and underwent angioplasty.""",

    "note4.pdf": """Clinical Note:

Patient Name: Emily Davis
Age: 45 years
Gender: Female

Chief Complaint:
Severe headache and dizziness.

History of Present Illness:
The patient presented with recurrent episodes of headache associated with light sensitivity.

Diagnosis:
Migraine

Past Medical History:
No history of diabetes, hypertension, or cardiovascular disease.

Treatment:
The patient was treated with analgesics and advised rest. No invasive procedures were performed.

Hospital Course:
Symptoms improved with medication.

Important Notes:
No surgical interventions such as bypass surgery were performed.

Discharge Plan:
Follow-up if symptoms persist.""",
    "claim4.pdf": """The patient underwent bypass surgery for coronary artery disease."""
}

for filename, text in data.items():
    create_pdf(text, filename)

print("✅ PDFs generated")