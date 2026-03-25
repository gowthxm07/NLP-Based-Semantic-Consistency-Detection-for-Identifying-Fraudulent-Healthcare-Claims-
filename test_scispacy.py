import spacy
from scispacy.linking import EntityLinker

print("Loading model...")
nlp = spacy.load("en_core_sci_lg")

print("Adding UMLS EntityLinker...")

# Correct method for your version
linker = EntityLinker(
    resolve_abbreviations=True,
    name="umls"
)

nlp.add_pipe("entity_linker", config={
    "resolve_abbreviations": True,
    "linker_name": "umls"
})

text = """
Patient diagnosed with acute myocardial infarction
and underwent coronary artery bypass graft surgery.
"""

doc = nlp(text)

print("\nDetected Entities:")

for ent in doc.ents:
    print("Entity:", ent.text)
    if ent._.kb_ents:
        print("  UMLS concepts:", ent._.kb_ents[:1])