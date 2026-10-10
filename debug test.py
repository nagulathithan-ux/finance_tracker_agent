from transformers import pipeline
print("Loading model...")
dir *pyclf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Model loaded. Classifying...")
result = clf("Domino's outlet payment", candidate_labels=["Food & Dining", "Shopping", "Miscellaneous"])
print(result)       