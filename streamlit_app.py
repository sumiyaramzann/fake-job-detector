import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "mohamedsobhy/fake-job-posting-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Streamlit UI
st.title("Fake Job Detector 🤖")
text = st.text_area("Paste any job post/ad below:")

if st.button("Check"):
    if text:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()
            if prediction == 1:
                st.error("❌ This looks like a *Fake Job* post.")
            else:
                st.success("✅ This appears to be a *Real Job* post.")
    else:
        st.warning("Please enter some text first.")
