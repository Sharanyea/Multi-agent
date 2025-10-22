import streamlit as st
import requests

st.title("🩺 Multi-Agent Breast Cancer Diagnosis System")

uploaded_file = st.file_uploader("Upload a mammogram image")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if st.button("Run Diagnosis"):
    try:
        response = requests.get("http://localhost:8000/diagnose").json()
        st.subheader("Diagnosis Results")
        st.json(response)
    except Exception as e:
        st.error(f"Error: {e}")
