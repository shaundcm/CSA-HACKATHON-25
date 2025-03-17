import streamlit as st
import os
from load_data import (
    load_csv, load_excel, load_pdf,
    summarize_dataframe_structure,
    get_sample_rows,
    get_eda_summary,
    filter_relevant_data,
    generate_visuals
)
from chatbot import ask_ollama
import sys
import torch

# Monkeypatch to prevent Streamlit from probing torch.classes
torch.classes.__path__ = []
sys.modules["torch.classes"].__path__ = []

# Page Setup
st.set_page_config(page_title="🧠 Smart AI Chatbot", layout="wide")
st.title("🤖 AI Data Chatbot with Memory & Semantic Filtering")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload CSV, Excel or PDF files",
    type=["csv", "xlsx", "xls", "pdf"],
    accept_multiple_files=True
)

df_dict = {}

# Load all uploaded files
for uploaded_file in uploaded_files:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == "csv":
        df_dict[uploaded_file.name] = load_csv(uploaded_file)
    elif file_type in ["xlsx", "xls"]:
        df_dict[uploaded_file.name] = load_excel(uploaded_file)
    elif file_type == "pdf":
        content = load_pdf(uploaded_file)
        st.subheader(f"📄 PDF Content: {uploaded_file.name}")
        st.text_area("Extracted Text", content[:3000], height=300)

# Select a file to work with
if df_dict:
    selected_file = st.selectbox("📂 Select a file to analyze:", list(df_dict.keys()))
    df = df_dict[selected_file]

    st.subheader(f"📈 Preview: {selected_file}")
    st.dataframe(df.head(20), use_container_width=True)

    structure = summarize_dataframe_structure(df)
    eda = get_eda_summary(df)
    sample = get_sample_rows(df)

    # Column Summary
    st.subheader("📌 Column Summary")
    st.text(structure)

    # EDA Summary
    st.subheader("📊 EDA Summary")
    if "📈 Numeric Summary" in eda:
        st.subheader("📈 Numeric Summary")
        st.dataframe(eda["📈 Numeric Summary"], use_container_width=True)

    if "🧠 Categorical Summary" in eda:
        st.subheader("🧠 Categorical Summary")
        st.dataframe(eda["🧠 Categorical Summary"], use_container_width=True)

    if "⚠️ Error" in eda:
        st.error(f"Error in generating EDA: {eda['⚠️ Error']}")

    # Visual Insights
    st.subheader("📊 Visual Insights")
    charts = generate_visuals(df)
    for path in charts:
        if "Employee ID" not in path:  # Avoid showing ID histograms
            st.image(path, caption=os.path.basename(path), use_container_width=True)

    # Chat Section
    st.subheader("💬 Ask Your Question")
    user_query = st.text_input("Type your question below:")

    if user_query:
        filtered_df, info = filter_relevant_data(df, user_query)
        filtered_rows = filtered_df.to_string(index=False)

        final_prompt = f"""
You are an intelligent data assistant.

📌 Data Structure:
{structure}

📊 EDA Summary:
{eda}

🔍 Column Match Info:
{info}

📄 Relevant Data (sample rows from matched columns):
{filtered_rows}

Your task: Use only the information provided above to answer the user's question. 
DO NOT provide code or Python syntax — just respond with the actual answer in natural language.
If you can't find the answer from the data shown, say so clearly.

Now answer this question:
{user_query}
"""

        with st.spinner("Thinking..."):
            print("DEBUG: Filtered Rows Length =", len(filtered_rows))
            response = ask_ollama(final_prompt)
        st.success("AI Answer:")
        st.write(response)

# Compare Multiple Files (Optional Feature)
if len(df_dict) >= 2:
    st.subheader("📊 Compare Two Files (Optional)")
    file1 = st.selectbox("Select First File", list(df_dict.keys()), key="f1")
    file2 = st.selectbox("Select Second File", list(df_dict.keys()), key="f2")

    if file1 != file2:
        st.write("🔍 Side-by-Side Comparison (First 10 Rows)")
        col1, col2 = st.columns(2)
        with col1:
            st.write(file1)
            st.dataframe(df_dict[file1].head(10), use_container_width=True)
        with col2:
            st.write(file2)
            st.dataframe(df_dict[file2].head(10), use_container_width=True)
