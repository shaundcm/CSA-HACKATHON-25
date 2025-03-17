# CSA-HACKATHON-25
Repository for Computational Sciences Hackathon - 1

Problem statement - 3 : AI - Powered Chatbot

Overview:

A simple AI chatbot that helps users explore structured data (CSV, Excel, PDF) through natural language. It answers queries, summarizes insights, and visualizes patterns—all in a user-friendly interface.

Key Features:

- Upload CSV/Excel/PDF files
- Auto EDA summary and charts
- Natural language Q&A with multi-turn memory
- Semantic column matching using NLP
- Multi-file upload and comparison

Tech Stack:

- Python for simplicity and ecosystem support
- Streamlit for quick and interactive UI
- Pandas for handling tabular data
- Sentence-Transformers (MiniLM) for semantic matching
- Ollama + Mistral 7B for local AI responses

Improvements Ahead:

- Code-execution-based responses
- Cloud deployment & authentication
- Advanced analytics & visualizations

Setup:

1. Install dependencies

    `pip install -r requirements.txt`

2. Install Ollama (https://ollama.com)

3. Run the Mistral model locally:

    `ollama run mistral`

4. Start the app:

    `streamlit run app.py`

Usage:

1. Upload your dataset (CSV, Excel, or PDF)

2. Explore automatic summaries and visualizations

3. Ask questions like:

    - “How many employees completed training in 2023?”
    - “Show average salary by department”
    - “What’s the most common training outcome?”
