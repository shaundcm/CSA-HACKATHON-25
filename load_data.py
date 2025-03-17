import pandas as pd
import pdfplumber
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer, util

# Load semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Skip plotting for these columns (usually identifiers)
SKIP_COLUMNS = ["employee id", "id", "s.no", "sno", "serial no"]

def load_csv(file):
    return pd.read_csv(file)

def load_excel(file):
    return pd.read_excel(file)

def load_pdf(file):
    text = ""
    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def summarize_dataframe_structure(df):
    summary = "📌 Column Summary\n\n"
    summary += "📌 Column Names & Data Types:\n"
    summary += "\n".join([f"- {col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    return summary

def get_sample_rows(df, n=5):
    return df.head(n).to_string(index=False)

def get_eda_summary(df):
    eda_output = {}

    try:
        # Numeric summary as a proper table
        numeric_cols = df.select_dtypes(include='number')
        if not numeric_cols.empty:
            numeric_summary = numeric_cols.describe().T.round(2)
            eda_output["📈 Numeric Summary"] = numeric_summary

        # Categorical summary as a proper table
        categorical_cols = df.select_dtypes(include='object')
        if not categorical_cols.empty:
            categorical_summary = []
            for col in categorical_cols.columns:
                top = df[col].mode()[0]
                freq = df[col].value_counts().iloc[0]
                unique = df[col].nunique()
                categorical_summary.append({
                    "Column": col,
                    "Unique Values": unique,
                    "Most Frequent": top,
                    "Frequency": freq
                })
            categorical_df = pd.DataFrame(categorical_summary)
            eda_output["🧠 Categorical Summary"] = categorical_df

    except Exception as e:
        eda_output["⚠️ Error"] = str(e)

    return eda_output

def generate_visuals(df, output_dir="eda_charts"):
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []

    # Plot numeric columns except skipped ones
    for col in df.select_dtypes(include='number').columns:
        if col.strip().lower() in SKIP_COLUMNS:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        chart_path = os.path.join(output_dir, f'{col}_hist.png')
        plt.tight_layout()
        plt.savefig(chart_path)
        chart_paths.append(chart_path)
        plt.close()

    # Plot categorical columns (only those with < 20 unique values)
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() < 20:
            value_counts = df[col].value_counts().nlargest(10)
            plt.figure(figsize=(8, 5))
            sns.barplot(x=value_counts.values, y=value_counts.index, palette='viridis')
            plt.title(f'Top Categories in {col}')
            chart_path = os.path.join(output_dir, f'{col}_bar.png')
            plt.tight_layout()
            plt.savefig(chart_path)
            chart_paths.append(chart_path)
            plt.close()

    return chart_paths

def get_similar_columns(df, query, threshold=0.4):
    query_emb = model.encode(query, convert_to_tensor=True)
    col_embeddings = model.encode(df.columns.tolist(), convert_to_tensor=True)

    similarities = util.cos_sim(query_emb, col_embeddings)[0]
    matched_cols = [(df.columns[i], float(score)) for i, score in enumerate(similarities) if score >= threshold]
    matched_cols.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in matched_cols]

def get_similar_rows(df, matched_cols, query, threshold=0.4):
    if not matched_cols:
        return df.head(30), "❌ No matching columns found. Showing default rows instead."

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure it's available here
    query_emb = model.encode(query, convert_to_tensor=True)

    similarity_scores = []

    for idx, row in df[matched_cols].iterrows():
        row_text = ' '.join(str(row[col]) for col in matched_cols)
        row_emb = model.encode(row_text, convert_to_tensor=True)
        sim_score = util.cos_sim(query_emb, row_emb).item()
        similarity_scores.append((idx, sim_score))

    # Filter top rows based on similarity
    top_rows = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:15]
    top_indices = [idx for idx, score in top_rows if score >= threshold]

    if not top_indices:
        return df.head(30), "⚠️ Columns matched, but no strongly relevant rows found. Showing general data instead."

    return df.loc[top_indices], f"✅ Matched Columns: {', '.join(matched_cols)} | Filtered {len(top_indices)} relevant rows."

def filter_relevant_data(df, query):
    matched_cols = get_similar_columns(df, query)
    filtered_df, info = get_similar_rows(df, matched_cols, query)
    return filtered_df, info
