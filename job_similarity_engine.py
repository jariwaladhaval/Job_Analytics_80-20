# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:42:42 2026

@author: Dhaval.Jariwala
"""
#Import libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# Load Dataset
df = pd.read_csv("jobs_dataset.csv", encoding="latin1")

df["Job ID"] = df["Job ID"].astype(str)
df = df.reset_index(drop=True)

#Text Feature Engineering (Role Understanding)
TEXT_COLS = [
    "Purpose",
    "Key Responsibilities",
    "Key Deliverables",
    "Outcomes & KPIs"
]

for col in TEXT_COLS:
    if col not in df.columns:
        df[col] = ""

df["combined_text"] = (
    df[TEXT_COLS]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

#Competency Extraction
COMP_COLS = [f"Competency {i}" for i in range(1, 13)]

for col in COMP_COLS:
    if col not in df.columns:
        df[col] = np.nan

def extract_competencies(row):
    return [
        str(row[c]).strip()
        for c in COMP_COLS
        if pd.notna(row[c]) and str(row[c]).strip() != ""
    ]

df["competency_list"] = df.apply(extract_competencies, axis=1)

#NLP Embeddings (Deep Learning)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Text embeddings
text_embeddings = model.encode(
    df["combined_text"].tolist(),
    normalize_embeddings=True
)

# Unique competencies embedding
all_competencies = sorted(
    set(c for comps in df["competency_list"] for c in comps)
)

comp_embeddings = model.encode(
    all_competencies,
    normalize_embeddings=True
)

comp2vec = dict(zip(all_competencies, comp_embeddings))


#####NLP Search Addition####
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

def search_by_natural_language(query, top_k=20):
    """
    Semantic search using NLP embeddings
    """

    # Encode user query
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Compute cosine similarity with all jobs
    scores = cosine_similarity(
        query_embedding,
        text_embeddings
    )[0]

    # Build result dataframe
    results = pd.DataFrame({
        "Job ID": df["Job ID"].values,
        "Similarity %": np.round(scores * 100, 2)
    })

    # Sort & take top matches
    results = (
        results
        .sort_values("Similarity %", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    return results


#Competency Similarity (FULL CROSS-MATCH FIXED)

def competency_similarity(job_a_comps, job_b_comps):
    if not job_a_comps or not job_b_comps:
        return 0.0

    vecs_a = np.array([comp2vec[c] for c in job_a_comps])
    vecs_b = np.array([comp2vec[c] for c in job_b_comps])

    sim_matrix = cosine_similarity(vecs_a, vecs_b)

    # Best match in Job B for each competency in Job A
    best_matches = sim_matrix.max(axis=1)

    return float(best_matches.mean())

#Build Similarity Matrices

n = len(df)

# Text similarity
text_sim_matrix = cosine_similarity(text_embeddings)

# Competency similarity
comp_sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        comp_sim_matrix[i, j] = competency_similarity(
            df.loc[i, "competency_list"],
            df.loc[j, "competency_list"]
        )
        

#Fusion Strategy (Configurable)
TEXT_WEIGHT = 0.7
COMP_WEIGHT = 0.3

final_similarity = (
    TEXT_WEIGHT * text_sim_matrix +
    COMP_WEIGHT * comp_sim_matrix
)

similarity_pct = np.round(final_similarity * 100, 2)

#Explainability 
def generate_similarity_reason(job_a_idx, job_b_idx):
    reasons = []

    if text_sim_matrix[job_a_idx, job_b_idx] > 0.75:
        reasons.append("Highly similar responsibilities and outcomes")

    elif text_sim_matrix[job_a_idx, job_b_idx] > 0.5:
        reasons.append("Moderately similar responsibilities and deliverables")

    shared_skills = set(df.loc[job_a_idx, "competency_list"]) & \
                    set(df.loc[job_b_idx, "competency_list"])

    if shared_skills:
        reasons.append(f"Shared competencies: {', '.join(list(shared_skills)[:3])}")

    if comp_sim_matrix[job_a_idx, job_b_idx] > 0.7:
        reasons.append("Strong skill proficiency alignment")

    if not reasons:
        reasons.append("Limited overlap in role scope and competencies")

    return "; ".join(reasons)

#Final Output Table
records = []

for i in range(n):
    for j in range(n):
        if i == j:
            continue

        records.append({
            "Job ID": df.loc[i, "Job ID"],
            "Compared Job ID": df.loc[j, "Job ID"],
            "Similarity %": similarity_pct[i, j],
            "Text Similarity": round(text_sim_matrix[i, j], 3),
            "Competency Similarity": round(comp_sim_matrix[i, j], 3),
            "Similarity Reason": generate_similarity_reason(i, j)
        })

results_df = pd.DataFrame(records)

# Standardize similarity formatting
results_df["Similarity %"] = results_df["Similarity %"].astype(float).round(2)
results_df["Text Similarity"] = (results_df["Text Similarity"] * 100).round(2)
results_df["Competency Similarity"] = (results_df["Competency Similarity"] * 100).round(2)


results_df.rename(columns={
    "Text Similarity": "Text Similarity %",
    "Competency Similarity": "Competency Similarity %"
}, inplace=True)


results_df.to_excel("job_similarity_output_v1.xlsx", index=False)

print("✅ Job similarity file exported successfully")

# job_ids must be aligned with embedding / competency matrices
job_ids = df['Job ID'].values

# final_similarity is already computed (0–1 scale)
similarity_pct = np.round(final_similarity * 100, 2)

# Create square similarity matrix
similarity_matrix = pd.DataFrame(
    similarity_pct,
    index=job_ids,
    columns=job_ids
)

# Optional: enforce perfect diagonal
np.fill_diagonal(similarity_matrix.values, 100.0)

# Export
similarity_matrix.to_excel("job_similarity_matrix.xlsx")
