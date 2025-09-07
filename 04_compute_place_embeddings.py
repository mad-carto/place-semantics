# -*- coding: utf-8 -*-´
# Compute Place Embeddings for Place-Based Semantic Search

"""

This script computes post-level embeddings and H3 cell-level place embeddings using topic 
assignments from a trained Geo-BERTopic model. Embeddings are stored in SQLite databases 
for use in downstream place-based semantic search and similarity analysis.

The model used for computing the embeddings is the `midGeoTextEmb` BERTopic model,
which combines semantic and spatial information with a weighting scheme of (0.3, 0.7).
This configuration was selected based on evaluation results detailed in the accompanying
Jupyter notebook `Evaluation.ipynb`.

Data availability:

The original geosocial media data used in this study, comprising geotagged posts from 
Instagram, Flickr, and X (formerly Twitter), cannot be publicly shared. All data
utilized in this study was collected through official APIs or authorized services
that, at the time of collection, explicitly prohibited the redistribution of 
user-generated content in accordance with their respective terms of service
(e.g., the Twitter Developer Agreement and Instagram's API Terms of Use). 
While these specific agreements are no longer publicly available due to platform 
ownership changes and service restructuring, their restrictions were in effect and
adhered to during the data acquisition period.

"""

import os
import sqlite3
import numpy as np
import pandas as pd
import torch

from collections import Counter
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# -----------------------------
# Paths
# -----------------------------

FOLDER_PATH = os.getcwd()
INPUT = os.path.join(FOLDER_PATH, "01_Input")
OUTPUT = os.path.join(FOLDER_PATH, "02_Output")

# -----------------------------
# Load Classified Data and Model
# -----------------------------

df = pd.read_csv(os.path.join(OUTPUT, "Dresden_GSMDataset_classified.csv"))
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model_path = os.path.join(OUTPUT, "Topic_Models", "midGeoTextEmb_Model")
topic_model = BERTopic.load(model_path)
topic_embeddings = topic_model.topic_embeddings_

# -----------------------------
# Build Topic Embedding Dictionary
# -----------------------------

topic_embedding_dict = {-1: topic_embeddings[0]}  # Outlier
topic_embedding_dict.update({i: emb for i, emb in enumerate(topic_embeddings[1:], start=0)})

# -----------------------------
# Compute Post Embeddings
# -----------------------------

def compute_post_embeddings(data, topic_col, topic_dict):
    post_embeds = {}
    for _, row in data.iterrows():
        post_guid = row["post_guid"]
        topic = row[topic_col]
        post_embeds[post_guid] = topic_dict[topic]
    return post_embeds

post_embeddings = compute_post_embeddings(df, "T#_GTE1", topic_embedding_dict)

# -----------------------------
# Save Post Embeddings to SQLite
# -----------------------------

def save_post_embeddings_to_db(db_path, table_name, embeddings_dict):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                post_guid TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)

        for post_guid, embedding in embeddings_dict.items():
            cursor.execute(f"""
                INSERT OR REPLACE INTO {table_name} (post_guid, embedding)
                VALUES (?, ?)
            """, (post_guid, sqlite3.Binary(np.array(embedding).tobytes())))

        conn.commit()
        print(f"Post embeddings saved successfully to table: {table_name}")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()

post_db_path = os.path.join(OUTPUT, "PostsEmbeddings.db")
save_post_embeddings_to_db(post_db_path, "post_embeddings", post_embeddings)

# -----------------------------
# Compute Document Embeddings (per H3 cell)
# -----------------------------

def compute_document_embeddings(data, topic_col, topic_dict, method):
    doc_embeds = {}
    sum_exp = len(data)
    total_counts = Counter(data[topic_col])

    for h3_id, group in data.groupby("h3_index_9"):
        exp = len(group)
        local_counts = Counter(group[topic_col])
        topics, counts = zip(*local_counts.items())
        embeddings = [topic_dict[t] for t in topics]

        if method == "mean_pooling":
            embed_tensor = torch.tensor(embeddings, dtype=torch.float32)
            doc_embeds[h3_id] = torch.mean(embed_tensor, dim=0).numpy()

        elif method == "chi_weighted_mean_pooling":
            chi_vals = []
            for i, topic in enumerate(topics):
                obs = counts[i]
                sum_obs = total_counts[topic]
                chi_val = ((obs * (sum_exp / sum_obs)) - exp) / np.sqrt(exp)
                chi_vals.append(chi_val)

            if len(set(chi_vals)) == 1:
                scaled = [1] * len(chi_vals)
            else:
                min_chi, max_chi = min(chi_vals), max(chi_vals)
                scaled = [(c - min_chi) / (max_chi - min_chi) for c in chi_vals]

            weights = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1)
            embed_tensor = torch.tensor(embeddings, dtype=torch.float32)
            weighted_sum = torch.sum(embed_tensor * weights, dim=0)
            total_weight = torch.sum(weights)

            if total_weight > 0:
                doc_embeds[h3_id] = (weighted_sum / total_weight).numpy()
            else:
                doc_embeds[h3_id] = torch.zeros_like(embed_tensor[0]).numpy()

        else:
            raise ValueError("Unknown method. Choose from 'mean_pooling' or 'chi_weighted_mean_pooling'.")

    return doc_embeds

# -----------------------------
# Save Document Embeddings
# -----------------------------

def save_document_embeddings_to_sqlite(doc_embeds, db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            h3_index TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)

    for h3_id, embedding in doc_embeds.items():
        cursor.execute(f"""
            INSERT OR REPLACE INTO {table_name} (h3_index, embedding)
            VALUES (?, ?)
        """, (h3_id, sqlite3.Binary(np.array(embedding).tobytes())))

    conn.commit()
    conn.close()

def save_multiple_document_embeddings_to_sqlite(embed_dicts, db_path):
    for name, embeddings in embed_dicts.items():
        save_document_embeddings_to_sqlite(embeddings, db_path, name)

# -----------------------------
# Generate and Save All Place Embeddings
# -----------------------------

print("Computing document (place) embeddings...")

doc_embeddings_mean = compute_document_embeddings(df, "T#_GTE1", topic_embedding_dict, method="mean_pooling")
doc_embeddings_chi = compute_document_embeddings(df, "T#_GTE1", topic_embedding_dict, method="chi_weighted_mean_pooling")

place_db_path = os.path.join(OUTPUT, "PlaceEmbeddings.db")

embedding_variants = {
    "mean_pooling": doc_embeddings_mean,
    "chi_weighted_mean_pooling": doc_embeddings_chi,
}

save_multiple_document_embeddings_to_sqlite(embedding_variants, place_db_path)

print("All place embeddings saved.")
