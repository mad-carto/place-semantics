# -*- coding: utf-8 -*-
# Train Geo-Topic Models using BERTopic and location Embeddings

"""

This script is part of the supplementary material for the paper:

    Assessing Place Similarity by Inferring and Modeling Place Semantics 
    from Geosocial Media Data

It combines sentence-transformers text embeddings with location embeddings trained with
the h32vec model (see script 02_train_h32vec.py) to train four BERTopic models on different
embeddings: a baseline model (TextEmb) with sentence-transformers text embeddings, and three
different models based on the GeoText embeddings computed based on 3 different weightings
schemes (lowGeoTextEmb, midGeoTextEmb, highGeoTextEmb).

Due to privacy constraints and platform data-sharing policies, the input data (the preprocessed
geosocial media dataset) used in this script cannot be made public. However, the trained topic 
models are available for reproducibility and evaluation.

"""

import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from spacy.lang.en import stop_words as en_stop_words
from spacy.lang.de import stop_words as de_stop_words
from gensim.models import KeyedVectors

# -----------------------------
# Paths and Environment Setup
# -----------------------------

FOLDER_PATH = Path.cwd()
INPUT = FOLDER_PATH / "01_Input"
OUTPUT = FOLDER_PATH / "02_Output"
FILE = OUTPUT/ "Dresden_GSMDataset_preprocessed.csv"

# -----------------------------
# Load Preprocessed Data
# -----------------------------

df = pd.read_csv(FILE)
docs = df["text"]

# -----------------------------
# Sentence Embeddings
# -----------------------------

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# -----------------------------
# Load Location Embeddings
# -----------------------------

# Load the embeddings from the binary file
h32vec_model_9 = KeyedVectors.load(f"{OUTPUT}/h32vec_res9.model")

# -----------------------------
# Generate GeoText Embeddings
# -----------------------------
# The weighting schemes are defined as tuples of (location_weight, semantic_weight
weighting_schemes = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]


def generate_geotext_embeddings(df_, text_emb, h3_model, weights):
    """
    Combine semantic and location embeddings into geotext GeoText embeddings.
    """
    geotext_dict = {}

    text_emb_norm = normalize(text_emb, norm="l2", axis=1)

    for scheme_id, (w_location, w_text) in enumerate(weights):
        
        geotext_embeddings = []

        for idx, row in df_.iterrows():
            h3_idx = str(row["h3_index_9"])
            text_vec = text_emb_norm[idx]

            if h3_idx in h3_model.wv:
                location_vec = h3_model.wv[h3_idx]
                location_vec_norm = normalize([location_vec], norm="l2")[0]

                weighted_location = w_location * location_vec_norm
                weighted_text = w_text * text_vec

                geotext = np.concatenate((weighted_text, weighted_location))
                geotext_embeddings.append(geotext)

        geotext_dict[scheme_id] = np.array(geotext_embeddings)

    return geotext_dict


geotext_embeddings = generate_geotext_embeddings(
    df, embeddings, h32vec_model_9, weighting_schemes
)

# -----------------------------
# BERTopic Parameters
# -----------------------------

# 1. Dimension Reduction (UMAP)
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=123,
    low_memory=True,
)

# 2. Clustering (HDBSCAN)
N = len(docs)

if N < 100000:
    # Use scaled-down clustering for small-to-medium datasets
    CLUSTER_SIZE = max(10, int(0.002 * N))     # ~0.2%
    MIN_SAMPLES  = max(10, int(0.0007 * N))     # ~0.05%
else:
    # Use scaled-up clustering for large datasets
    CLUSTER_SIZE = max(100, int((N * 0.0005) // 100 * 100))
    MIN_SAMPLES  = 50

print(f"[INFO] Dataset size: {N} → CLUSTER_SIZE = {CLUSTER_SIZE}, MIN_SAMPLES = {MIN_SAMPLES}")


hdbscan_model = HDBSCAN(
    min_cluster_size=CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# 3. Vectorizer
stop_words = list(en_stop_words.STOP_WORDS) + list(de_stop_words.STOP_WORDS)
vectorizer_model = CountVectorizer(
    stop_words=stop_words,
    max_df=0.8,
    min_df=5,
)

# -----------------------------
# Train TextEmb Model
# -----------------------------

print("Training Text-only BERTopic model...")

text_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=20,
    verbose=True,
    calculate_probabilities=False,
)

topics, probs = text_model.fit_transform(docs, embeddings=embeddings)

# Save the model
text_model.save(
    OUTPUT / "Topic_Models" / "TextEmb_Model",
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model=embedding_model,
)

# -----------------------------
# Train GeoTextEmb Models
# -----------------------------

print("Training Geo-Topic models...")

geo_model_paths = [
    "lowGeoTextEmb_Model",
    "midGeoTextEmb_Model",
    "highGeoTextEmb_Model"
]

geo_models = []  # Store model instances for classification

for i, model_path in enumerate(geo_model_paths):
    print(f"Training model {i} with weighting {weighting_schemes[i]}")

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language=None,
        top_n_words=20,
        verbose=True,
        calculate_probabilities=False,
    )

    _ = topic_model.fit_transform(docs, embeddings=geotext_embeddings[i])
    geo_models.append(topic_model)

    topic_model.save(
        OUTPUT / "Topic_Models" / model_path,
        serialization="safetensors",
        save_ctfidf=True,
    )

print("All topic models trained and saved successfully.")

# -----------------------------
# Classify GSM Posts Based on Topic Models
# -----------------------------

def append_topic_columns(df_, model, topic_ids, model_code):
    """
    Append topic classification metadata to the DataFrame.
    """
    df_[f"T#_{model_code}"] = topic_ids

    topic_info = model.get_topic_info().set_index("Topic")
    df_[f"TName_{model_code}"] = df_[f"T#_{model_code}"].map(topic_info["Name"])
    df_[f"TCount_{model_code}"] = df_[f"T#_{model_code}"].map(topic_info["Count"])
    df_[f"Repre_{model_code}"] = df_[f"T#_{model_code}"].map(topic_info["Representation"])

    return df_

print("Appending topic assignments to DataFrame...")

df = append_topic_columns(df, text_model, text_model.topics_, "TE")
df = append_topic_columns(df, geo_models[0], geo_models[0].topics_, "GTE0")
df = append_topic_columns(df, geo_models[1], geo_models[1].topics_, "GTE1")
df = append_topic_columns(df, geo_models[2], geo_models[2].topics_, "GTE2")

# Save the DataFrame with topic assignments
df.to_csv(OUTPUT / "Dresden_GSMDataset_classified.csv", index=False)
