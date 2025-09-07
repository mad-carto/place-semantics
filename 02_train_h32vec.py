# -*- coding: utf-8 -*-
# Train H3-based Spatial Embeddings Using Word2Vec
"""
This script uses H3 hexagonal spatial indices and Word2Vec to learn location embeddings 
based on adjacency-based topological relationships (neighboring cells).

By treating H3 cell indices as "words" and their neighbors as "context," we apply the 
Word2Vec skip-gram model to learn vector representations of geolocations. This captures
the spatial relationships between cells in a similar way to how Word2Vec captures semantic and
syntactic relationships in language.

Output:
- Trained H32Vec model saved to disk in Gensim format.
"""

import os
from pathlib import Path
import warnings
import random

import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import pickle
from gensim.models import Word2Vec
from scipy.stats import sem
import h3  # v3.7.7

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Set PROJ_LIB for GeoPandas (for coordinate transformations, if needed)
os.environ["PROJ_LIB"] = "/envs/bertopic_env2/share/proj"

# Input and output directories
INPUT = Path.cwd() / "01_Input"
OUTPUT = Path.cwd() / "02_Output"
GEOJSON_PATH = INPUT / "DD_admin.geojson"


def geojson_to_h3_indices(geojson_path, resolution):
    """
    Convert a GeoJSON polygon or multipolygon to a set of H3 indices.

    Args:
        geojson_path (str or Path): Path to the GeoJSON file.
        resolution (int): H3 resolution level.

    Returns:
        set: A set of H3 indices covering the polygon.
    """
    gdf = gpd.read_file(geojson_path)

    if gdf.empty or "geometry" not in gdf.columns:
        raise ValueError("GeoJSON file does not contain valid geometries.")

    polygon = gdf.iloc[0].geometry
    if polygon.geom_type not in ["Polygon", "MultiPolygon"]:
        raise ValueError(f"Expected Polygon or MultiPolygon, got {polygon.geom_type}")

    geojson = {
        "type": "Polygon",
        "coordinates": [list(polygon.exterior.coords)]
    }

    return h3.polyfill(geojson, resolution, geo_json_conformant=True)


def generate_training_data(h3_indices):
    """
    Generate training samples for Word2Vec from H3 indices.

    Each sample consists of a central H3 index and its first ring of neighbors.

    Args:
        h3_indices (iterable): List or set of H3 indices.

    Returns:
        list of tuples: Each tuple contains a central H3 index and its neighbors.
    """
    training_data = []

    for h3_index in h3_indices:
        neighbors = list(h3.k_ring(h3_index, 1))
        neighbors.remove(h3_index)
        training_data.append((h3_index, *neighbors))

    return training_data


def train_word2vec_on_h3(dataset, vector_size=384, window=6, sg=1,
                         min_count=1, workers=6, epochs=10,
                         negative=10, ns_exponent=0.75):
    """
    Train a Word2Vec model using spatial H3 index sequences.

    Args:
        dataset (list of tuples): Training data of H3 index and neighbors.
        vector_size (int): Size of embedding vectors.
        window (int): Context window size.
        sg (int): Use skip-gram if 1, CBOW if 0.
        min_count (int): Minimum count threshold for words.
        workers (int): Number of parallel workers.
        epochs (int): Number of training epochs.
        negative (int): Number of negative samples.
        ns_exponent (float): Exponent used for negative sampling distribution.

    Returns:
        gensim.models.Word2Vec: Trained model.
    """
    sentences = [[str(center)] + [str(neighbor) for neighbor in neighbors]
                 for center, *neighbors in dataset]

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        sg=sg,
        min_count=min_count,
        workers=workers,
        negative=negative,
        ns_exponent=ns_exponent,
        epochs=epochs
    )

    return model


def main():
    """Main execution function."""
    # Generate H3 indices from polygon
    h3_indices_res9 = geojson_to_h3_indices(GEOJSON_PATH, resolution=9)
    print(f"Generated {len(h3_indices_res9)} unique H3 resolution 9 indices.")

    # Generate Word2Vec-compatible training data
    training_data = generate_training_data(h3_indices_res9)

    # Train the embedding model
    model = train_word2vec_on_h3(training_data)
    model_path = OUTPUT / "h32vec_res9.model"
    model.save(str(model_path))

    print(f"Model training complete. Saved to: {model_path}")


if __name__ == "__main__":
    main()
