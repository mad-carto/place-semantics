[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17073248.svg)](https://doi.org/10.5281/zenodo.17073248)

This repository accompanies the study:
# From Public Discourse to Place Semantics: Modeling Semantic Similarity of Places using Geosocial Media Data
to be published in the International Journal of Geographic Information Science

**Authors: Madalina Gugulica & Dirk Burghardt**

## Abstract
Understanding how people describe, experience, and attach meaning to places is central to advancing human-centered approaches in GIScience. This study introduces a data-driven framework for inferring and modeling place semantics from geosocial media discourse to assess semantic similarity between urban places. Place semantics are conceptualized as distributions of latent topics extracted via transformer-based topic modeling, capturing cognitive-affective, socio-cultural, and functional dimensions of meaning. A key methodological innovation is the integration of geospatial context into topic modeling through geo-text feature engineering. This involves constructing joint vector representations that combine sentence-transformers with spatial information encoded using H3 indexing and Word2Vec-based location embeddings. This enables topic extraction across varying semantic and spatial granularities, enhancing geographic specificity. Topic distributions are aggregated into geospatially indexed place embeddings to support semantic comparison. The framework was implemented in a place-based semantic search system featuring thematic querying and similarity-based place retrieval. Applied to geosocial media data from Flickr, Instagram, and X for the city of Dresden, results show that geospatial context improves topic resolution, reveals both localized and distributed place semantics, and enables exploration of place similarity. This study advances place-based GIScience by bridging natural language modeling and spatial analysis through scalable, human-centered tools for understanding urban places.

## Prerequisites
1. Hardware Requirements
 - At least 16GB RAM recommended (especially for embedding and topic modeling)
 - Sufficient disk space (the trained models and embeddings can require several GB)
2. Software Requirements
 - Python 3.9 or higher
 - Conda for environment management (The provided environment.yml specifies the necessary Python dependencies)

All required Python packages and their versions are specified in the environment.yml. Key libraries include:
    - H3 for spatial indexing
    - Gensim for training the H32Vec model
    - BERTopic for topic modeling
    - UMAP and HDBSCAN for dimensionality reduction and clustering
    - Pandas, NumPy, SciPy, Matplotlib, Seaborn for data manipulation and visualization
    - scikit-learn and NLTK for additional preprocessing and modeling tasks
    - SQLite for database management of embeddings

## Data availability:
The original geosocial media data used in this study, comprising GSM posts from Instagram, Flickr, and X (formerly Twitter), cannot be publicly shared. All data were collected via official APIs or authorized services that, at the time of collection, explicitly prohibited redistribution of user-generated content, in accordance with their respective terms of service (e.g., the Twitter Developer Agreement and Instagram’s API Terms of Use). Although these agreements are no longer publicly accessible due to platform ownership changes and service restructuring, the restrictions were in effect and strictly followed during the data acquisition period. 

To support methodological transparency and facilitate replication of the methodology, we provide a representative subset of openly available Flickr data extracted from the Yahoo Flickr Creative Commons 100 Million (YFCC100M) dataset (Thomee et al., 2016). This subset is restricted to geotagged posts for the city of Dresden and is used to demonstrate the functionality of our codebase and methodological framework.

**Reference:**
 _Thomee, B., Shamma, D. A., Friedland, G., Elizalde, B., Ni, K., Poland, D., Borth, D., & Li, L.-J. (2016). YFCC100M: The New Data in Multimedia Research. https://doi.org/10.1145/2812802_
    
## Materials
This repository provides the materials to reproduce the results and visualizations from the paper as follows:

1. **Code**
- 01_preprocessing_gsm_data.py: Preprocesses raw geosocial media data (text cleaning and H3 indexing).
- 02_train_h32vec.py: Trains the H32Vec location embedding model.
- 03_train_topic_models.py: Trains BERTopic models for topic extraction.
- 04_compute_place_embeddings.py: Computes post and place embeddings for place-based semantic search and visualization.

2. **Jupyter Notebooks**
- Evaluation.ipynb:
  - Computes evaluation metrics (topic coherence, diversity, spatial dispersion).
  - Generates geographic dispersion (relative entropy) visualizations.
- TopicQueryDemo_PlaceBasedSemanticSearch.ipynb:
  - Demonstrates free-text topic queries using precomputed embeddings.
- PlaceQueryDemo_PlaceBasedSemanticSearch.ipynb:
  - Demonstrates semantic similarity search between places.

3. **Data**
The original geosocial media data used in this study—comprising geotagged posts from Instagram, Flickr, and X (formerly Twitter)—cannot be publicly shared. All data were collected via official APIs or authorized services, which, at the time of collection, explicitly prohibited redistribution of user-generated content in accordance with their respective terms of service (e.g., the Twitter Developer Agreement and Instagram’s API Terms of Use). Although these agreements are no longer publicly available due to platform changes, their restrictions were strictly followed during data acquisition.

To support reproducibility:
- A sample dataset based on Flickr CC-BY data for Dresden from the YFCC100M (Thomee et al., 2016) dataset (YFCC_Dresden_Subset.csv) is provided in the folder 01_Input
- This dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
- Please cite the original reference when using the data: _Thomee, B., Shamma, D. A., Friedland, G., Elizalde, B., Ni, K., Poland, D., Borth, D., & Li, L.-J. (2016). YFCC100M: The New Data in Multimedia Research. https://doi.org/10.1145/2812802_
- Trained topic models and place embeddings (all computed using the original dataset) are included in this repository.

4. **Models**
- H32Vec Model: h32vec_res9.model, trained on geosocial data.
- BERTopic Models: Stored in Topic_Models/ folder: includes TextEmb, lowGeoTextEmb, midGeoTextEmb, highGeoTextEmb.
- Precomputed Embeddings: PlaceEmbeddings.db: place-level semantic embeddings (H3 cell-based).

5. Results and Interactive Visualizations
Generated within:
- Evaluation.ipynb (Tables 1 to 3, Figures 7 to 10)
- TopicQueryDemo_PlaceBasedSemanticSearch.ipynb (Figure 11)
- PlaceQueryDemo_PlaceBasedSemanticSearch.ipynb (Figure 12)
- Interactive maps (.html) for Figure 10 to 12 in folder 03_Visualizations


## 📚 How to Cite

If you use this repository, please cite the associated article

> Gugulica and Burghardt. 2005. “From Public Discourse to Place Semantics: Modeling Semantic Similarity of Places using Geosocial Media Data” *International Journal of Geographical Information Science (IJGIS). * **In Press** DOI to be added upon publication.
> 
