# Spotify Track Clustering Using Audio Features

## Project Overview

This project explores whether Spotify tracks can be grouped into meaningful musical clusters using audio features alone.

The analysis uses unsupervised machine learning, specifically **K-Means clustering**, to identify natural groupings of songs based on characteristics such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and duration.

Spotify genre labels are **not used to train the clustering algorithm**. They are only used afterward to evaluate whether the discovered clusters correspond to existing genre categories.

---

## Research Questions

1. What is the most appropriate number of clusters?
2. Which audio features most strongly distinguish the clusters?
3. How can the clusters be interpreted as meaningful musical profiles?
4. Do the discovered clusters correspond to Spotify's existing genre labels?
5. Can dimensionality reduction reveal meaningful separation between the clusters?

---

## Dataset

The project uses a Spotify tracks dataset sourced from Kaggle.

The dataset contains approximately **114,000 tracks** and includes:

- track metadata
- Spotify audio features
- popularity information
- genre labels

Examples of audio features used in clustering include:

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_ms`

Metadata such as track name, artist, album, track ID, and genre were excluded from the clustering model.

---

## Methodology

### 1. Data Exploration

The dataset was explored for:

- missing values
- duplicate observations
- feature distributions
- potential outliers
- correlations between numerical variables

Several variables showed skewed distributions, particularly speechiness, instrumentalness, liveness, acousticness, and loudness.

A correlation matrix was also used to identify potentially redundant features.

---

### 2. Feature Preparation

The final clustering variables were standardized using `StandardScaler`.

Standardization was necessary because the features were measured on very different scales, for example:

- danceability: approximately 0–1
- tempo: BPM
- loudness: dB
- duration: milliseconds

Without scaling, variables with larger numerical ranges could dominate the Euclidean distance used by K-Means.

---

### 3. Selecting the Number of Clusters

Several methods were used to evaluate the appropriate number of clusters:

- Elbow Method
- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score
- Adjusted Rand Index (ARI) for stability
- Centroid profiling for interpretability

The elbow plot did not show a clear optimal value.

The 7-cluster solution was selected because it provided the best balance between:

- cluster separation
- stability
- interpretability

The 7-cluster solution had a mean ARI of approximately **0.994**, indicating extremely stable clustering across repeated K-Means runs.

Although the 8-cluster solution achieved slightly better internal separation on some metrics, it was substantially less stable.

---

## Final Cluster Profiles

The final model identified seven broad musical profiles:

| Cluster | Interpretation |
|---|---|
| 0 | Live & Energetic |
| 1 | Acoustic & Mellow |
| 2 | Instrumental & Extended |
| 3 | Danceable & Upbeat |
| 4 | Fast & Intense |
| 5 | Speech-Heavy & Live |
| 6 | Ambient Instrumental |

These names were assigned after examining the standardized centroid profiles of each cluster.

The labels are descriptive interpretations and were not used during training.

---

## Which Features Most Strongly Distinguish the Clusters?

Feature differentiation was evaluated using the range of standardized centroid values across clusters.

The strongest differentiating features were:

1. Speechiness
2. Loudness
3. Liveness
4. Energy
5. Instrumentalness
6. Acousticness

Danceability and valence also contributed meaningfully.

Tempo and track duration showed relatively smaller differences across cluster centroids.

Overall, the clusters were primarily distinguished by:

- speech content
- intensity and loudness
- live-performance characteristics
- instrumental qualities
- acoustic qualities
- danceability and mood

---

## Comparison with Spotify Genres

Spotify genre labels were used only after clustering as an external validation step.

A cross-tabulation showed that several genres were highly concentrated within specific clusters.

Examples included:

- reggaeton → Danceable & Upbeat
- metalcore → Fast & Intense
- comedy → Speech-Heavy & Live
- new-age → Ambient Instrumental
- minimal-techno → Instrumental & Extended

However, the overall **Normalized Mutual Information (NMI)** score between genre labels and cluster assignments was:

```text
0.167