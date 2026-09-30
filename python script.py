# %% [markdown]
# # Question
# 
# 1. What is the most appropriate number of clusters?
# 2. Which audio features most strongly distinguish the clusters?
# 3. How can the clusters be interpreted as meaningful musical profiles?
# 4. Do the discovered clusters correspond to Spotify's existing genre labels?
#     - Genre should not be used to train the clustering algorithm. Use it only afterward to evaluate whether certain genres concentrate within particular clusters.
# 5. Can dimensionality reduction reveal meaningful separation between the clusters?
#     - Use PCA to visualize the songs in 2D and examine how clearly the groups separate.

# %%
#imports

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# %%
spotify = pd.read_csv('spotify-tracks-dataset.csv')
spotify = spotify.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# %%
#data exploration and cleaning

# print(spotify.head())
# spotify.info()
spotify['track_genre'].describe()
spotify['track_genre'].value_counts()
#spotify.duplicated().sum()

# %%
potential_features = spotify.drop(['track_name', 'artists', 'track_id', 'album_name', 'track_genre'], axis=1)

potential_features.hist(bins=30)
plt.show()

# %% [markdown]
# Close to normal:
# - danceability
# - tempo
# - popularity (maybe not very useful for my question)
# - valence
# - duration_ms
# 
# Strongly skewed:
# - duration_ms
# - energy
# - instrumentalness
# - liveness
# - acousticness
# - loudness
# - speechiness

# %%
# box plot
# features_excluding_explicit = potential_features.drop(['explicit'], axis=1)

# for feature in features_excluding_explicit:
#     plt.figure(figsize=(6, 2))
#     plt.boxplot(features_excluding_explicit[feature], vert=False)
#     plt.title(feature)
#     plt.show()

# %%
# Checking correlation between features
correlation_matrix = potential_features.corr()
plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f'
)

plt.title('Correlation Matrix of Spotify Features')
plt.tight_layout()
plt.show()

# %% [markdown]
# Assuming thershold of 0.7 and -0.7 as highly correlated, the following factors are of concern:
# - loudness vs energy
# - acousticness vs energy
# 
# Based on research and intution, the loudness factor maybe redundant. To confirm if it should be kept or not in the final model, **ablation analysis** will be performed.

# %%
# final features
continuous_features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms'
]

categorical_or_discrete_features = [
    'key',
    'mode',
    'time_signature'
]

# %%
# standardization
X = spotify[continuous_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# testing for right amount of clusters

inertia_values = []

k_values= range(2, 20)

for k in k_values:
    kmeans= KMeans(
        n_clusters=k,
        random_state=1,
        n_init=10 #means K-Means tries 10 different starting centroid positions and keeps the best solution.
    )

    kmeans.fit(X_scaled)

    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.show()

# %% [markdown]
# No obvious elbow. Using the silhouette score to decide confidently.

# %%
from sklearn.metrics import silhouette_score

silhouette_scores = []

k_values = range(2, 13)

for k in k_values:
    
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=10000,
        random_state=42
    )
    
    silhouette_scores.append(score)

    print(f"k = {k}: Silhouette Score = {score:.4f}")

# %%
#Other tests to decide k
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

k_values = list(range(2, 13))

results = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    ch_score = calinski_harabasz_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    results.append((k, ch_score, db_score))

    print(
        f"k={k} | "
        f"Calinski-Harabasz={ch_score:.2f} | "
        f"Davies-Bouldin={db_score:.4f}"
    )

# %% [markdown]
# 2 clusters is too small. Doing cluster stability and interpretability comparion for k= 6, 7 and 8.

# %%
# stability test with ARI
from sklearn.metrics import adjusted_rand_score
import numpy as np

def cluster_stability(X, k, seeds=range(10)):
    label_sets = []

    for seed in seeds:
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10
        )
        
        labels = model.fit_predict(X)
        label_sets.append(labels)

    ari_scores = []

    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            ari = adjusted_rand_score(
                label_sets[i],
                label_sets[j]
            )
            ari_scores.append(ari)

    return np.mean(ari_scores), np.std(ari_scores)

mean_ari_6, std_ari_6 = cluster_stability(X_scaled, 6)
mean_ari_7, std_ari_7 = cluster_stability(X_scaled, 7)
mean_ari_8, std_ari_8 = cluster_stability(X_scaled, 8)

print("k=6:", mean_ari_6, std_ari_6)
print("k=7:", mean_ari_7, std_ari_7)
print("k=8:", mean_ari_8, std_ari_8)

# %% [markdown]
# ## ARI
# 
# The **Adjusted Rand Index** measures how similar two different clustering results are.
# 
# It compares whether *pairs of observations are grouped together or separated in the same way across two clusterings*.
# 
# ARI is useful when we want to compare two sets of cluster labels.
# 
# **Typical use cases**:
# - checking whether clustering results are stable across different random seeds
# - comparing two clustering algorithms
# - comparing clustering results before and after changing features
# - comparing predicted clusters with known labels, if external labels exist
# 
# **Interpreting the scores**:
# - 1  perfectly stable
# - more than 0.90   very stable
# - 0.75–0.90 strong stability
# - 0.50–0.75 moderate stability
# - less than 0.50   weak stability
# - 0 perfectly instable
# 
# **Limitation**:
# So ARI answers:
# >“Are the clusters reproducible?”
# 
# not:
# >“Are these the best possible clusters?”

# %%
# For practice also doing centroid profiling

#Fitting KMeans models
kmeans_6 = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

labels_6 = kmeans_6.fit_predict(X_scaled)

kmeans_7 = KMeans(
    n_clusters=7,
    random_state=42,
    n_init=10
)

labels_7 = kmeans_7.fit_predict(X_scaled)

# %%
#Getting the centroids
centroids_6_original = scaler.inverse_transform(
    kmeans_6.cluster_centers_
)

centroids_7_original = scaler.inverse_transform(
    kmeans_7.cluster_centers_
)


# %%
profiles_6_scaled = pd.DataFrame(
    kmeans_6.cluster_centers_,
    columns=continuous_features
)

profiles_7_scaled = pd.DataFrame(
    kmeans_7.cluster_centers_,
    columns=continuous_features
)

profiles_6_scaled.index.name = 'cluster'
profiles_7_scaled.index.name = 'cluster'


plt.figure(figsize=(12, 6))

sns.heatmap(
    profiles_6_scaled,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0
)

plt.title('Standardized Centroid Profiles: k = 6')
plt.xlabel('Features')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

sns.heatmap(
    profiles_7_scaled,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0
)

plt.title('Standardized Centroid Profiles: k = 7')
plt.xlabel('Features')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

# %%
#Checking number of songs in extra cluster
cluster_sizes_7 = pd.Series(labels_7).value_counts().sort_index()

print(cluster_sizes_7)

print("\nPercentages:")
print(cluster_sizes_7 / len(labels_7) * 100)

# %% [markdown]
# ## Question 1
# 
# Seven clusters were selected as the final K-Means solution because it provided the best balance between cluster quality, stability, and interpretability. 
# 
# Although the silhouette score was slightly higher for \(k=8\), the \(k=8\) solution was substantially less stable across repeated runs, with a mean Adjusted Rand Index (ARI) of 0.805 and a relatively high standard deviation. 
# 
# In contrast, \(k=7\) produced an extremely stable solution, with a mean ARI of approximately 0.994 and very low variability. The \(k=7\) solution also achieved a better silhouette score and Davies–Bouldin score than \(k=6\), while maintaining clearly differentiated centroid profiles. Therefore, \(k=7\) was chosen as the most robust and interpretable clustering solution.

# %%
# Which audio features most strongly distinguish the clusters?

feature_separation = (
    profiles_7_scaled.max() - profiles_7_scaled.min()
).sort_values(ascending=False)

print(feature_separation)

feature_separation.plot(
    kind='bar',
    figsize=(10, 5)
)

plt.ylabel('Range Across Standardized Cluster Centroids')
plt.title('Features That Most Distinguish the Clusters')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Question 2
# 
# **Speechiness was the strongest feature distinguishing the clusters**, with a standardized centroid range of 7.38. This indicates that at least one cluster differs very substantially from the others in terms of spoken-word content. 
# 
# Loudness (3.14) and liveness (3.04) were the next most differentiating features, followed by energy (2.56), instrumentalness (2.51), and acousticness (2.37). Danceability (2.05) and valence (1.95) also contributed meaningfully to cluster differentiation, while tempo (1.26) and duration (0.95) showed comparatively smaller differences across clusters. 
# 
# Overall, the clusters appear to be distinguished primarily by **speech content, intensity/loudness, live-performance characteristics, and the balance between instrumental and acoustic qualities**, rather than by track duration or tempo alone.

# %%
# Naming the clusters based on their profiles
for cluster in profiles_7_scaled.index:

    print(f"\nCluster {cluster}")

    print(
        profiles_7_scaled
        .loc[cluster]
        .sort_values(ascending=False)
    )

# %%
#Adding cluster names
spotify_clustered = spotify.copy()

spotify_clustered['cluster'] = labels_7

cluster_names = {
    0: 'Live & Energetic',
    1: 'Acoustic & Mellow',
    2: 'Instrumental & Extended',
    3: 'Danceable & Upbeat',
    4: 'Fast & Intense',
    5: 'Speech-Heavy & Live',
    6: 'Ambient Instrumental'
}

spotify_clustered['cluster_name'] = (
    spotify_clustered['cluster']
    .map(cluster_names)
)

spotify_clustered[
    ['track_name', 'artists', 'cluster', 'cluster_name']
].head()

# %% [markdown]
# ## Question 3
# 
# Based on the characteristics, the clusters were named with the help of ChatGPT.

# %%
#Comparing with the given genres

genre_cluster_pct = pd.crosstab(
    spotify_clustered['track_genre'],
    spotify_clustered['cluster_name'],
    normalize='index'
) * 100

#Finding dominant cluster for each genre and the percentage of songs in that cluster
dominant_cluster = genre_cluster_pct.idxmax(axis=1)

dominant_percentage = genre_cluster_pct.max(axis=1)

genre_summary = pd.DataFrame({
    'dominant_cluster': dominant_cluster,
    'percentage_in_dominant_cluster': dominant_percentage
})

genre_summary.sort_values(
    'percentage_in_dominant_cluster',
    ascending=False
)



# %% [markdown]
# ### Normalized Mutual Information
# 
# Normalized Mutual Information measures **how much information two different labelings share**.
# 
# In clustering, it is commonly used to compare:
# - the clusters found by an unsupervised algorithm
# - with some known external labels, such as class or genre labels
# 
# **Interpretation**:
# |           NMI | Meaning                          |
# | ------------: | -------------------------------- |
# |           `0` | no meaningful shared information |
# |  close to `0` | weak relationship                |
# | middle values | partial relationship             |
# |  close to `1` | strong correspondence            |
# |           `1` | perfect agreement                |
# 

# %%
# Performing NMI to confirm results

from sklearn.metrics import normalized_mutual_info_score
nmi_score = normalized_mutual_info_score(
    spotify_clustered['track_genre'],
    spotify_clustered['cluster']
)

print("NMI score:", nmi_score)


# %% [markdown]
# ## Question 4
# 
# The Normalized Mutual Information score between the audio-based clusters and Spotify genre labels was 0.167, indicating weak overall correspondence. This suggests that the seven clusters do not reproduce Spotify’s genre taxonomy directly. However, the genre-level analysis showed that several individual genres were strongly concentrated within specific clusters. 
# 
# Therefore, the **clusters appear to capture broader sonic characteristics shared across multiple genres** rather than one-to-one genre categories.

# %%
#PCA to check if less features can be used
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


print(pca.explained_variance_ratio_)
print("Total variance explained:",
      pca.explained_variance_ratio_.sum())

# %%
# Visualizing PCA

pca_df = pd.DataFrame(
    X_pca,
    columns=['PC1', 'PC2']
)

pca_df['cluster'] = labels_7
pca_df['cluster_name'] = pca_df['cluster'].map(cluster_names)

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='cluster_name',
    alpha=0.4,
    s=15
)

plt.title('PCA Visualization of Spotify Track Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(
    title='Cluster',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Question 5
# 
# The first two principal components explained approximately 44% of the total variance in the standardized audio features, with PC1 accounting for 28.7% and PC2 for 15.2%. Therefore, the two-dimensional PCA representation captures a meaningful but incomplete portion of the original feature space.
# 
# Thus, PCA reveals recognizable regions associated with several clusters, while also showing substantial overlap, suggesting that Spotify tracks form overlapping sonic profiles rather than perfectly separated musical categories.


