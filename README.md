# Unsupervised ML: Clustering Practice

My hands-on notes and code for **clustering**, the most common type of unsupervised machine learning. In unsupervised learning there are **no labels** (no "right answers"). The algorithm looks at the data and finds groups on its own.

Everything is written in Python with scikit-learn, and each notebook is meant to be re-run and re-read as a quick refresher.

---

## Table of Contents

1. [Repository structure](#repository-structure)
2. [Setup](#setup)
3. [Core concepts in plain language](#core-concepts-in-plain-language)
4. [Notebook 1: KMean.ipynb](#notebook-1-kmeanipynb)
5. [Notebook 2: KMeanForiris.ipynb](#notebook-2-kmeanforirisipynb)
6. [Notebook 3: Hierarchical_clustering.ipynb](#notebook-3-hierarchical_clusteringipynb)
7. [Notebook 4: DBSCAN.ipynb](#notebook-4-dbscanipynb)
8. [Algorithm comparison](#algorithm-comparison)
9. [Common code patterns](#common-code-patterns)
10. [Mistakes to avoid](#mistakes-to-avoid)
11. [What to learn next](#what-to-learn-next)

---

## Repository structure

```
unsupervised-ml-clustering/
├── KMean.ipynb                    # K-Means, elbow method, silhouette score
├── KMeanForiris.ipynb             # K-Means on Iris with scaling + PCA
├── Hierarchical_clustering.ipynb  # Dendrogram + Agglomerative clustering
├── DBSCAN.ipynb                   # DBSCAN on Iris and on non-linear (moons) data
├── images/                        # Visuals used for LinkedIn posts
└── README.md
```

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy kneed
jupyter notebook
```

`kneed` is only needed in `KMean.ipynb` (automatic elbow detection).

---

## Core concepts in plain language

| Term | Meaning |
|------|---------|
| **Clustering** | Grouping similar data points together without being told the groups |
| **Row** | One sample (one flower, one customer) |
| **Column** | One feature (petal length, age, salary) |
| **Centroid** | The center point of a cluster (the average of all its points) |
| **Label** | The cluster number assigned to each point (0, 1, 2 ...) |
| **Feature scaling** | Putting all columns on a similar range so no single column dominates the distance |

### How to read `X[:, 0]`

`X[rows, columns]`. The `:` means "all". So:

- `X[:, 0]` is the **first column** for every row
- `X[:, 2]` is the **third column** for every row
- `X[1, :]` is **row 1**, all columns

Counting starts at **0**. `X.shape` gives `(rows, columns)`.

```python
sns.scatterplot(x=X[:, 0], y=X[:, 2])   # x-axis = column 0, y-axis = column 2
```

### Why scaling matters

K-Means, hierarchical clustering and DBSCAN all use **distance** between points. If one column is in thousands (salary) and another is in single digits (age), the big one wins unfairly. `StandardScaler` rewrites each column to have **mean 0** and **standard deviation 1**, so every feature counts equally.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

After scaling, a negative number means "below average" for that column and a positive number means "above average".

---

## Notebook 1: KMean.ipynb

**Goal:** Learn K-Means from scratch on made-up data and learn how to **choose K**.

### Data

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=42)
```

- Creates 1000 points with 2 features, grouped around 4 centers.
- `y` holds the true group of each point, but K-Means does not use it.

### How K-Means works (4 steps)

1. Pick K random starting centroids.
2. Assign every point to its **nearest** centroid.
3. Move each centroid to the **average** of its points.
4. Repeat steps 2 and 3 until the centroids stop moving.

### Fit the model

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)      # labels[i] = cluster number of point i

sns.scatterplot(x=X[:, 0], y=X[:, 1], c=labels)
```

### Choosing K, method 1: Elbow method

Run K-Means for K = 1 to 20 and record **WCSS** (inertia): the total squared distance from each point to its centroid. WCSS always falls as K grows, but at some point the drop becomes small. That bend is the "elbow".

```python
wcss = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)

sns.lineplot(x=range(1, 21), y=wcss, marker='o')
```

**Automatic elbow detection with `kneed`:**

```python
from kneed import KneeLocator
knee = KneeLocator(range(1, 21), wcss, curve="convex", direction="decreasing")
print(knee.knee)     # My result: 4
```

My result matched the 4 centers I generated.

### Choosing K, method 2: Silhouette score

For each point, the silhouette score measures how close it is to its **own** cluster versus the **nearest other** cluster.

| Score | Meaning |
|-------|---------|
| close to **+1** | Well separated, a good clustering |
| close to **0** | Clusters overlap |
| negative | Points are probably in the wrong cluster |

```python
from sklearn.metrics import silhouette_score

ss = []
for k in range(2, 21):               # starts at 2: silhouette needs at least 2 clusters
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    ss.append(silhouette_score(X, labels))

sns.lineplot(x=range(2, 21), y=ss, marker="o")
```

**Pick the K with the highest silhouette score.**

### Random initialization trap

K-Means starts from random centroids. Unlucky starting positions can give a poor result (stuck in a bad local optimum). Fixes:

- scikit-learn uses **K-Means++** by default, which spreads the starting centroids out smartly
- `n_init` runs the algorithm several times and keeps the best result
- `random_state=42` makes results repeatable

### Key takeaways

- Elbow method: look for the bend in the WCSS curve.
- Silhouette score: look for the peak.
- Use both together for a more confident K.

---

## Notebook 2: KMeanForiris.ipynb

**Goal:** Run the full K-Means workflow on a real dataset: **Iris** (150 flowers, 4 measurements each).

### Data

```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # true species 0/1/2, used only to compare at the end
```

The 4 columns are:

| Column | Feature |
|--------|---------|
| 0 | sepal length |
| 1 | sepal width |
| 2 | petal length |
| 3 | petal width |

### Workflow

**Step 1. Look at raw data**

```python
sns.scatterplot(x=X[:, 0], y=X[:, 2])   # sepal length vs petal length
```

**Step 2. Scale the data**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Step 3. (Optional) Reduce to 2 dimensions with PCA**

PCA squeezes 4 columns into 2 new columns that keep as much information as possible. This makes the clusters easy to plot.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
```

**Step 4. Elbow method on the PCA data (K = 1 to 9)**

```python
wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)
```

My WCSS values:

| K | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| WCSS | 574.9 | 197.4 | 115.2 | 90.1 | 81.3 |

The big drops end around **K = 3**, which matches the 3 real species.

**Step 5. Apply K-Means with K = 3 and plot with centroids**

```python
kmeans = KMeans(n_clusters=3, random_state=10)
labels = kmeans.fit_predict(pca_data)

sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], c=labels)
sns.scatterplot(x=kmeans.cluster_centers_[:, 0],
                y=kmeans.cluster_centers_[:, 1],
                marker="x", c="red", s=200)
```

**Step 6. Compare with the real species**

```python
sns.scatterplot(x=X[:, 0], y=X[:, 2], c=y)
```

Putting the K-Means plot beside this one shows how close the unsupervised result is to the truth.

### Key takeaways

- Order of work: **load, scale, (PCA), elbow, fit, visualize**.
- K-Means never saw the labels `y`, yet it recovered groups very close to the real species.
- One species separates cleanly. The other two overlap slightly.

---

## Notebook 3: Hierarchical_clustering.ipynb

**Goal:** Cluster without choosing K first, by building a **tree of clusters**.

### Two types

| Type | Direction | How it works |
|------|-----------|--------------|
| **Agglomerative** | Bottom-up | Every point starts as its own cluster. Repeatedly merge the two closest clusters. This is the common one. |
| **Divisive** | Top-down | Start with one big cluster. Repeatedly split it. |

### Linkage (how to measure distance between clusters)

I used **`ward`**, which merges the pair of clusters that increases the total within-cluster variance the least. Other options are `single`, `complete` and `average`.

### Dendrogram

A dendrogram is the tree of all merges.

- **x-axis:** the samples
- **y-axis:** the distance at which two clusters merged
- **Tall vertical lines** mean the merged clusters were far apart

**How to choose the number of clusters:** draw a horizontal line across the dendrogram where the vertical lines are longest. The number of vertical lines it crosses is the number of clusters.

```python
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(X_scaled, method="ward")     # linkage matrix

plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.xlabel("samples")
plt.ylabel("distance")
plt.title("Dendogram for hierarchical clustering")
```

### Get the actual cluster labels

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X_scaled)

sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 2], c=labels)
```

### Key takeaways

- The dendrogram **helps you decide K**; `AgglomerativeClustering` then assigns the labels.
- It does not need random starting points, so the results are the same every run.
- It gets slow on large datasets (the whole tree has to be built).

---

## Notebook 4: DBSCAN.ipynb

**Goal:** Cluster by **density**, find outliers automatically, and handle **non-round shapes** where K-Means fails.

### How DBSCAN works

Two settings control everything:

| Parameter | Meaning |
|-----------|---------|
| `eps` | The radius of a point's neighborhood |
| `min_samples` | How many points must be inside that radius to count as "dense" |

Each point becomes one of three types:

- **Core point:** has at least `min_samples` neighbors within `eps`
- **Border point:** near a core point but not dense itself
- **Noise point:** neither, labeled **`-1`**

Clusters are formed by chaining core points together. **You do not choose K.** The algorithm discovers it.

### Part A: DBSCAN on Iris

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 2], c=labels)
```

**My result:** DBSCAN found **2 clusters (labels 0 and 1) plus a few noise points (label -1)**, not the 3 real species. Two of the Iris species overlap, so DBSCAN treats them as one dense region. This is a useful reminder that DBSCAN follows **density**, not "number of species".

### Part B: DBSCAN on non-linear data (the main lesson)

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
X_scaled = scaler.fit_transform(X)
```

This makes two interlocking half-moon shapes.

**K-Means fails here**, because it draws straight boundaries and assumes round clusters:

```python
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```

**DBSCAN succeeds**, because it follows the dense curve of each moon:

```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
```

### Tuning tips

- `eps` too **small**: everything becomes noise (`-1`).
- `eps` too **large**: all clusters merge into one.
- Always **scale** the data first, since `eps` is a distance.

### Key takeaways

- Use DBSCAN when clusters have **odd shapes** or the data has **outliers**.
- `-1` in the labels means **noise**.
- No need to pick K, but you do need to tune `eps` and `min_samples`.

---

## Algorithm comparison

| | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Need to choose K? | Yes | Decide from the dendrogram | No |
| Cluster shape | Round / spherical | Mostly compact | **Any shape** |
| Handles outliers? | No | No | **Yes** (labels them -1) |
| Speed on big data | **Fast** | Slow | Medium |
| Same result every run? | Not without `random_state` | **Yes** | Yes |
| Needs scaling? | Yes | Yes | Yes |
| Main settings | `n_clusters` | `n_clusters`, `linkage` | `eps`, `min_samples` |

**Quick guide:**

- Round, similar-sized groups and a large dataset: **K-Means**
- Want to see the structure or explore K: **Hierarchical**
- Weird shapes or outliers present: **DBSCAN**

---

## Common code patterns

**Basic pipeline (used in every notebook):**

```python
X_scaled = StandardScaler().fit_transform(X)   # 1. scale
model = SomeClusteringAlgorithm(...)           # 2. choose algorithm
labels = model.fit_predict(X_scaled)           # 3. fit and get labels
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], c=labels)   # 4. visualize
```

**Useful attributes:**

| Code | Gives |
|------|-------|
| `kmeans.inertia_` | WCSS for the elbow method |
| `kmeans.cluster_centers_` | Coordinates of the centroids |
| `X.shape` | `(rows, columns)` |
| `set(labels)` | The unique cluster numbers |

---

## Mistakes to avoid

1. **Forgetting to scale** the data before distance-based clustering.
2. **Starting the silhouette loop at K = 1**: it needs at least 2 clusters.
3. **Reading `-1` as a cluster** in DBSCAN: it means noise.
4. **Trusting one K-Means run**: set `random_state` or `n_init` for stable results.
5. **Using K-Means on curved or ring shapes**: use DBSCAN instead.
6. **Using the true labels `y` for training**: in unsupervised learning, `y` is only for checking afterwards.

---

## What to learn next

- PCA in depth (dimensionality reduction)
- Gaussian Mixture Models (soft clustering)
- t-SNE and UMAP (visualizing high-dimensional data)
- Anomaly detection (Isolation Forest)
- Evaluating clusters further: Davies-Bouldin index, Calinski-Harabasz score

---

*Built while learning Unsupervised Machine Learning with Python.*
