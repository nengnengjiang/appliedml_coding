import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 0. Sample Data Setup (Mock Transaction Table) ---
np.random.seed(42)
num_transactions = 500
start_date = datetime(2025, 1, 1)

data = {
    'date': [start_date + timedelta(days=int(d)) for d in np.random.randint(0, 100, num_transactions)],
    'customer_id': np.random.randint(1001, 1050, num_transactions), # 50 customers
    'dollars': np.random.randint(50, 400, num_transactions)
}
df = pd.DataFrame(data)

# --- STAGE 1 & 2: CORE PROCESSING (Shared) ---
df['date'] = pd.to_datetime(df['date'])
df['dollars'] = df['dollars'].astype(float)

# Sort by customer and date
df = df.sort_values(['customer_id', 'date']).reset_index(drop=True)

# A. Prep: Add simple week number (%W) and year
df['week'] = df['date'].dt.strftime('%W')
df['year'] = df['date'].dt.year

# B. YTD Sum: Calculate YTD cumulative spending
df['ytd_spending'] = df.groupby(['customer_id', 'year'])['dollars'].cumsum()

# C. Qualify: Find first date and week a customer qualified
qualified = df[df['ytd_spending'] >= 1000]
first_qualified = qualified.groupby('customer_id')['date'].min().reset_index()
first_qualified['week'] = first_qualified['date'].dt.strftime('%W')

# D. Count: New qualifiers per week
weekly_new_qualifiers = first_qualified.groupby('week').size().reset_index(name='new_qualifiers')

# E. Accumulate: Calculate the cumulative sum (Q1 Numerator)
weekly_new_qualifiers['cum_qualified'] = weekly_new_qualifiers['new_qualifiers'].cumsum()

# Create a master DataFrame for merging results
final_metrics = weekly_new_qualifiers[['week', 'cum_qualified']].set_index('week')


# --- QUESTION 1: FINAL RESULT ---
final_metrics['num_customers_ytd_over_1000'] = final_metrics['cum_qualified']
print("--- Question 1 Result (Cumulative Qualified Customers) ---")
print(final_metrics[['num_customers_ytd_over_1000']].head())


# --- QUESTION 2: SCENARIO A (Dynamic Denominator) ---

# F. Denominator (First Seen): Find the first week a customer was seen
first_seen = df.groupby('customer_id')['date'].min().reset_index()
first_seen['week'] = first_seen['date'].dt.strftime('%W')

# G. Accumulate (Total Customers): Calculate new customers per week
weekly_new_customers = first_seen.groupby('week').size().reset_index(name='new_customers')

# Merge with the final metrics table
final_metrics = final_metrics.merge(
    weekly_new_customers.set_index('week'),
    left_index=True,
    right_index=True,
    how='outer' # Use outer merge to include all weeks present in either set
).fillna(0)

# Calculate the running total of all customers seen so far (Denominator)
final_metrics['cum_total_customers'] = final_metrics['new_customers'].cumsum()

# H. Ratio: Calculate the Dynamic Ratio
final_metrics['ratio_dynamic'] = final_metrics['cum_qualified'] / final_metrics['cum_total_customers']

print("\n--- Question 2 Scenario A Result (Dynamic Ratio: Cumulative Qualified / Cumulative Total Customers) ---")
print(final_metrics[['cum_qualified', 'cum_total_customers', 'ratio_dynamic']].head())


# --- QUESTION 2: SCENARIO B (Static Denominator) ---

# F. Denominator (Static): Total unique customers in the entire dataset
total_customers_static = df['customer_id'].nunique()

# G. Ratio: Calculate the Static Ratio
final_metrics['ratio_static'] = final_metrics['cum_qualified'] / total_customers_static

print(f"\n--- Question 2 Scenario B Result (Static Denominator: / {total_customers_static}) ---")
print(final_metrics[['cum_qualified', 'ratio_static']].head())




import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------
# Assume df is already loaded
# df = pd.read_csv("data.csv")
# -----------------------

# If you have a label column, drop it for clustering
label_col = "has_product_click"
if label_col in df.columns:
    df_features = df.drop(columns=[label_col])
else:
    df_features = df.copy()

# -----------------------
# Keep numeric features only (K-means works on numeric vectors)
# -----------------------
X = df_features.select_dtypes(include=["int64", "float64"]).copy()

# Optional: drop obvious ID-like columns if they exist (usually not meaningful for clustering)
id_like = ["request_id_anon", "retailer_token_anon", "Unnamed: 0", "Unnamed: 0.1"]
X = X.drop(columns=[c for c in id_like if c in X.columns], errors="ignore")

print("Raw X shape:", X.shape)

# -----------------------
# Impute missing values (fit on full data since clustering is unsupervised)
# -----------------------
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

# -----------------------
# Scale features (very important for K-means / Euclidean distance)
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# -----------------------
# Train K-means
# -----------------------
k = 5  # <-- change as needed
kmeans = KMeans(
    n_clusters=k,
    random_state=78,
    n_init=10  # stable default across sklearn versions
)

cluster_labels = kmeans.fit_predict(X_scaled)

print("Inertia (sum of squared distances):", kmeans.inertia_)
print("Cluster sizes:", np.bincount(cluster_labels))

# -----------------------
# Attach cluster assignments back to df for analysis
# -----------------------
df_out = df.copy()
df_out["cluster_id"] = cluster_labels

# -----------------------
# Quick cluster interpretation:
# mean of original (unscaled) numeric features per cluster
# -----------------------
cluster_summary = df_out.groupby("cluster_id")[X.columns].mean()
display(cluster_summary)





import random

# -----------------------
# Basic math helpers (no numpy)
# -----------------------
def squared_euclidean(a, b):
    # squared distance avoids sqrt (same nearest-centroid result, faster)
    s = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        s += diff * diff
    return s

def mean_of_points(points, dim):
    # compute coordinate-wise mean for a list of points
    centroid = [0.0] * dim
    n = len(points)
    for p in points:
        for j in range(dim):
            centroid[j] += p[j]
    for j in range(dim):
        centroid[j] /= n
    return centroid

def centroid_shift_sq(old_c, new_c):
    # how much centroid moved (squared)
    return squared_euclidean(old_c, new_c)

# -----------------------
# K-means main routine
# -----------------------
def kmeans(X, k, max_iters=100, tol=1e-6, seed=78):
    """
    X: list[list[float]]  (n_samples x n_features)
    k: number of clusters
    Returns:
      centroids: list[list[float]] (k x n_features)
      labels: list[int] (n_samples,)
    """
    random.seed(seed)

    n = len(X)
    dim = len(X[0])

    # -----------------------
    # 1) Initialize centroids by sampling k unique points from X
    # -----------------------
    init_indices = random.sample(range(n), k)
    centroids = [X[i][:] for i in init_indices]  # copy points as centroids

    labels = [0] * n

    # -----------------------
    # 2) Iterate assignment + update
    # -----------------------
    for it in range(max_iters):
        # ----- Assignment step -----
        changed = 0
        for i in range(n):
            # find nearest centroid for point X[i]
            best_j = 0
            best_dist = squared_euclidean(X[i], centroids[0])

            for j in range(1, k):
                d = squared_euclidean(X[i], centroids[j])
                if d < best_dist:
                    best_dist = d
                    best_j = j

            if labels[i] != best_j:
                labels[i] = best_j
                changed += 1

        # ----- Update step -----
        clusters = [[] for _ in range(k)]
        for i in range(n):
            clusters[labels[i]].append(X[i])

        new_centroids = []
        for j in range(k):
            if len(clusters[j]) == 0:
                # handle empty cluster: re-seed centroid to a random data point
                new_centroids.append(X[random.randrange(n)][:])
            else:
                new_centroids.append(mean_of_points(clusters[j], dim))

        # ----- Convergence check -----
        max_shift = 0.0
        for j in range(k):
            shift = centroid_shift_sq(centroids[j], new_centroids[j])
            if shift > max_shift:
                max_shift = shift

        centroids = new_centroids

        # Stop if centroids barely move (tol is on squared shift)
        if max_shift < tol:
            # print(f"Converged at iter {it+1}")
            break

        # Optional early stop: if no labels changed, clustering is stable
        if changed == 0:
            # print(f"No label changes at iter {it+1}")
            break

    return centroids, labels

# -----------------------
# Example usage
# -----------------------
X = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
    [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0]
]

centroids, labels = kmeans(X, k=2, max_iters=100, tol=1e-8, seed=78)
print("Centroids:", centroids)
print("Labels   :", labels)




import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------
# Example: numeric vectors only
# Each row is a point in R^d
# -----------------------
X = np.array([
    [0.2, 0.3, -0.5],
    [0.1, 0.4, -0.6],
    [2.0, 1.9,  2.1],
    [2.2, 2.1,  1.8],
    [-1.0, -0.8, -1.2],
    [-0.9, -1.1, -1.0],
], dtype=float)

# -----------------------
# Scaling: important for Euclidean distance
# If your vectors are already comparable scale, you can skip this
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# Train K-means
# -----------------------
k = 3  # number of clusters
kmeans = KMeans(
    n_clusters=k,
    random_state=78,
    n_init=10   # multiple initializations for stability
)

labels = kmeans.fit_predict(X_scaled)       # cluster assignment for each point
centroids_scaled = kmeans.cluster_centers_  # centroids in scaled space
inertia = kmeans.inertia_                  # sum of squared distances to centroids

print("labels:", labels)
print("inertia:", inertia)

# -----------------------
# If you want centroids in the original feature scale
# -----------------------
centroids_original = scaler.inverse_transform(centroids_scaled)
print("centroids (original scale):\n", centroids_original)




import numpy as np

# -----------------------
# Euclidean distance
# -----------------------
def euclidean_distance(a, b):
    # sqrt(sum((a - b)^2))
    return np.sqrt(np.sum((a - b) ** 2))


# -----------------------
# Assign each point to nearest centroid
# -----------------------
def assign_clusters(X, centroids):
    """
    X: (n_samples, n_features)
    centroids: (k, n_features)
    return: cluster index for each sample
    """
    labels = []

    for x in X:
        # compute distance to each centroid
        distances = [euclidean_distance(x, c) for c in centroids]
        # pick the closest centroid
        labels.append(np.argmin(distances))

    return np.array(labels)


# -----------------------
# Recompute centroids
# -----------------------
def update_centroids(X, labels, k):
    """
    X: data points
    labels: cluster assignment
    k: number of clusters
    """
    centroids = []

    for i in range(k):
        cluster_points = X[labels == i]

        # handle empty cluster
        if len(cluster_points) == 0:
            centroids.append(X[np.random.randint(0, len(X))])
        else:
            centroids.append(cluster_points.mean(axis=0))

    return np.array(centroids)


# -----------------------
# K-means main loop
# -----------------------
def kmeans(X, k, max_iters=100, tol=1e-4, random_state=78):
    np.random.seed(random_state)

    n_samples = X.shape[0]

    # 1) initialize centroids by sampling k points
    init_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[init_idx]

    for it in range(max_iters):
        # 2) assignment step
        labels = assign_clusters(X, centroids)

        # 3) update step
        new_centroids = update_centroids(X, labels, k)

        # 4) check convergence (centroid movement)
        shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if shift < tol:
            # print(f"Converged at iteration {it+1}")
            break

    return centroids, labels


# -----------------------
# Example usage
# -----------------------
X = np.array([
    [0.2, 0.3, -0.5],
    [0.1, 0.4, -0.6],
    [2.0, 1.9,  2.1],
    [2.2, 2.1,  1.8],
    [-1.0, -0.8, -1.2],
    [-0.9, -1.1, -1.0],
])

centroids, labels = kmeans(X, k=3)

print("Centroids:")
print(centroids)
print("Labels:")
print(labels)




#####


import numpy as np

# -----------------------
# Euclidean distance
# -----------------------
def euclidean_distance(a, b):
    # sqrt(sum((a - b)^2))
    return np.sqrt(np.sum((a - b) ** 2))


# -----------------------
# Assign each point to nearest centroid
# -----------------------
def assign_clusters(X, centroids):
    """
    X: (n_samples, n_features)
    centroids: (k, n_features)
    return: cluster index for each sample
    """
    labels = []

    for x in X:
        # compute distance to each centroid
        distances = [euclidean_distance(x, c) for c in centroids]
        # pick the closest centroid
        labels.append(np.argmin(distances))

    return np.array(labels)


# -----------------------
# Recompute centroids
# -----------------------
def update_centroids(X, labels, k):
    """
    X: data points
    labels: cluster assignment
    k: number of clusters
    """
    centroids = []

    for i in range(k):
        cluster_points = X[labels == i]

        # handle empty cluster
        if len(cluster_points) == 0:
            centroids.append(X[np.random.randint(0, len(X))])
        else:
            centroids.append(cluster_points.mean(axis=0))

    return np.array(centroids)


# -----------------------
# K-means main loop
# -----------------------
def kmeans(X, k, max_iters=100, tol=1e-4, random_state=78):
    np.random.seed(random_state)

    n_samples = X.shape[0]

    # 1) initialize centroids by sampling k points
    init_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[init_idx]

    for it in range(max_iters):
        # 2) assignment step
        labels = assign_clusters(X, centroids)

        # 3) update step
        new_centroids = update_centroids(X, labels, k)

        # 4) check convergence (centroid movement)
        shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if shift < tol:
            # print(f"Converged at iteration {it+1}")
            break

    return centroids, labels


# -----------------------
# Example usage
# -----------------------
X = np.array([
    [0.2, 0.3, -0.5],
    [0.1, 0.4, -0.6],
    [2.0, 1.9,  2.1],
    [2.2, 2.1,  1.8],
    [-1.0, -0.8, -1.2],
    [-0.9, -1.1, -1.0],
])

centroids, labels = kmeans(X, k=3)

print("Centroids:")
print(centroids)
print("Labels:")
print(labels)





$$$$$$





###############################
# machine learning in action, p #209
###############################

import numpy as np
import sys
import math


# support functions, read files

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(list(fltLine))
    return np.mat(dataMat)


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    '''
    creates a set of k random (initial) centroids need to be within the bounds of the dataset
    The random centroids need to be within the bounds of the dataset.
    :param dataSet:
    :param k:
    :return: initialize centroid (k,n)
    '''
    n = dataSet.shape[1]  # number of features
    centroids = np.mat(np.zeros((k, n)))  # k rows, n features
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        p1 = minJ
        p2 = rangeJ * np.random.rand(k, 1)
        centroids[:, j] = p1+p2 # k rows and 1 col
    return centroids

def kMeans(dataSet, k ,distMeas=distEclud,createCent = randCent):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2))) # 1st col for index, 2nd col for mindist
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # loop for each ob in the dataSet, row i
            minDist = sys.maxsize
            minIndex = -1
            for j in range(k):
                # loop for each centroid, centroids, every row (j), has a vector for the certain centroids
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            #check if ob i's group keep same
            if clusterAssment[i, 0] !=minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2  # update index, mindist
        for cent in range(k):
            # np.nonzero, check the notes in onenote
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get the row/index number of of the none-zero row
        centroids[cent, :] = np.mean(ptsInClust, axis=0)  # average the vector along row direction
    return centroids, clusterAssment


if __name__ == "__main__":
    datMat = loadDataSet('testSet.txt')
    myCentroids, clustAssing = kMeans(datMat, 4)
    print(myCentroids)
    print(clustAssing)




#####



import numpy as np

# -----------------------
# Euclidean distance
# -----------------------
def euclidean_distance(a, b):
    # sqrt(sum((a - b)^2))
    return np.sqrt(np.sum((a - b) ** 2))


# -----------------------
# Assign each point to nearest centroid
# -----------------------
def assign_clusters(X, centroids):
    """
    X: (n_samples, n_features)
    centroids: (k, n_features)
    return: cluster index for each sample
    """
    labels = []

    for x in X:
        # compute distance to each centroid
        distances = [euclidean_distance(x, c) for c in centroids]
        # pick the closest centroid
        labels.append(np.argmin(distances))

    return np.array(labels)


# -----------------------
# Recompute centroids
# -----------------------
def update_centroids(X, labels, k):
    """
    X: data points
    labels: cluster assignment
    k: number of clusters
    """
    centroids = []

    for i in range(k):
        cluster_points = X[labels == i]

        # handle empty cluster
        if len(cluster_points) == 0:
            centroids.append(X[np.random.randint(0, len(X))])
        else:
            centroids.append(cluster_points.mean(axis=0))

    return np.array(centroids)


# -----------------------
# K-means main loop
# -----------------------
def kmeans(X, k, max_iters=100, tol=1e-4, random_state=78):
    np.random.seed(random_state)

    n_samples = X.shape[0]

    # 1) initialize centroids by sampling k points
    init_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[init_idx]

    for it in range(max_iters):
        # 2) assignment step
        labels = assign_clusters(X, centroids)

        # 3) update step
        new_centroids = update_centroids(X, labels, k)

        # 4) check convergence (centroid movement)
        shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if shift < tol:
            # print(f"Converged at iteration {it+1}")
            break

    return centroids, labels


# -----------------------
# Example usage
# -----------------------
X = np.array([
    [0.2, 0.3, -0.5],
    [0.1, 0.4, -0.6],
    [2.0, 1.9,  2.1],
    [2.2, 2.1,  1.8],
    [-1.0, -0.8, -1.2],
    [-0.9, -1.1, -1.0],
])

centroids, labels = kmeans(X, k=3)

print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
