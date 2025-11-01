import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Load and Prepare Data
# The dataset typically contains CustomerID, Genre, Age, Annual Income (k$), and Spending Score (1-100).
# Assuming the file is named 'Mall_Customers.csv' after download from the Kaggle link.
try:
    data = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("Error: 'Mall_Customers.csv' not found. Please ensure you have downloaded and placed the file in the correct directory.")
    # Exit or use a sample dataframe for demonstration if necessary
    
# We select the features for clustering: Annual Income (k$) and Spending Score (1-100)
# These are columns 3 and 4 (0-indexed) or use the column names
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Step 2: Determine Optimal Number of Clusters (K) using the Elbow Method
# The goal is to find the K where the WCSS starts to decrease linearly (the 'elbow').
wcss = []
max_clusters = 11

for i in range(1, max_clusters):
    # Initialize K-Means model
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    # Fit the model to the data
    kmeans.fit(X)
    # Append the Within-Cluster Sum of Squares (inertia) to the list
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# Based on the typical output of this dataset, the optimal K is K=5 (The elbow point).
optimal_k = 5
print(f"\nOptimal number of clusters (K) determined from the plot is likely: {optimal_k}")


# Step 3: Apply K-Means Clustering with Optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)

# Fit the model and predict the cluster for each data point
y_kmeans = kmeans.fit_predict(X) 

# Step 4: Visualize the Clusters

# Create the scatter plot
plt.figure(figsize=(12, 8))

# Plot each of the 5 clusters separately
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1: Low Income, High Spend (Careful)')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2: Mid Income, Mid Spend (Standard)')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3: High Income, High Spend (Target)')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4: Low Income, Low Spend (Frugal)')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5: High Income, Low Spend (Miser)')

# Plot the cluster centers (centroids)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', marker='*', label='Centroids', edgecolors='black')

plt.title('Customer Segments using K-Means Clustering (K=5)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Add the cluster column back to the original DataFrame for analysis
data['Cluster'] = y_kmeans
print("\nFirst 5 rows of data with new Cluster labels:")
print(data.head())