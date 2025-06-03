import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data from CSV file
# Make sure your CSV file is named 'test_data.csv' and has columns 'x' and 'y'
data = pd.read_csv('test_data.csv')
X = data[['x', 'y']].values

# Number of clusters to find (you can change this)
n_clusters = 3

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering from CSV Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
