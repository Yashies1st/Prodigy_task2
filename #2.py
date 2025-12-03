import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load data
data = pd.read_csv(r"C:\Users\Yashc\Downloads\responsivewebsiteangular12-master (1)\responsivewebsiteangular12-master\src\Mall_Customers.csv")  # make sure this file is in the same folder

# 2. Select numeric features for clustering
# Common choice: Annual Income and Spending Score
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# 3. Scale features (recommended for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Choose number of clusters (here we directly use 5 as a common example)
kmeans = KMeans(n_clusters=5, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

# 5. Inspect cluster centers (in scaled space)
print("Cluster centers (scaled):")
print(kmeans.cluster_centers_)

# 6. See how many customers in each cluster
print("\nCustomers per cluster:")
print(data["Cluster"].value_counts())

# 7. Save results with cluster labels
data.to_csv("mall_customers_with_clusters.csv", index=False)
print("mall_customers_with_clusters.csv has been created.")
