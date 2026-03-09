import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
df = pd.read_csv("../dummy-data/vehicles_ml_dataset.csv")
X = df[SEGMENT_FEATURES]

# Standardize features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using elbow method and silhouette analysis
best_score = -1
best_n_clusters = 3
best_kmeans = None

for n_clusters in range(2, 8):
    kmeans_test = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans_test.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters
        best_kmeans = kmeans_test

print(f"Best number of clusters: {best_n_clusters} with silhouette score: {best_score:.3f}")

# Use the best model
kmeans = best_kmeans
df["cluster_id"] = kmeans.predict(X_scaled)
centers = kmeans.cluster_centers_

# Transform centers back to original scale
centers_original = scaler.inverse_transform(centers)

# Calculate coefficient of variation for each cluster
cluster_stats = []
for i in range(best_n_clusters):
    cluster_data = df[df["cluster_id"] == i]
    income_cv = cluster_data["estimated_income"].std() / cluster_data["estimated_income"].mean()
    price_cv = cluster_data["selling_price"].std() / cluster_data["selling_price"].mean()
    cluster_stats.append({
        'cluster_id': i,
        'income_cv': income_cv,
        'price_cv': price_cv,
        'avg_income_cv': cluster_data["estimated_income"].std() / cluster_data["estimated_income"].mean()
    })

# Sort clusters by income
sorted_clusters = centers_original[:, 0].argsort()
cluster_mapping = {}

# Create dynamic cluster mapping based on number of clusters
if best_n_clusters == 2:
    cluster_mapping = {
        sorted_clusters[0]: "Standard",
        sorted_clusters[1]: "Premium",
    }
elif best_n_clusters == 3:
    cluster_mapping = {
        sorted_clusters[0]: "Economy",
        sorted_clusters[1]: "Standard",
        sorted_clusters[2]: "Premium",
    }
else:
    # For more than 3 clusters
    segment_names = ["Economy", "Standard", "Premium"] + [f"Segment_{i}" for i in range(3, best_n_clusters)]
    for i in range(best_n_clusters):
        cluster_mapping[sorted_clusters[i]] = segment_names[i]

df["client_class"] = df["cluster_id"].map(cluster_mapping)
os.makedirs("model_generators/clustering", exist_ok=True)
joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")
joblib.dump(scaler, "model_generators/clustering/scaler.pkl")

silhouette_avg = round(best_score, 2)
cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]

# Calculate overall coefficient of variation
overall_income_cv = df["estimated_income"].std() / df["estimated_income"].mean()
overall_price_cv = df["selling_price"].std() / df["selling_price"].mean()
avg_cv = (overall_income_cv + overall_price_cv) / 2

def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "coefficient_of_variation": round(avg_cv, 3),
        "income_cv": round(overall_income_cv, 3),
        "price_cv": round(overall_price_cv, 3),
        "cluster_stats": cluster_stats,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }
