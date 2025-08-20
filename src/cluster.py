# src/cluster.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from src.viz import set_style, plot_elbow_sil, plot_pca_clusters, plot_dendrogram, plot_feature_heatmap

PROC = Path("data/processed")
FIGS = Path("reports/figures")
FIGS.mkdir(parents=True, exist_ok=True)

def load_data():
    Xs = pd.read_csv(PROC / "feature_matrix_scaled.csv")   # scaled
    Xr = pd.read_csv(PROC / "feature_matrix_raw.csv")      # raw means (for heatmap labels)
    return Xs, Xr

def k_scan(X_scaled, k_range=range(2, 5)):
    X = X_scaled.drop(columns=["iso3c"]).values
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": inertia, "silhouette": sil})
    return pd.DataFrame(rows)

def fit_kmeans(X_scaled, k):
    X = X_scaled.drop(columns=["iso3c"]).values
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X)
    out = X_scaled[["iso3c"]].copy()
    out["cluster"] = labels
    return km, out

def main():
    set_style()
    Xs, Xr = load_data()

    # 1) scan k
    scores = k_scan(Xs, range(2, 5))
    print(scores)
    plot_elbow_sil(scores, save=FIGS / "01_elbow_silhouette.png")

    # 2) pick best k by silhouette
    k_best = int(scores.sort_values("silhouette", ascending=False).iloc[0]["k"])
    print(f"\nSelected k = {k_best} (best silhouette)")
    km, clusters = fit_kmeans(Xs, k_best)
    clusters.to_csv(PROC / f"clusters_k{k_best}.csv", index=False)
    print("\nCluster assignments:")
    print(clusters)

    # 3) visuals
    plot_pca_clusters(Xs, clusters["cluster"], save=FIGS / "02_pca_clusters.png")
    Z = linkage(Xs.drop(columns=["iso3c"]).values, method="ward")
    plot_dendrogram(Z, labels=Xs["iso3c"].tolist(), save=FIGS / "03_dendrogram.png")
    plot_feature_heatmap(Xr, clusters, save=FIGS / "04_cluster_profiles_heatmap.png")

if __name__ == "__main__":
    main()
