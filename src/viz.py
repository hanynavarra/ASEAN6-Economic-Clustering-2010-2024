# src/viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram

def set_style():
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#E6E6E6",
        "figure.dpi": 110,
        "savefig.bbox": "tight"
    })

def plot_elbow_sil(df_scores, save=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df_scores["k"], df_scores["inertia"], marker="o")
    ax2.plot(df_scores["k"], df_scores["silhouette"], marker="s", linestyle="--")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (elbow)")
    ax2.set_ylabel("Silhouette")
    ax1.set_title("Model selection: Elbow & Silhouette")
    if save: plt.savefig(str(save))
    plt.show()

def plot_pca_clusters(X_scaled, labels, save=None):
    feats = [c for c in X_scaled.columns if c != "iso3c"]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled[feats].values)
    dfp = pd.DataFrame(coords, columns=["PC1","PC2"])
    dfp["iso3c"] = X_scaled["iso3c"].values
    dfp["cluster"] = labels

    fig, ax = plt.subplots()
    sns.scatterplot(data=dfp, x="PC1", y="PC2", hue="cluster", s=140, ax=ax)
    for _, r in dfp.iterrows():
        ax.text(r["PC1"]+0.03, r["PC2"]+0.03, r["iso3c"], fontsize=11)
    ax.set_title("PCA projection of clusters")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    if save: plt.savefig(str(save))
    plt.show()

def plot_dendrogram(Z, labels, save=None):
    fig, ax = plt.subplots()
    dendrogram(Z, labels=labels, leaf_rotation=0, leaf_font_size=11, ax=ax, color_threshold=0.0)
    ax.set_title("Hierarchical clustering (dendrogram)")
    if save: plt.savefig(str(save))
    plt.show()

def plot_feature_heatmap(df_raw, clusters, save=None):
    X = df_raw.merge(clusters, on="iso3c")
    features = [c for c in df_raw.columns if c!="iso3c"]
    Xg = X.groupby("cluster")[features].mean()
    fig, ax = plt.subplots()
    sns.heatmap(Xg, annot=True, fmt=".1f", cmap="Greys", cbar=False, ax=ax)
    ax.set_title("Cluster profiles (feature means, 2020–2024)")
    if save: plt.savefig(str(save))
    plt.show()
