# src/report.py
from pathlib import Path
import pandas as pd

PROC = Path("data/processed")
FIGS = Path("reports/figures")
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

def latest_clusters_csv():
    files = sorted(PROC.glob("clusters_k*.csv"))
    if not files:
        raise FileNotFoundError("No clusters_k*.csv found. Run: python -m src.cluster")
    return files[-1]

def main():
    # load data
    X_raw = pd.read_csv(PROC / "feature_matrix_raw.csv")
    clusters = pd.read_csv(latest_clusters_csv())
    df = X_raw.merge(clusters, on="iso3c")
    k = int(df["cluster"].nunique())

    # per‑cluster means (pretty)
    features = [c for c in X_raw.columns if c != "iso3c"]
    profile = (
        df.groupby("cluster")[features]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"cluster": "Cluster"})
    )

    # membership list
    members = (
        df.sort_values(["cluster","iso3c"])
          .groupby("cluster")["iso3c"]
          .apply(lambda s: ", ".join(s))
          .rename("Members")
          .reset_index()
          .rename(columns={"cluster":"Cluster"})
    )

    # write Markdown
    md = []
    md.append("# ASEAN‑6 Economic Clustering — Summary")
    md.append("")
    md.append(f"- **k (clusters)**: **{k}**")
    md.append("- **Window**: 2020–2024 mean of each indicator")
    md.append("- **Indicators**: GDP per capita (US$), Trade openness (% GDP), Inflation (CPI %), Remittances (% GDP), FDI inflows (% GDP), Credit to private sector (% GDP)")
    md.append("")
    md.append("## Cluster membership")
    md.append(members.to_markdown(index=False))
    md.append("")
    md.append("## Cluster profiles (means)")
    md.append(profile.to_markdown(index=False))
    md.append("")
    md.append("## Figures")
    for f in ["01_elbow_silhouette.png","02_pca_clusters.png","03_dendrogram.png","04_cluster_profiles_heatmap.png"]:
        p = FIGS / f
        if p.exists():
            md.append(f"![{f}]({p.as_posix()})")
    md_text = "\n".join(md)

    out = REPORTS / "clusters_summary.md"
    out.write_text(md_text, encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
