# src/features.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Works with:  python -m src.features   OR   python src/features.py
try:
    from src.fetch import INDICATORS
except Exception:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from fetch import INDICATORS

RAW_PATH = Path("data/raw/wb_asean6_2010_2024.csv")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_raw(path=RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path)

def to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your raw file has columns like:
      iso3c | year(=indicator_code) | YR2010 ... YR2024
    We must rename that indicator-code column to avoid colliding with the real numeric year.
    """
    # Identify year columns like YR2010..YR2024
    yr_cols = [c for c in df.columns if str(c).startswith("YR")]
    if not yr_cols:
        return df.copy()

    # Detect the indicator-code column: the one that matches our known codes
    candidate_cols = [c for c in df.columns if c not in ["iso3c"] + yr_cols]
    ind_code_col = None
    known_codes = set(INDICATORS.keys())
    for c in candidate_cols:
        # score by how many rows look like a known code
        sample = df[c].astype(str).head(30)
        score = sum(s in known_codes for s in sample)
        if score >= 3:  # threshold
            ind_code_col = c
            break
    if ind_code_col is None:
        # fallback: your file shows this column is literally named 'year'
        ind_code_col = "year"

    # Rename indicator-code column to avoid name clash with the numeric year we will create
    df = df.rename(columns={ind_code_col: "ind_code"})

    # Melt to long
    long_df = df.melt(
        id_vars=["iso3c", "ind_code"],
        value_vars=yr_cols,
        var_name="year_tag",
        value_name="value"
    )

    # year_tag like "YR2010" -> numeric year
    long_df["year"] = long_df["year_tag"].str[2:].astype(int)
    long_df = long_df.drop(columns=["year_tag"])

    # Map indicator codes -> friendly feature names
    rev_map = {k: v for k, v in INDICATORS.items()}
    long_df["feature"] = long_df["ind_code"].map(lambda c: rev_map.get(c, c))

    # Pivot to wide: one row per (iso3c, year), columns are features
    tidy = long_df.pivot_table(
        index=["iso3c", "year"],
        columns="feature",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Ensure expected feature columns exist
    for f in INDICATORS.values():
        if f not in tidy.columns:
            tidy[f] = np.nan

    tidy = tidy.sort_values(["iso3c", "year"]).reset_index(drop=True)
    tidy.to_csv(PROC_DIR / "wb_asean6_tidy_2010_2024.csv", index=False)
    return tidy

def aggregate_window(tidy: pd.DataFrame, start=2020, end=2024) -> pd.DataFrame:
    window = tidy[(tidy["year"] >= start) & (tidy["year"] <= end)].copy()
    feats = list(INDICATORS.values())
    # mean that works on older pandas too
    agg = window.groupby("iso3c")[feats].agg(
        lambda s: pd.to_numeric(s, errors="coerce").mean()
    ).reset_index()
    agg.to_csv(PROC_DIR / f"feature_matrix_raw_{start}_{end}.csv", index=False)
    return agg

def impute_and_scale(agg: pd.DataFrame):
    X = agg.copy()
    features = [c for c in X.columns if c != "iso3c"]

    # median imputation
    imputer = SimpleImputer(strategy="median")
    X_num = imputer.fit_transform(X[features])

    scaler = StandardScaler()
    Z = scaler.fit_transform(X_num)

    X_scaled = pd.DataFrame(Z, columns=features)
    X_scaled.insert(0, "iso3c", X["iso3c"].values)

    # save
    agg.to_csv(PROC_DIR / "feature_matrix_raw.csv", index=False)
    X_scaled.to_csv(PROC_DIR / "feature_matrix_scaled.csv", index=False)
    joblib.dump({"imputer": imputer, "scaler": scaler, "features": features},
                PROC_DIR / "scaler.joblib")

    return X_scaled, imputer, scaler

def main():
    print("Loading raw CSV ...")
    df = load_raw()
    print(df.head(3))

    print("\nConverting to tidy format ...")
    tidy = to_tidy(df)
    print(tidy.head(6))

    print("\nAggregating 5-year window (2020–2024) ...")
    agg = aggregate_window(tidy, 2020, 2024)
    print(agg)

    print("\nImputing missing values and scaling ...")
    X_scaled, _, _ = impute_and_scale(agg)
    print("\nScaled feature matrix:")
    print(X_scaled)

if __name__ == "__main__":
    main()
