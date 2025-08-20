# src/fetch.py
from pathlib import Path
import pandas as pd
import wbgapi as wb

# --- ASEAN countries of interest ---
ASEAN6 = ["PHL", "IDN", "MYS", "THA", "VNM", "SGP"]

# --- Indicators (World Bank codes mapped to friendly names) ---
INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_pc_usd",            # GDP per capita (current US$)
    "NE.TRD.GNFS.ZS": "trade_open",            # Trade (% of GDP)
    "FP.CPI.TOTL.ZG": "inflation",             # Inflation, CPI (annual %)
    "BX.TRF.PWKR.DT.GD.ZS": "remit_pct_gdp",   # Remittances (% of GDP)
    "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",     # FDI inflows (% of GDP)
    "FS.AST.PRVT.GD.ZS": "credit_priv_pct_gdp" # Domestic credit to private sector (% of GDP)
}

def fetch_worldbank(start=2010, end=2024, outdir="data/raw"):
    """Fetch ASEAN6 data from World Bank and save as CSV"""
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Pull data (may return MultiIndex or columns depending on wbgapi version)
    df = wb.data.DataFrame(
        list(INDICATORS.keys()),
        economy=ASEAN6,
        time=range(start, end + 1),
        labels=False
    )

    # Ensure the index has names
    if df.index.nlevels == 2:
        df.index = df.index.set_names(["iso3c", "year"])
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)
        if "economy" in df.columns:
            df = df.rename(columns={"economy": "iso3c"})
        if "time" in df.columns:
            df = df.rename(columns={"time": "year"})

    # Rename indicators to friendly names
    df = df.rename(columns=INDICATORS)

    # Sort for readability
    if "year" in df.columns:
        df = df.sort_values(["iso3c", "year"])

    # Save to CSV
    out_path = Path(outdir) / f"wb_asean6_{start}_{end}.csv"
    df.to_csv(out_path, index=False)
    return out_path, df

if __name__ == "__main__":
    path, df = fetch_worldbank()
    print(f"Saved -> {path}")
    print(df.head(12))
