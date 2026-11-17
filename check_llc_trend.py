"""
LLC/corporate share of rental-building PURCHASES over time (NYC, from ACRIS deeds).

Metric per year = (sales whose BUYER name is a corporate/LLC entity) / (all sales that year),
restricted to sales of buildings in the rental-apartment sample.

This is a FLOW measure (who is buying each year), not the standing ownership stock.
The corporate/LLC flag is a name-based heuristic applied identically every year, so the
TREND is robust even if any single year's level is fuzzy.
"""
import os, re
import pandas as pd

FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# 1. universe = sample rental lots
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm)
sample = set(df["BBL"].unique())

# 2. ACRIS deed sales: parse year, normalize BBL
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
ac["yr"]  = pd.to_datetime(ac["date"], errors="coerce").dt.year

# 3. keep only sales of sample buildings, with a valid year
ac = ac[ac["BBL"].isin(sample)].dropna(subset=["yr"]).copy()
ac["yr"] = ac["yr"].astype(int)

# 4. flag buyer as corporate/LLC (non-capturing group -> no pandas warning)
corp = (r"(?:\bLLC\b|L\.L\.C|\bINC\b|CORP|\bLP\b|\bLTD\b|\bLLP\b|COMPANY|HOLDING|"
        r"REALTY|MGMT|MANAGEMENT|PARTNER|PROPERT|ASSOC|GROUP|EQUITIES|CAPITAL|ENTERPRISE)")
ac["corp_buyer"] = ac["buyer"].fillna("").str.upper().str.contains(corp, regex=True)

# 5. share by year
g = (ac[(ac["yr"] >= 2003) & (ac["yr"] <= 2024)]
     .groupby("yr")
     .agg(n_sales=("corp_buyer", "size"), pct_corp=("corp_buyer", "mean")))
g["pct_corp"] = (100 * g["pct_corp"]).round(1)

print("Year   N sales   % buyer is LLC/corporate")
for yr, row in g.iterrows():
    print(f"{yr}   {int(row['n_sales']):>6,}      {row['pct_corp']:.1f}%")

g.to_csv(p("llc_buyer_trend.csv"))
print("\nsaved -> llc_buyer_trend.csv")
