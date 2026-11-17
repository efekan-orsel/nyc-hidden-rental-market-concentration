"""
Table 1 (final) — descriptive summary statistics by ownership group.
Characteristics + portfolio (camouflage) rows only. NO harm rows:
harm is measured post-acquisition (windowed) in the results section, on the
acquired subsample — not as a full-panel descriptive rate here.

Dedup: keep-first (matches the funnel paragraph: 120,869 lots / 84,769 owners).
Groups (threshold T=10 lots): genuinely-small (true<10), hidden-big
(true>=10 & apparent<10), visible-big (true>=10 & apparent>=10).
"""

import os, re
import pandas as pd
import numpy as np

FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"   # <-- edit to your path
T = 10
THIS_YEAR = 2026
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm)
df["unitsres_n"] = pd.to_numeric(df["unitsres_n"], errors="coerce")
df["yearbuilt"]  = pd.to_numeric(df["yearbuilt"],  errors="coerce")
df["ownername"]  = df["ownername"].fillna(df["BBL"])

lot = df.drop_duplicates("BBL").copy()          # keep-first
lot = lot[lot["fingerprint"].notna()].copy()    # drop unresolvable owners

lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")

def group(r):
    if r["true_sz"] < T: return "genuinely-small"
    return "hidden-big" if r["app_sz"] < T else "visible-big"
lot["grp"]  = lot.apply(group, axis=1)
lot["age"]  = THIS_YEAR - lot["yearbuilt"].where(lot["yearbuilt"] > 1800)
lot["cls1"] = lot["bldgclass"].str[0]

def stats(g):
    return pd.Series({
        "Number of lots":              len(g),
        "Residential units (mean)":    round(g["unitsres_n"].mean(), 1),
        "Residential units (median)":  int(g["unitsres_n"].median()),
        "Building age (mean, years)":  round(g["age"].mean(), 1),
        "% Class C (walk-up)":         round(100*(g["cls1"]=="C").mean(), 1),
        "% Class D (elevator)":        round(100*(g["cls1"]=="D").mean(), 1),
        "% Class S (resid. w/ store)": round(100*(g["cls1"]=="S").mean(), 1),
        "True portfolio size (mean)":  round(g["true_sz"].mean(), 1),
        "Apparent portfolio size (mean)": round(g["app_sz"].mean(), 1),
    })

order = ["genuinely-small", "hidden-big", "visible-big"]
tab = pd.DataFrame({g: stats(lot[lot["grp"]==g]) for g in order})
tab["All"] = stats(lot)

pd.set_option("display.width", 140, "display.max_columns", 10)
print(tab.to_string())
tab.to_csv(p("table1_final.csv"))
print("\nsaved -> table1_final.csv")
print("total lots:", len(lot), "| distinct owners:", lot["fingerprint"].nunique())
