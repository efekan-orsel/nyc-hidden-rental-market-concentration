"""
Table 1 — descriptive summary statistics by ownership group.
Groups (threshold T=10 lots):
  genuinely-small : true portfolio  < 10
  hidden-big      : true >= 10 AND apparent < 10   (looks small, isn't)
  visible-big     : true >= 10 AND apparent >= 10

Harm rates here are DESCRIPTIVE full-panel "share of lots with >=1 ... ever".
They are NOT the windowed (K=4 post-acquisition) headline numbers.
"""

import os, re
import pandas as pd
import numpy as np

FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
p = lambda f: os.path.join(FOLDER, f)
T = 10          # portfolio-size threshold
THIS_YEAR = 2026

def bblnorm(s):
    return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---------- load sample ----------
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm)
df["unitsres_n"] = pd.to_numeric(df["unitsres_n"], errors="coerce")
df["yearbuilt"]  = pd.to_numeric(df["yearbuilt"],  errors="coerce")
df["ownername"]  = df["ownername"].fillna(df["BBL"])          # avoid NaN owner groups

# one row per lot (dedup); keep latest registration
df["d"] = pd.to_datetime(df["LastRegistrationDate"], errors="coerce")
lot = df.sort_values("d").drop_duplicates("BBL", keep="last").copy()

# drop lots we could not unmask (no fingerprint) -> can't classify
lot = lot[lot["fingerprint"].notna()].copy()

# ---------- portfolio sizes (count distinct lots per owner) ----------
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")

# ---------- group assignment ----------
def group(r):
    if r["true_sz"] < T:                 return "genuinely-small"
    return "hidden-big" if r["app_sz"] < T else "visible-big"
lot["grp"] = lot.apply(group, axis=1)

# ---------- merge harm outcomes: binary "ever >=1 in panel", by BBL ----------
def ever(fname, bblcol, valcol=None):
    d = pd.read_csv(p(fname), dtype=str, low_memory=False)
    d["BBL"] = d[bblcol].map(bblnorm)
    return set(d["BBL"].unique())

haz_bbls  = ever("hpd_haz_byyear.csv",     "bbl")
omo_bbls  = ever("omo_bybbl_year.csv",     "BBL")
evict_bbls= ever("evictions_bybbl_year.csv","BBL")
lot["has_haz"]   = lot["BBL"].isin(haz_bbls).astype(int)
lot["has_omo"]   = lot["BBL"].isin(omo_bbls).astype(int)
lot["has_evict"] = lot["BBL"].isin(evict_bbls).astype(int)

lot["age"] = THIS_YEAR - lot["yearbuilt"].where(lot["yearbuilt"] > 1800)
lot["cls1"] = lot["bldgclass"].str[0]

# ---------- build Table 1 ----------
def stats(g):
    return pd.Series({
        "N lots":              len(g),
        "Mean resid. units":   g["unitsres_n"].mean(),
        "Median resid. units": g["unitsres_n"].median(),
        "Mean building age":   g["age"].mean(),
        "% class C":           100*(g["cls1"]=="C").mean(),
        "% class D":           100*(g["cls1"]=="D").mean(),
        "% class S":           100*(g["cls1"]=="S").mean(),
        "Mean true portfolio": g["true_sz"].mean(),
        "Mean app. portfolio": g["app_sz"].mean(),
        "% >=1 hazard (ever)": 100*g["has_haz"].mean(),
        "% >=1 emerg. repair": 100*g["has_omo"].mean(),
        "% >=1 eviction":      100*g["has_evict"].mean(),
    })

order = ["genuinely-small", "hidden-big", "visible-big"]
tab = pd.DataFrame({grp: stats(lot[lot["grp"]==grp]) for grp in order})
tab["ALL"] = stats(lot)
tab = tab.round(2)

pd.set_option("display.width", 140, "display.max_columns", 10)
print(tab.to_string())
tab.to_csv(p("table1_summary_stats.csv"))
print("\nsaved -> table1_summary_stats.csv")
