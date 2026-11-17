"""
Table 4 heterogeneity: harm gap (hidden-big vs genuinely-small) across
neighbourhood income terciles and racial-composition terciles.
Tracts come from PLUTO bct2020, joined to ACS B19013 (median income) and
B03002 (race -> % people of colour). Post-acquisition window K=4.
Reports: relative ratio (hidden/small) per tercile, and how much larger the
ABSOLUTE gap (hidden-small, pp) is in the poorest/most-POC tercile vs the richest/least-POC.
"""
import os, re
import numpy as np, pandas as pd

FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))
T, K = 10, 4
COUNTY = {"1":"061","2":"005","3":"047","4":"081","5":"085"}  # boro digit -> NYC county FIPS

# ---- lots + groups + acquisition ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy()
lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
lot["grp"] = np.where(lot["true_sz"]<T,"small",np.where(lot["app_sz"]<T,"hidden","visible"))

def bct_to_geoid(b):
    b = re.sub(r"\D","",str(b))
    if len(b) < 7: b = b.zfill(7)
    boro, tract = b[0], b[1:7]
    if boro not in COUNTY: return None
    return "1400000US36" + COUNTY[boro] + tract
lot["geoid"] = lot["bct2020"].map(bct_to_geoid)

ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm); ac["yr"] = pd.to_datetime(ac["date"],errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="left")

# ---- ACS income + race per tract ----
inc = pd.read_csv(p("ACSDT5Y2024.B19013-Data.csv"), skiprows=[1])
inc["med_income"] = pd.to_numeric(inc["B19013_001E"], errors="coerce")
inc = inc[["GEO_ID","med_income"]]
rac = pd.read_csv(p("ACSDT5Y2024.B03002-Data.csv"), skiprows=[1])
tot = pd.to_numeric(rac["B03002_001E"], errors="coerce")       # total
nhw = pd.to_numeric(rac["B03002_003E"], errors="coerce")       # non-Hispanic white
rac["pct_poc"] = 100*(1 - nhw/tot)
rac = rac[["GEO_ID","pct_poc"]]

lot = lot.merge(inc, left_on="geoid", right_on="GEO_ID", how="left")
lot = lot.merge(rac, left_on="geoid", right_on="GEO_ID", how="left")

# ---- post-acq harm hit sets ----
OUT = [("hpd_haz_byyear.csv","bbl","yr","hazard"),
       ("omo_bybbl_year.csv","BBL","year","emergency"),
       ("evictions_bybbl_year.csv","BBL","year","eviction")]
HIT = {}
for fn, bc, yc, lab in OUT:
    d = pd.read_csv(p(fn), dtype=str, low_memory=False)
    d["BBL"] = d[bc].map(bblnorm); d["cy"] = pd.to_numeric(d[yc], errors="coerce")
    j = d.dropna(subset=["cy"]).merge(lot.dropna(subset=["acq"])[["BBL","acq"]], on="BBL")
    j["t"] = j["cy"]-j["acq"]
    HIT[lab] = set(j[(j["t"]>=0)&(j["t"]<=K)]["BBL"].unique())

A = lot.dropna(subset=["acq"]).copy()
for lab in HIT: A[lab] = A["BBL"].isin(HIT[lab]).astype(int)

def terciles(frame, col, labels):
    f = frame.dropna(subset=[col]).copy()
    f["ter"] = pd.qcut(f[col], 3, labels=labels)
    return f

def panel(frame, col, labels, title, low, high):
    f = terciles(frame, col, labels)
    print(f"\n{title}")
    print(f"  {'tercile':12s}  {'hazard':>16s} {'emergency':>16s} {'eviction':>16s}")
    absgap = {}
    for ter in labels:
        sub = f[f["ter"]==ter]; row=[]; absgap[ter]={}
        for lab in ("hazard","emergency","eviction"):
            s=100*sub[sub["grp"]=="small"][lab].mean(); h=100*sub[sub["grp"]=="hidden"][lab].mean()
            absgap[ter][lab]=h-s
            row.append(f"{h/s:.2f}x g={h-s:4.1f}")
        print(f"  {str(ter):12s}  {row[0]:>16s} {row[1]:>16s} {row[2]:>16s}")
    print(f"  ABS-gap ratio {high} vs {low}: " +
          "  ".join(f"{lab} {absgap[high][lab]/absgap[low][lab]:.1f}x" for lab in ("hazard","emergency","eviction")))

panel(A, "med_income", ["poorest","mid","richest"],
      "INCOME terciles (relative ratio hidden/small + absolute gap pp):", "richest","poorest")
panel(A, "pct_poc", ["leastPOC","mid","mostPOC"],
      "RACE terciles:", "leastPOC","mostPOC")

f = terciles(A,"pct_poc",["leastPOC","mid","mostPOC"])
print(f"\n  least-POC tercile mean %POC: {f[f['ter']=='leastPOC']['pct_poc'].mean():.0f}%")
print(f"  income-vs-%POC correlation (tract-weighted lots): "
      f"{A[['med_income','pct_poc']].dropna().corr().iloc[0,1]:.2f}")
