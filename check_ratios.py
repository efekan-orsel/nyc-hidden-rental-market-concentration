"""
Check harm ratios yourself: full-universe vs post-acquisition, all three outcomes.
Groups: genuinely-small / hidden-big / visible-big (threshold T=10 lots), keep-first dedup.

Full-universe   = share of ALL resolved lots with >=1 event EVER over the panel.
Post-acquisition = share of ACQUIRED lots with >=1 event in years t=0..4 after the deed sale.
Ratio column = hidden-big / genuinely-small.
"""
import os, re
import pandas as pd

FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
T, K = 10, 4
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- sample + groups (keep-first) ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy()
lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
def grp(r):
    if r["true_sz"] < T: return "small"
    return "hidden" if r["app_sz"] < T else "visible"
lot["grp"] = lot.apply(grp, axis=1)

# ---- acquisition year (most recent ACRIS sale) ----
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
ac["yr"]  = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="left")

OUT = [("hpd_haz_byyear.csv","bbl","yr","hazardous violation"),
       ("omo_bybbl_year.csv","BBL","year","emergency repair"),
       ("evictions_bybbl_year.csv","BBL","year","eviction")]

def rates(frame, member_bbls, label):
    frame = frame.copy(); frame["hit"] = frame["BBL"].isin(member_bbls).astype(int)
    sm = 100*frame[frame["grp"]=="small"]["hit"].mean()
    hb = 100*frame[frame["grp"]=="hidden"]["hit"].mean()
    vb = 100*frame[frame["grp"]=="visible"]["hit"].mean()
    print(f"  {label:20s} small {sm:5.1f}  hidden {hb:5.1f}  visible {vb:5.1f}   ratio(H/S) {hb/sm:.2f}x")

acq_lot = lot.dropna(subset=["acq"]).copy()

print("="*72)
print("FULL-UNIVERSE  (all resolved lots; >=1 event ever)")
print("="*72)
for fn, bc, yc, lab in OUT:
    d = pd.read_csv(p(fn), dtype=str, low_memory=False); d["BBL"] = d[bc].map(bblnorm)
    rates(lot, set(d["BBL"].unique()), lab)

print("\n" + "="*72)
print(f"POST-ACQUISITION  (acquired subsample; >=1 event in t=0..{K})")
print("="*72)
for fn, bc, yc, lab in OUT:
    d = pd.read_csv(p(fn), dtype=str, low_memory=False)
    d["BBL"] = d[bc].map(bblnorm); d["yr"] = pd.to_numeric(d[yc], errors="coerce")
    j = d.dropna(subset=["yr"]).merge(acq_lot[["BBL","acq"]], on="BBL")
    j["t"] = j["yr"] - j["acq"]
    win = set(j[(j["t"]>=0) & (j["t"]<=K)]["BBL"].unique())
    rates(acq_lot, win, lab)

print("\nGroup Ns (full):    ", {g:int((lot['grp']==g).sum()) for g in ['small','hidden','visible']})
print("Group Ns (acquired):", {g:int((acq_lot['grp']==g).sum()) for g in ['small','hidden','visible']})
