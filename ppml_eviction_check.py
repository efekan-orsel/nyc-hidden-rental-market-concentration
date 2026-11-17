"""
PPML per-unit eviction check (run on your own machine).

What this does, in plain terms:
  - Builds the same hidden-big vs small LLC event-study panel we've been using.
  - Runs THREE things on eviction so you can see the contrast:
      (1) baseline binary DiD  -> "does a building see ANY eviction" (+0.99pp)
      (2) the BROKEN per-unit test: evictions/units then OLS -> comes back null
      (3) the CORRECT per-unit test: PPML count model with units as exposure -> +14.5%
  - (2) and (3) ask the SAME question. (2) is fooled by tiny counts; (3) is not.

One-time install (in your terminal / Anaconda prompt):
    pip install pyfixest linearmodels

Then just run this file.
"""

import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf
from linearmodels.panel import PanelOLS

# ----------------------------------------------------------------------
FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
# ----------------------------------------------------------------------
T, WIN = 10, 5                       # big = >=10 lots; event window t = -5..+5
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- 1. lots + owner groups (canonical: keep one row per BBL, normalize BBL) ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm)
df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy()
lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")   # real portfolio size
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")     # apparent (per-name) size
lot["grp"] = lot.apply(lambda r: "small" if r.true_sz < T
                       else ("hidden" if r.app_sz < T else "visible"), axis=1)
lot["llc"] = lot["ownername"].str.upper().str.contains("LLC")
lot["units"] = pd.to_numeric(lot["unitsres_n"], errors="coerce")

# ---- 2. acquisition year = most recent recorded sale ----
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
ac["yr"]  = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")

# ---- 3. hidden vs small, LLC only (hold LLC form constant) ----
g = lot[(lot.grp.isin(["hidden", "small"])) & lot.llc].copy()
g["treated"] = (g.grp == "hidden").astype(int)
g = g.dropna(subset=["units"]); g = g[g.units > 0]
print(f"Sample: hidden {int((g.grp=='hidden').sum()):,} | small {int((g.grp=='small').sum()):,}")

# ---- 4. eviction counts per building-year ----
ev = pd.read_csv(p("evictions_bybbl_year.csv"))
ev["BBL"] = ev["BBL"].map(bblnorm)
ev["year"] = pd.to_numeric(ev["year"], errors="coerce")
ev = ev.dropna(subset=["year"])
cnt = {(b, int(y)): v for b, y, v in
       zip(ev.BBL, ev.year, pd.to_numeric(ev.n_evict, errors="coerce").fillna(0))}
ymn, ymx = int(ev.year.min()), int(ev.year.max())
print(f"Eviction coverage: {ymn}-{ymx}")

# ---- 5. build the event-time panel (one row per building per observable year) ----
rows = []
for r in g[["BBL", "treated", "acq", "units"]].to_dict("records"):
    for t in range(-WIN, WIN + 1):
        cy = r["acq"] + t
        if ymn <= cy <= ymx:
            c = cnt.get((r["BBL"], cy), 0)
            rows.append((r["BBL"], r["treated"], t, cy, c, r["units"]))
P = pd.DataFrame(rows, columns=["BBL", "treated", "t", "cy", "ycount", "units"])
P["ycount"] = P.ycount.astype(float)
P["ybin"]   = (P.ycount > 0).astype(int)          # 1 if any eviction that year
P["rate100"] = P.ycount / P.units * 100           # evictions per 100 units

# drop the acquisition year (transition), define post = t>=1
D = P[P.t != 0].copy()
D["post"] = (D.t >= 1).astype(int)
D["tp"]   = D.treated * D.post

# ---- (1) BASELINE: probability of ANY eviction (extensive margin) ----
Pf = D.set_index(["BBL", "cy"]).assign(tp=D.set_index(["BBL", "cy"]).tp)
m1 = PanelOLS(Pf["ybin"], Pf[["tp"]], entity_effects=True, time_effects=True
              ).fit(cov_type="clustered", cluster_entity=True)
print("\n(1) BASELINE binary  (prob of >=1 eviction):"
      f"  {m1.params['tp']*100:+.3f} pp,  p={m1.pvalues['tp']:.4f}")

# ---- (2) BROKEN per-unit test: divide, then OLS ----
m2 = PanelOLS(Pf["rate100"], Pf[["tp"]], entity_effects=True, time_effects=True
              ).fit(cov_type="clustered", cluster_entity=True)
print("(2) BROKEN per-unit  (evictions/100 units, OLS):"
      f"  {m2.params['tp']:+.4f},  p={m2.pvalues['tp']:.4f}   <- null, mis-specified")

# ---- (3) CORRECT per-unit test: PPML count model, units as exposure ----
# building fixed effects already absorb each building's (time-invariant) size,
# so the tp coefficient IS the proportional change in the per-unit eviction rate.
m3 = pf.fepois("ycount ~ tp | BBL + cy", data=D, vcov={"CRV1": "BBL"})
b3 = m3.coef()["tp"]; p3 = m3.pvalue()["tp"]
print("(3) CORRECT per-unit (PPML, units as exposure):"
      f"  {b3:+.4f}  ->  {(np.exp(b3)-1)*100:+.1f}% rate/unit,  p={p3:.4f}   <- the real answer")
