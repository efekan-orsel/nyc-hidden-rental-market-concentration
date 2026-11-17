"""
Class-C (immediately hazardous) hazard battery, run on hpd_classC_byyear.csv.
Reproduces: binary DiD, joint pre-trend Wald test, and the behavioral fork.
Same LLC-only hidden-vs-small design as the B+C hazard figures, just swapping in Class C.

Needs: pandas, numpy, linearmodels   (pip install linearmodels)
Assumes you already made hpd_classC_byyear.csv (bbl, yr, n) from the raw file.
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS

# ----------------------------------------------------------------------
FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
# ----------------------------------------------------------------------
T, WIN = 10, 5
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- lots + owner groups (canonical) ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy(); lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
lot["grp"] = lot.apply(lambda r: "small" if r.true_sz < T else ("hidden" if r.app_sz < T else "visible"), axis=1)
lot["llc"] = lot["ownername"].str.upper().str.contains("LLC")
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm); ac["yr"] = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")
g = lot[(lot.grp.isin(["hidden", "small"])) & lot.llc].copy()
g["treated"] = (g.grp == "hidden").astype(int)

# ---- Class-C events (from the small summary you built) ----
c = pd.read_csv(p("hpd_classC_byyear.csv")); c["BBL"] = c["bbl"].map(bblnorm)
c["year"] = pd.to_numeric(c["yr"], errors="coerce"); c = c.dropna(subset=["year"])
c = c[c.year >= 2010]
cnt = {(b, int(y)): v for b, y, v in zip(c.BBL, c.year, pd.to_numeric(c.n, errors="coerce").fillna(0))}
S = {k for k, v in cnt.items() if v > 0}
ymn, ymx = int(c.year.min()), int(c.year.max())
print(f"Class C coverage: {ymn}-{ymx}")

rows = []
for r in g[["BBL", "treated", "acq"]].to_dict("records"):
    for t in range(-WIN, WIN + 1):
        cy = r["acq"] + t
        if ymn <= cy <= ymx:
            rows.append((r["BBL"], r["treated"], t, cy, int((r["BBL"], cy) in S)))
P = pd.DataFrame(rows, columns=["BBL", "treated", "t", "cy", "ybin"])

# ---- (1) binary DiD (t=0 excluded, post=t>=1) ----
D = P[P.t != 0].copy(); D["post"] = (D.t >= 1).astype(int)
Pf = D.set_index(["BBL", "cy"]).assign(tp=lambda x: x.treated * x.post)
mb = PanelOLS(Pf["ybin"], Pf[["tp"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
print(f"(1) BINARY DiD (>=1 Class-C viol): {mb.params['tp']*100:+.2f} pp, p={mb.pvalues['tp']:.4f}")

# ---- (2) joint pre-trend Wald ----
ks = [k for k in range(-WIN, WIN+1) if k != -1]
nm = {k: ("m%d" % (-k) if k < 0 else "p%d" % k) for k in ks}
E = P.copy()
for k in ks: E["d_"+nm[k]] = ((E.t == k) & (E.treated == 1)).astype(int)
Ef = E.set_index(["BBL", "cy"])
me = PanelOLS(Ef["ybin"], Ef[["d_"+nm[k] for k in ks]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
wt = me.wald_test(formula=" = ".join(["d_"+nm[k] for k in [-5, -4, -3, -2]]) + " = 0")
print(f"(2) JOINT PRE-TREND: Wald chi2(4)={wt.stat:.2f}, p={wt.pval:.4f}  -> {'PASSES' if wt.pval>0.10 else 'FAILS'}")

# ---- (3) behavioral fork: within-lot change in ln(1+Class C) vs year before sale ----
idx = {(b, int(y)): v for b, y, v in zip(c.BBL, c.year, pd.to_numeric(c.n, errors="coerce").fillna(0))}
def fork(mask):
    out = {}
    for r in lot[mask][["BBL", "acq"]].to_dict("records"):
        by = r["acq"] - 1
        if not (ymn <= by <= ymx): continue
        base = np.log1p(idx.get((r["BBL"], by), 0))
        for t in range(-4, 5):
            cy = r["acq"] + t
            if ymn <= cy <= ymx: out.setdefault(t, []).append(np.log1p(idx.get((r["BBL"], cy), 0)) - base)
    return {t: (np.mean(out[t]), len(out[t])) for t in range(-4, 5)}
H = fork((lot.grp == "hidden") & lot.llc); Sm = fork((lot.grp == "small") & lot.llc)
print(f"(3) FORK (Class C)  hidden n={H[-1][1]:,}  small n={Sm[-1][1]:,}")
print("   t   hidden   small    gap")
for t in range(-4, 5):
    print(f"  {t:+d}   {H[t][0]:+.3f}   {Sm[t][0]:+.3f}   {H[t][0]-Sm[t][0]:+.3f}")
