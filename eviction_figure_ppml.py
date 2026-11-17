"""
Eviction event-study figure, updated to show ALL THREE headline numbers:
  - joint pre-trend test (Wald chi-square)   <- referee-grade parallel-trends check
  - DiD, any eviction (extensive margin)     <- +0.99 pp
  - DiD, per-unit rate (PPML)                <- +14.5%
Everything is computed live, nothing hardcoded. Hidden-big red, small blue.

One-time install:  pip install pyfixest linearmodels
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
FIGNUM = 3                                                    # eviction figure number
# ----------------------------------------------------------------------
T, WIN = 10, 5
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- lots + groups (canonical) ----
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

# ---- eviction events ----
ev = pd.read_csv(p("evictions_bybbl_year.csv")); ev["BBL"] = ev["BBL"].map(bblnorm)
ev["year"] = pd.to_numeric(ev["year"], errors="coerce"); ev = ev.dropna(subset=["year"])
cnt = {(b, int(y)): v for b, y, v in zip(ev.BBL, ev.year, pd.to_numeric(ev.n_evict, errors="coerce").fillna(0))}
S = {k for k, v in cnt.items() if v > 0}
ymn, ymx = int(ev.year.min()), int(ev.year.max())

rows = []
for r in g[["BBL", "treated", "acq"]].to_dict("records"):
    for t in range(-WIN, WIN + 1):
        cy = r["acq"] + t
        if ymn <= cy <= ymx:
            rows.append((r["BBL"], r["treated"], t, cy, cnt.get((r["BBL"], cy), 0), int((r["BBL"], cy) in S)))
P = pd.DataFrame(rows, columns=["BBL", "treated", "t", "cy", "ycount", "ybin"])
P["ycount"] = P.ycount.astype(float)

# ---- (a) binary DiD (any eviction) ----
D = P[P.t != 0].copy(); D["post"] = (D.t >= 1).astype(int); D["tp"] = D.treated*D.post
Pf = D.set_index(["BBL", "cy"]); Pf = Pf.assign(tp=Pf.treated*Pf.post)
mb = PanelOLS(Pf["ybin"], Pf[["tp"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
did, dp = mb.params["tp"]*100, mb.pvalues["tp"]

# ---- (b) joint pre-trend Wald ----
ks = [k for k in range(-WIN, WIN+1) if k != -1]
nm = {k: ("m%d" % (-k) if k < 0 else "p%d" % k) for k in ks}
E = P.copy()
for k in ks: E["d_"+nm[k]] = ((E.t == k) & (E.treated == 1)).astype(int)
Ef = E.set_index(["BBL", "cy"]); exog = ["d_"+nm[k] for k in ks]
me = PanelOLS(Ef["ybin"], Ef[exog], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
pre = ["d_"+nm[k] for k in [-5, -4, -3, -2]]
wt = me.wald_test(formula=" = ".join(pre)+" = 0")

# ---- (c) PPML per-unit rate ----
mp = pf.fepois("ycount ~ tp | BBL + cy", data=D, vcov={"CRV1": "BBL"})
ppml = (np.exp(mp.coef()["tp"]) - 1)*100; pp = mp.pvalue()["tp"]

# ---- figure ----
fig, ax = plt.subplots(figsize=(9, 5.5))
for gv, c, lab in [(1, "#c0392b", "Hidden-big (LLC)"), (0, "#2e6da4", "Small LLC")]:
    s = P[P.treated == gv]; agg = s.groupby("t")["ybin"].agg(["mean", "count"])
    share = agg["mean"]*100; se = np.sqrt(agg["mean"]*(1-agg["mean"])/agg["count"])*100
    ax.errorbar(agg.index, share, yerr=1.96*se, marker="o", capsize=3, color=c, label=lab)
ax.axvline(-0.5, ls="--", color="grey"); ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Years relative to acquisition (t=0)")
ax.set_ylabel("Share with \u22651 eviction (%)")
ax.set_title(f"Figure {FIGNUM}: Event study: evictions around acquisition")
ax.text(0.02, 0.97,
        f"pre-trend (joint): Wald \u03c7\u00b2(4)={wt.stat:.2f}, p={wt.pval:.2f}\n"
        f"DiD, any eviction: {did:+.2f} pp, p={dp:.3f}\n"
        f"DiD, rate/unit (PPML): {ppml:+.1f}%, p={pp:.3f}",
        transform=ax.transAxes, va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(p("eventstudy_eviction.png"), dpi=130)
print(f"pre-trend joint Wald chi2(4)={wt.stat:.2f} p={wt.pval:.3f} | DiD {did:+.2f}pp p={dp:.3f} | PPML {ppml:+.1f}% p={pp:.3f}")
print("saved eventstudy_eviction.png")
