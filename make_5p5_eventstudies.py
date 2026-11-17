"""
Section 5.5 event-study figures, all three, consistent annotation.
  Figure 3 Eviction            : joint pre-trend (PASS) + DiD + PPML per-unit rate
  Figure 4 Emergency repair    : joint pre-trend (FAIL) + DiD, marked DESCRIPTIVE
  Figure 5 Hazardous violation : joint pre-trend (FAIL) + DiD, marked DESCRIPTIVE
Everything computed live. Hidden-big red, small blue.

One-time install:  pip install pyfixest linearmodels
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
FIG = {"eviction": 3, "emergency": 4, "hazard": 5}
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

def load_events(fn, bc, yc, vc, ymin=None):
    d = pd.read_csv(p(fn)); d["BBL"] = d[bc].map(bblnorm); d["year"] = pd.to_numeric(d[yc], errors="coerce")
    d = d.dropna(subset=["year"])
    if ymin: d = d[d.year >= ymin]
    val = pd.to_numeric(d[vc], errors="coerce").fillna(0)
    cnt = {(b, int(y)): v for b, y, v in zip(d.BBL, d.year, val)}
    return cnt, int(d.year.min()), int(d.year.max())

# name, file, bblcol, yrcol, valcol, ymin, causal?
OUT = [
    ("eviction",  "Eviction",            "evictions_bybbl_year.csv", "BBL", "year", "n_evict",  None, True),
    ("emergency", "Emergency repair",     "omo_bybbl_year.csv",       "BBL", "year", "n_orders", None, False),
    ("hazard",    "Hazardous violation",  "hpd_haz_byyear.csv",       "bbl", "yr",   "n",        2010, False),
]

for key, name, fn, bc, yc, vc, ymin, causal in OUT:
    cnt, ymn, ymx = load_events(fn, bc, yc, vc, ymin)
    S = {k for k, v in cnt.items() if v > 0}
    rows = []
    for r in g[["BBL", "treated", "acq"]].to_dict("records"):
        for t in range(-WIN, WIN + 1):
            cy = r["acq"] + t
            if ymn <= cy <= ymx:
                rows.append((r["BBL"], r["treated"], t, cy, cnt.get((r["BBL"], cy), 0.0), int((r["BBL"], cy) in S)))
    P = pd.DataFrame(rows, columns=["BBL", "treated", "t", "cy", "ycount", "ybin"])
    P["ycount"] = P.ycount.astype(float)

    # binary DiD
    D = P[P.t != 0].copy(); D["post"] = (D.t >= 1).astype(int); D["tp"] = D.treated*D.post
    Pf = D.set_index(["BBL", "cy"]).assign(tp=lambda x: x.treated*x.post)
    mb = PanelOLS(Pf["ybin"], Pf[["tp"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    did, dp = mb.params["tp"]*100, mb.pvalues["tp"]

    # joint pre-trend Wald
    ks = [k for k in range(-WIN, WIN+1) if k != -1]
    nm = {k: ("m%d" % (-k) if k < 0 else "p%d" % k) for k in ks}
    E = P.copy()
    for k in ks: E["d_"+nm[k]] = ((E.t == k) & (E.treated == 1)).astype(int)
    Ef = E.set_index(["BBL", "cy"])
    me = PanelOLS(Ef["ybin"], Ef[["d_"+nm[k] for k in ks]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    wt = me.wald_test(formula=" = ".join(["d_"+nm[k] for k in [-5, -4, -3, -2]])+" = 0")
    passes = wt.pval > 0.10
    pstr = f"p={wt.pval:.2f}" if wt.pval >= 0.005 else "p<0.001"

    # annotation
    lines = [f"pre-trend (joint): Wald \u03c7\u00b2(4)={wt.stat:.1f}, {pstr}"
             f"  \u2192 {'PASSES' if passes else 'FAILS'}"]
    if causal:
        mp = pf.fepois("ycount ~ tp | BBL + cy", data=D, vcov={"CRV1": "BBL"})
        ppml = (np.exp(mp.coef()["tp"]) - 1)*100; pp = mp.pvalue()["tp"]
        lines.append(f"DiD, any eviction: {did:+.2f} pp, p={dp:.3f}")
        lines.append(f"DiD, rate/unit (PPML): {ppml:+.1f}%, p={pp:.3f}")
    else:
        lines.append(f"DiD: {did:+.2f} pp  (DESCRIPTIVE \u2014 fails pre-trend)")

    # figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gv, c, lab in [(1, "#c0392b", "Hidden-big (LLC)"), (0, "#2e6da4", "Small LLC")]:
        s = P[P.treated == gv]; agg = s.groupby("t")["ybin"].agg(["mean", "count"])
        share = agg["mean"]*100; se = np.sqrt(agg["mean"]*(1-agg["mean"])/agg["count"])*100
        ax.errorbar(agg.index, share, yerr=1.96*se, marker="o", capsize=3, color=c, label=lab)
    ax.axvline(-0.5, ls="--", color="grey"); ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Years relative to acquisition (t=0)")
    ax.set_ylabel(f"Share with \u22651 {name.lower()} (%)")
    ax.set_title(f"Figure {FIG[key]}: Event study: {name.lower()} around acquisition")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(p(f"eventstudy_{key}.png"), dpi=130)
    plt.close(fig)
    print(f"Figure {FIG[key]} {name}: joint Wald chi2(4)={wt.stat:.1f} {pstr} {'PASS' if passes else 'FAIL'} | DiD {did:+.2f}pp p={dp:.3f}")
