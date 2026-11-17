"""
Event-study + DiD, from scratch, with BULLETPROOF display.

The trick: we never rely on matplotlib's interactive backend (that's what kept
getting stuck). We save each figure to a PNG and then show that PNG through
IPython's image display, which always works in a notebook regardless of backend.
"""

import os, re
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")                    # Agg always saves; we display the saved PNG instead
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
# ----------------------------------------------------------------------
T, WIN, LLC_ONLY = 10, 5, True

p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# how we display: save PNG, then show it via IPython (bypasses the backend problem)
def show_png(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)
    try:
        from IPython.display import Image, display
        display(Image(filename=path))
    except Exception:
        print(f"(saved {path})")

# ---- data ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy(); lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
lot["grp"] = lot.apply(lambda r: "small" if r["true_sz"] < T else ("hidden" if r["app_sz"] < T else "visible"), axis=1)
lot["llc"] = lot["ownername"].str.upper().str.contains("LLC")
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm); ac["yr"] = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")

g = lot[lot["grp"].isin(["hidden", "small"])].copy()
if LLC_ONLY: g = g[g["llc"]]
g["treated"] = (g["grp"] == "hidden").astype(int)
print(f"Buildings: hidden {int((g.grp=='hidden').sum())} | small {int((g.grp=='small').sum())}\n")

def load_events(fname, bc, yc, vc, ymin=None):
    d = pd.read_csv(p(fname)); d["BBL"] = d[bc].map(bblnorm); d["year"] = pd.to_numeric(d[yc], errors="coerce")
    d = d.dropna(subset=["year"])
    if ymin: d = d[d["year"] >= ymin]
    return set(zip(d[d[vc] > 0]["BBL"], d[d[vc] > 0]["year"].astype(int))), int(d.year.min()), int(d.year.max())

def build(ev, ymin, ymax):
    rows = []
    for r in g[["BBL", "treated", "acq"]].to_dict("records"):
        for t in range(-WIN, WIN + 1):
            cy = r["acq"] + t
            if ymin <= cy <= ymax:
                rows.append((r["BBL"], r["treated"], t, cy, int((r["BBL"], cy) in ev)))
    return pd.DataFrame(rows, columns=["BBL", "treated", "t", "cy", "y"])

def stats(P):
    pre = P[(P.t >= -WIN) & (P.t <= -1)]
    mp = smf.ols("y ~ treated + t + treated:t", pre).fit(cov_type="cluster", cov_kwds={"groups": pre.BBL})
    D = P[P.t != 0].copy(); D["post"] = (D.t >= 1).astype(int)
    slope, sp = mp.params["treated:t"] * 100, mp.pvalues["treated:t"]
    fe_did, fe_p = np.nan, np.nan
    try:
        from linearmodels.panel import PanelOLS
        Pf = D.set_index(["BBL", "cy"]); Pf = Pf.assign(tp=Pf.treated * Pf.post)
        m = PanelOLS(Pf["y"], Pf[["tp"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
        fe_did, fe_p = m.params["tp"] * 100, m.pvalues["tp"]
    except ImportError:
        pass
    return slope, sp, fe_did, fe_p

def draw(P, name, slope, sp, did, dp, outfile):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gv, c, lab in [(1, "#c0392b", "Hidden-big (LLC)"), (0, "#2e6da4", "Small LLC")]:
        s = P[P.treated == gv]; agg = s.groupby("t")["y"].agg(["mean", "count"])
        share = agg["mean"] * 100; se = np.sqrt(agg["mean"] * (1 - agg["mean"]) / agg["count"]) * 100
        ax.errorbar(agg.index, share, yerr=1.96 * se, marker="o", capsize=3, color=c, label=lab)
    ax.axvline(-0.5, ls="--", color="grey"); ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Years relative to acquisition (t=0)")
    ax.set_ylabel(f"Share with \u22651 {name.lower()} (%)")
    ax.set_title(f"Event study: {name.lower()} around acquisition")
    ax.text(0.02, 0.97, f"pre-slope: {slope:+.2f} pp/yr, p={sp:.3f}\nDiD (FE): {did:+.2f} pp, p={dp:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    ax.legend()
    fig.tight_layout()
    show_png(fig, outfile)

for name, fn, bc, yc, vc, ymin in [
    ("Eviction", "evictions_bybbl_year.csv", "BBL", "year", "n_evict", None),
    ("Emergency repair", "omo_bybbl_year.csv", "BBL", "year", "n_orders", None),
    ("Hazardous violation", "hpd_haz_byyear.csv", "bbl", "yr", "n", 2010),
]:
    ev, ymin_, ymax_ = load_events(fn, bc, yc, vc, ymin)
    P = build(ev, ymin_, ymax_)
    slope, sp, did, dp = stats(P)
    print(f"=== {name} ===\n  pre-slope {slope:+.2f} pp/yr p={sp:.3f} | FE DiD {did:+.2f} pp p={dp:.3f}")
    draw(P, name, slope, sp, did, dp, p(f"eventstudy_{name.split()[0].lower()}.png"))
