"""
Event-study + difference-in-differences around acquisition.
Hidden-big vs small owners, three outcomes: eviction, emergency repair (OMO), hazardous violation.

For each building we line time up by the year it was bought (t = 0), then track whether it had
>=1 event each year from t = -5 to +5. We compare hidden-big-owned buildings vs small-owned ones.

It prints, per outcome:
  - differential pre-slope  : were the two groups already drifting apart BEFORE the sale? (honesty check)
  - pooled DiD              : after-sale jump, hidden vs small, no controls
  - fixed-effects DiD       : same jump, but each building is its own baseline and every year's
                              common shock (e.g. the COVID eviction freeze) is stripped out  <- the trustworthy one
And it saves one event-study graph per outcome.

Needs: pandas, numpy, statsmodels, matplotlib, linearmodels
  If linearmodels is missing:  pip install linearmodels
"""

import os, re
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
# DISPLAY: this script does NOT set a backend. Run `%matplotlib inline` in a cell
# once (restart the kernel first if figures still won't show), then run this.

# ----------------------------------------------------------------------
# EDIT THIS to your data folder:
FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
# ----------------------------------------------------------------------
T      = 10     # ownership size threshold (>=10 lots = "big")
WIN    = 5      # event-study window: t = -WIN .. +WIN
LLC_ONLY = True # restrict both groups to LLC-held buildings (hidden vs small, LLC held constant)
SAVE   = False  # False = just display in the notebook (no files written). True = also save PNGs to FOLDER.

p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- 1. buildings + owner groups (keep one row per building) ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm)
df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy()
lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")   # real size (unmasked)
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")     # apparent size (per LLC name)
def group(r):
    if r["true_sz"] < T:   return "small"
    return "hidden" if r["app_sz"] < T else "visible"
lot["grp"] = lot.apply(group, axis=1)
lot["is_llc"] = lot["ownername"].str.upper().str.contains("LLC")

# ---- 2. acquisition year = most recent recorded sale ----
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
ac["yr"]  = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")

# ---- keep hidden vs small (optionally LLC-only) ----
g = lot[lot["grp"].isin(["hidden", "small"])].copy()
if LLC_ONLY:
    g = g[g["is_llc"]]
g["treated"] = (g["grp"] == "hidden").astype(int)
print(f"Buildings: hidden {int((g.grp=='hidden').sum())}  |  small {int((g.grp=='small').sum())}"
      f"   (LLC_ONLY={LLC_ONLY})\n")

# ---- 3. one row per building per event-year ----
def build_panel(event_bbl_year, ymin, ymax):
    """event_bbl_year = set of (BBL, calendar_year) that had >=1 event."""
    rows = []
    for r in g[["BBL", "treated", "acq"]].to_dict("records"):
        for t in range(-WIN, WIN + 1):
            cy = r["acq"] + t
            if ymin <= cy <= ymax:                          # only years the data actually covers
                rows.append((r["BBL"], r["treated"], t, cy,
                             int((r["BBL"], cy) in event_bbl_year)))
    return pd.DataFrame(rows, columns=["BBL", "treated", "t", "cy", "y"])

def load_events(fname, bbl_col, yr_col, val_col, year_min=None):
    d = pd.read_csv(p(fname))
    d["BBL"]  = d[bbl_col].map(bblnorm)
    d["year"] = pd.to_numeric(d[yr_col], errors="coerce")
    d = d.dropna(subset=["year"])
    if year_min:                                            # drops junk years (e.g. a stray 217)
        d = d[d["year"] >= year_min]
    hit = d[d[val_col] > 0]
    return set(zip(hit["BBL"], hit["year"].astype(int))), int(d.year.min()), int(d.year.max())

# ---- 4. the three regressions ----
def analyse(panel, name):
    # (a) honesty check: were the groups drifting apart before the sale? (pre-period only)
    pre = panel[(panel.t >= -WIN) & (panel.t <= -1)]
    m_pre = smf.ols("y ~ treated + t + treated:t", pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre["BBL"]})
    slope, slope_p = m_pre.params["treated:t"] * 100, m_pre.pvalues["treated:t"]

    # (b) pooled DiD (drop the sale year; after = t >= 1)
    D = panel[panel.t != 0].copy()
    D["post"] = (D.t >= 1).astype(int)
    m_did = smf.ols("y ~ treated + post + treated:post", D).fit(
        cov_type="cluster", cov_kwds={"groups": D["BBL"]})
    did, did_p = m_did.params["treated:post"] * 100, m_did.pvalues["treated:post"]

    # (c) fixed-effects DiD (each building its own baseline + strip common yearly shocks)
    fe_did, fe_p = np.nan, np.nan
    try:
        from linearmodels.panel import PanelOLS
        Pf = D.set_index(["BBL", "cy"]).copy()
        Pf["tp"] = Pf["treated"] * Pf["post"]
        m_fe = PanelOLS(Pf["y"], Pf[["tp"]], entity_effects=True, time_effects=True
                        ).fit(cov_type="clustered", cluster_entity=True)
        fe_did, fe_p = m_fe.params["tp"] * 100, m_fe.pvalues["tp"]
    except ImportError:
        print("  (install linearmodels for the fixed-effects DiD: pip install linearmodels)")

    verdict = ("PARALLEL before sale (honesty check OK)" if slope_p > 0.10
               else "DIVERGES before sale (gap is confounded by selection)")
    print(f"=== {name} ===")
    print(f"  pre-slope (before sale) : {slope:+.2f} pp/yr, p={slope_p:.3f}  -> {verdict}")
    print(f"  pooled DiD (after jump) : {did:+.2f} pp, p={did_p:.3f}")
    print(f"  fixed-effects DiD       : {fe_did:+.2f} pp, p={fe_p:.3f}   <- report this one")
    return slope, slope_p, fe_did, fe_p

# ---- 5. the graph ----
def plot(panel, name, slope, slope_p, did, did_p, outfile):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for grp_val, colour, label in [(1, "#c0392b", "Hidden-big (LLC)"),
                                    (0, "#2e6da4", "Small LLC")]:
        sub = panel[panel.treated == grp_val]
        agg = sub.groupby("t")["y"].agg(["mean", "count"])
        share = agg["mean"] * 100
        se    = np.sqrt(agg["mean"] * (1 - agg["mean"]) / agg["count"]) * 100  # CI for a proportion
        ax.errorbar(agg.index, share, yerr=1.96 * se, marker="o", capsize=3,
                    color=colour, label=label)
    ax.axvline(-0.5, ls="--", color="grey")           # sale happens between t=-1 and t=0
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Years relative to acquisition (t=0)")
    ax.set_ylabel(f"Share with \u22651 {name.lower()} (%)")
    ax.set_title(f"Event study: {name.lower()} around acquisition")
    ax.legend()
    ax.text(0.02, 0.97,
            f"differential pre-slope: {slope:+.2f} pp/yr, p={slope_p:.3f}\n"
            f"DiD (fixed effects): {did:+.2f} pp, p={did_p:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    fig.tight_layout()
    plt.show()                       # display it in the notebook
    if SAVE:                         # only writes a file when you ask (SAVE = True at top)
        fig.savefig(outfile, dpi=130)
        print(f"  (saved -> {outfile})")
    print()

# ---- 6. run all three ----
OUTCOMES = [
    ("Eviction",            "evictions_bybbl_year.csv", "BBL", "year", "n_evict", None),
    ("Emergency repair",    "omo_bybbl_year.csv",       "BBL", "year", "n_orders", None),
    ("Hazardous violation", "hpd_haz_byyear.csv",       "bbl", "yr",   "n",       2010),
]
for name, fname, bc, yc, vc, ymin_filter in OUTCOMES:
    events, ymin, ymax = load_events(fname, bc, yc, vc, ymin_filter)
    panel = build_panel(events, ymin, ymax)
    slope, slope_p, fe_did, fe_p = analyse(panel, name)
    plot(panel, name, slope, slope_p, fe_did, fe_p,
         p(f"eventstudy_{name.split()[0].lower()}.png"))
