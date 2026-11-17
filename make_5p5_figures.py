"""
Makes ALL Section 5.5 figures, with figure numbers defined in ONE place (FIG below).
To renumber, just edit the numbers in FIG -- every title updates automatically.

Figures produced (bulletproof display: saved to PNG, shown via IPython):
  - pre-trend / three-group selection figure
  - eviction / emergency / hazard DiD event studies
  - hidden-big vs small behavioral fork (hazard)

Ordering (edit here):
"""
FIG = {
    "pretrend":  2,   # three-group selection figure (first figure in 5.5)
    "eviction":  3,   # eviction DiD event study
    "emergency": 4,   # emergency-repair DiD event study
    "hazard":    5,   # hazardous-violation DiD event study
    "fork":      6,   # behavioral fork (hazard)
}
# (Figure 1 is your 5.2 headline chart, made elsewhere.)

import os, re
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")                     # Agg always saves; we display the saved PNG
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"   # <-- edit to your data folder
SAVE   = True     # save PNGs to FOLDER too (set False to only display)
# ----------------------------------------------------------------------
T, WIN = 10, 5
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

def show(fig, fname):
    if SAVE: fig.savefig(p(fname), dpi=130)
    else:    fig.savefig("/tmp/_fig.png", dpi=130)
    try:
        from IPython.display import Image, display
        display(Image(filename=p(fname) if SAVE else "/tmp/_fig.png"))
    except Exception:
        print(f"(saved {fname})" if SAVE else "(display unavailable)")
    plt.close(fig)

# ---------- shared data ----------
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy(); lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
lot["small"]   = lot.true_sz < T
lot["hidden"]  = (lot.true_sz >= T) & (lot.app_sz < T)
lot["visible"] = (lot.true_sz >= T) & (lot.app_sz >= T)   # disjoint from hidden, matches the 3-group table
lot["grp"] = lot.apply(lambda r: "small" if r.small else ("hidden" if r.hidden else "visible"), axis=1)
lot["llc"] = lot["ownername"].str.upper().str.contains("LLC")
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm); ac["yr"] = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")

def load_events(fname, bc, yc, vc, ymin=None):
    d = pd.read_csv(p(fname)); d["BBL"] = d[bc].map(bblnorm); d["year"] = pd.to_numeric(d[yc], errors="coerce")
    d = d.dropna(subset=["year"])
    if ymin: d = d[d["year"] >= ymin]
    return set(zip(d[d[vc] > 0]["BBL"], d[d[vc] > 0]["year"].astype(int))), int(d.year.min()), int(d.year.max())

OUT = {"eviction":  ("Eviction",            "evictions_bybbl_year.csv", "BBL", "year", "n_evict",  None),
       "emergency": ("Emergency repair",    "omo_bybbl_year.csv",       "BBL", "year", "n_orders", None),
       "hazard":    ("Hazardous violation", "hpd_haz_byyear.csv",       "bbl", "yr",   "n",        2010)}

# ========== Figure: pre-trend / three groups ==========
def make_pretrend():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    groups = [("hidden", "#c0392b", "Hidden-big (\u226510, looks small)"),
              ("visible", "#2e8b57", "Visible-big (\u226510)"),
              ("small", "#2e6da4", "Small (<10)")]
    titles = {"eviction": "Evictions", "emergency": "City emergency repairs", "hazard": "Hazardous violations"}
    for ax, key in zip(axes, ["emergency", "eviction", "hazard"]):
        name, fn, bc, yc, vc, ymin = OUT[key]
        ev, ymn, ymx = load_events(fn, bc, yc, vc, ymin)
        for col, colour, lab in groups:
            res = {}
            for r in lot[lot[col]][["BBL", "acq"]].to_dict("records"):
                for t in range(-WIN, WIN + 1):
                    cy = r["acq"] + t
                    if ymn <= cy <= ymx: res.setdefault(t, []).append((r["BBL"], cy) in ev)
            ts = sorted(res); ax.plot(ts, [100*np.mean(res[t]) for t in ts], marker="o", ms=4, color=colour, label=lab)
        ax.axvline(-0.5, ls="--", color="grey"); ax.set_title(titles[key]); ax.set_xlabel("Years relative to acquisition")
    axes[0].set_ylabel("% of lots affected"); axes[0].legend(fontsize=8)
    fig.suptitle(f"Figure {FIG['pretrend']}: Hidden-big and visible-big lots both sit far above small ones",
                 fontweight="bold")
    fig.tight_layout(); show(fig, "pretrend_three_groups.png")

# ========== Figures: DiD event studies ==========
def make_eventstudy(key):
    name, fn, bc, yc, vc, ymin = OUT[key]
    ev, ymn, ymx = load_events(fn, bc, yc, vc, ymin)
    g = lot[(lot.grp.isin(["hidden", "small"])) & lot.llc].copy(); g["treated"] = (g.grp == "hidden").astype(int)
    rows = []
    for r in g[["BBL", "treated", "acq"]].to_dict("records"):
        for t in range(-WIN, WIN + 1):
            cy = r["acq"] + t
            if ymn <= cy <= ymx: rows.append((r["BBL"], r["treated"], t, cy, int((r["BBL"], cy) in ev)))
    P = pd.DataFrame(rows, columns=["BBL", "treated", "t", "cy", "y"])
    pre = P[(P.t >= -WIN) & (P.t <= -1)]
    mp = smf.ols("y ~ treated + t + treated:t", pre).fit(cov_type="cluster", cov_kwds={"groups": pre.BBL})
    slope, sp = mp.params["treated:t"]*100, mp.pvalues["treated:t"]
    D = P[P.t != 0].copy(); D["post"] = (D.t >= 1).astype(int)
    did, dp = np.nan, np.nan
    try:
        from linearmodels.panel import PanelOLS
        Pf = D.set_index(["BBL", "cy"]); Pf = Pf.assign(tp=Pf.treated*Pf.post)
        m = PanelOLS(Pf["y"], Pf[["tp"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
        did, dp = m.params["tp"]*100, m.pvalues["tp"]
    except ImportError: pass
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gv, c, lab in [(1, "#c0392b", "Hidden-big (LLC)"), (0, "#2e6da4", "Small LLC")]:
        s = P[P.treated == gv]; agg = s.groupby("t")["y"].agg(["mean", "count"])
        share = agg["mean"]*100; se = np.sqrt(agg["mean"]*(1-agg["mean"])/agg["count"])*100
        ax.errorbar(agg.index, share, yerr=1.96*se, marker="o", capsize=3, color=c, label=lab)
    ax.axvline(-0.5, ls="--", color="grey"); ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Years relative to acquisition (t=0)"); ax.set_ylabel(f"Share with \u22651 {name.lower()} (%)")
    ax.set_title(f"Figure {FIG[key]}: Event study: {name.lower()} around acquisition")
    ax.text(0.02, 0.97, f"pre-trend: {slope:+.2f} pp/yr, p={sp:.3f}\nDiD (FE): {did:+.2f} pp, p={dp:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    ax.legend(); fig.tight_layout(); show(fig, f"eventstudy_{key}.png")
    print(f"Figure {FIG[key]} ({name}): pre-slope {slope:+.2f} p={sp:.3f} | FE DiD {did:+.2f} p={dp:.3f}")

# ========== Figure: behavioral fork (hazard) ==========
def make_fork():
    haz = pd.read_csv(p("hpd_haz_byyear.csv")); haz["BBL"] = haz["bbl"].map(bblnorm)
    haz["year"] = pd.to_numeric(haz["yr"], errors="coerce"); haz = haz[haz.year >= 2010]
    cnt = haz.groupby(["BBL", "year"])["n"].sum(); idx = {(b, y): v for (b, y), v in cnt.items()}
    ymn, ymx = int(haz.year.min()), int(haz.year.max())
    def path(sample):
        out = {}
        for r in sample[["BBL", "acq"]].to_dict("records"):
            by = r["acq"] - 1
            if not (ymn <= by <= ymx): continue
            base = np.log1p(idx.get((r["BBL"], by), 0))
            for t in range(-4, 5):
                cy = r["acq"] + t
                if ymn <= cy <= ymx: out.setdefault(t, []).append(np.log1p(idx.get((r["BBL"], cy), 0)) - base)
        ts = list(range(-4, 5)); return ts, [np.mean(out[t]) for t in ts], [np.std(out[t])/np.sqrt(len(out[t])) for t in ts], len(out[-1])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mask, colour, marker, lab in [((lot.grp == "hidden") & lot.llc, "#c0392b", "s", "HIDDEN-BIG owner (\u226510, looks small)"),
                                       ((lot.grp == "small") & lot.llc, "#2e6da4", "o", "SMALL owner (<10 lots)")]:
        ts, m, se, n = path(lot[mask])
        ax.errorbar(ts, m, yerr=[1.96*s for s in se], marker=marker, capsize=3, color=colour, label=f"{lab}, n={n:,}")
    ax.axvline(-0.5, ls=":", color="red"); ax.axhline(0, ls="--", color="grey")
    ax.set_xlabel("Years relative to purchase"); ax.set_ylabel("Change in ln(1+hazardous) vs year before sale")
    ax.set_title(f"Figure {FIG['fork']}: What happens to a lot after a small vs hidden-big landlord buys it")
    ax.legend(); fig.tight_layout(); show(fig, "fork_hidden_vs_small_hazard.png")

# ---------- run in figure order ----------
make_pretrend()
make_eventstudy("eviction")
make_eventstudy("emergency")
make_eventstudy("hazard")
make_fork()
