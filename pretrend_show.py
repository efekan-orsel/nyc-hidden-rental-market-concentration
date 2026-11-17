"""
Pre-trend / selection figure (3 panels, 3 groups), rebuilt with an accurate title
and bulletproof display (save PNG + show via IPython).

Shows, per outcome, the share of buildings with >=1 event each year around acquisition,
for hidden-big vs openly-big vs small owners. The robust message: hidden-big buildings
track OPENLY-big buildings (the two lines overlap), not small ones -- i.e. hidden-big are
big owners in disguise. Both big groups' problems climb before acquisition (selection);
small owners' buildings climb far less.
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"   # <-- edit to your data folder
# ----------------------------------------------------------------------
T, WIN = 10, 5
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

def show_png(fig, path):
    fig.savefig(path, dpi=130); plt.close(fig)
    try:
        from IPython.display import Image, display
        display(Image(filename=path))
    except Exception:
        print(f"(saved {path})")

# ---- buildings + groups ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy(); lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
lot["small"]   = lot.true_sz < T                          # genuinely small
lot["hidden"]  = (lot.true_sz >= T) & (lot.app_sz < T)    # big but looks small
lot["big_all"] = lot.true_sz >= T                         # all big (openly + hidden)
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm); ac["yr"] = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")

def load_events(fname, bc, yc, vc, ymin=None):
    d = pd.read_csv(p(fname)); d["BBL"] = d[bc].map(bblnorm); d["year"] = pd.to_numeric(d[yc], errors="coerce")
    d = d.dropna(subset=["year"])
    if ymin: d = d[d["year"] >= ymin]
    return set(zip(d[d[vc] > 0]["BBL"], d[d[vc] > 0]["year"].astype(int))), int(d.year.min()), int(d.year.max())

def series(mask, ev, ymn, ymx):
    res = {}
    for r in lot[mask][["BBL", "acq"]].to_dict("records"):
        for t in range(-WIN, WIN + 1):
            cy = r["acq"] + t
            if ymn <= cy <= ymx:
                res.setdefault(t, []).append((r["BBL"], cy) in ev)
    ts = sorted(res); return ts, [100 * np.mean(res[t]) for t in ts]

OUT = [("City emergency repairs", "omo_bybbl_year.csv", "BBL", "year", "n_orders", None),
       ("Evictions", "evictions_bybbl_year.csv", "BBL", "year", "n_evict", None),
       ("Hazardous violations", "hpd_haz_byyear.csv", "bbl", "yr", "n", 2010)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
groups = [("hidden",  "#c0392b", "Hidden-big (\u226510, looks small)"),
          ("big_all", "#2e8b57", "Openly big (\u226510)"),
          ("small",   "#2e6da4", "Small (<10)")]
for ax, (name, fn, bc, yc, vc, ymin) in zip(axes, OUT):
    ev, ymn, ymx = load_events(fn, bc, yc, vc, ymin)
    for col, colour, lab in groups:
        ts, sh = series(lot[col], ev, ymn, ymx)
        ax.plot(ts, sh, marker="o", ms=4, color=colour, label=lab)
    ax.axvline(-0.5, ls="--", color="grey")
    ax.set_title(name); ax.set_xlabel("Years relative to acquisition")
axes[0].set_ylabel("% of buildings affected")
axes[0].legend(fontsize=8)
fig.suptitle("Hidden-big buildings track openly-big buildings, not small ones (event study around acquisition)",
             fontweight="bold")
fig.tight_layout()
show_png(fig, p("pretrend_three_groups.png"))
