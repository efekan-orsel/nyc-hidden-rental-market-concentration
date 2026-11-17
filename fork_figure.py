"""
Behavioral fork: what happens to a building's hazardous violations after a
small vs a hidden-big landlord buys it.

For each building we anchor to the year before it sold (t = -1 -> 0 on the y-axis)
and track the change in ln(1 + hazardous violations) over the next/previous years.
Both groups buy into decline (both dip before the sale); after purchase small owners
level off while hidden-big owners keep deteriorating.

Groups: hidden-big-LLC (true owner >=10 buildings, looks small on the deed) vs small-LLC
(true owner <10). LLC-restricted so it matches the DiD event-study figures.

Needs: pandas, numpy, matplotlib
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# DISPLAY: this script does NOT set a backend. Run `%matplotlib inline` in a cell
# once (restart the kernel first if figures still won't show), then run this.

# ----------------------------------------------------------------------
FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"   # <-- edit to your data folder
# ----------------------------------------------------------------------
T    = 10       # size threshold (>=10 buildings = big)
SAVE = False    # False = just display; True = also save the PNG to FOLDER

p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- buildings + owner groups (one row per building) ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm)
df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy()
lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")   # real size
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")     # apparent size
def group(r):
    if r["true_sz"] < T:   return "small"
    return "hidden" if r["app_sz"] < T else "visible"
lot["grp"] = lot.apply(group, axis=1)
lot["llc"] = lot["ownername"].str.upper().str.contains("LLC")

# ---- acquisition year = most recent sale ----
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
ac["yr"]  = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")

# ---- hazardous violations per building-year ----
haz = pd.read_csv(p("hpd_haz_byyear.csv"))
haz["BBL"]  = haz["bbl"].map(bblnorm)
haz["year"] = pd.to_numeric(haz["yr"], errors="coerce")
haz = haz[haz["year"] >= 2010]                      # drop junk year values
cnt = haz.groupby(["BBL", "year"])["n"].sum()
idx = {(b, y): v for (b, y), v in cnt.items()}
YMIN, YMAX = int(haz.year.min()), int(haz.year.max())

# ---- within-building change path, anchored to the year before sale ----
def path(sample):
    out = {}
    for r in sample[["BBL", "acq"]].to_dict("records"):
        base_year = r["acq"] - 1
        if not (YMIN <= base_year <= YMAX):          # need the t-1 anchor observed
            continue
        base = np.log1p(idx.get((r["BBL"], base_year), 0))
        for t in range(-4, 5):
            cy = r["acq"] + t
            if YMIN <= cy <= YMAX:
                out.setdefault(t, []).append(np.log1p(idx.get((r["BBL"], cy), 0)) - base)
    ts = list(range(-4, 5))
    mean = [np.mean(out[t]) for t in ts]
    se   = [np.std(out[t]) / np.sqrt(len(out[t])) for t in ts]
    n    = len(out[-1])
    return ts, mean, se, n

# ---- the figure (hidden-big red, small blue — matches the DiD figures) ----
fig, ax = plt.subplots(figsize=(9, 5.5))
series = [
    (lot[(lot.grp == "hidden") & lot.llc], "#c0392b", "s", "HIDDEN-BIG owner (\u226510, looks small)"),
    (lot[(lot.grp == "small")  & lot.llc], "#2e6da4", "o", "SMALL owner (<10 bldgs)"),
]
for sample, colour, marker, label in series:
    ts, mean, se, n = path(sample)
    ax.errorbar(ts, mean, yerr=[1.96 * s for s in se], marker=marker, capsize=3,
                color=colour, label=f"{label}, n={n:,}")
ax.axvline(-0.5, ls=":", color="red")       # sale happens between t=-1 and t=0
ax.axhline(0, ls="--", color="grey")
ax.set_xlabel("Years relative to purchase")
ax.set_ylabel("Change in ln(1+hazardous) vs year before sale")
ax.set_title("What happens to a building after a small vs hidden-big landlord buys it")
ax.legend()
fig.tight_layout()
plt.show()
if SAVE:
    out = p("fork_hidden_vs_small_hazard.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")
