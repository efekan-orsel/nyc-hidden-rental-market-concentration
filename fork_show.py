"""
Behavioral fork (hidden-big vs genuinely-small), with BULLETPROOF display.
Saves the figure to a PNG and shows it via IPython, so the stuck matplotlib
backend is never in the display path.

Groups: hidden-big-LLC (true >=10, looks small) vs small-LLC (true <10),
matching the DiD event-study figures. Hidden-big red, small blue.
Outcome: within-building change in ln(1+hazardous) vs the year before sale.
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # Agg always saves; we display the saved PNG
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
# ----------------------------------------------------------------------
T = 10
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

def show_png(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)
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
lot["grp"] = lot.apply(lambda r: "small" if r["true_sz"] < T else ("hidden" if r["app_sz"] < T else "visible"), axis=1)
lot["llc"] = lot["ownername"].str.upper().str.contains("LLC")
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm); ac["yr"] = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="inner")

# ---- hazardous violations per building-year ----
haz = pd.read_csv(p("hpd_haz_byyear.csv"))
haz["BBL"] = haz["bbl"].map(bblnorm); haz["year"] = pd.to_numeric(haz["yr"], errors="coerce")
haz = haz[haz["year"] >= 2010]
cnt = haz.groupby(["BBL", "year"])["n"].sum(); idx = {(b, y): v for (b, y), v in cnt.items()}
YMIN, YMAX = int(haz.year.min()), int(haz.year.max())

def path_series(sample):
    out = {}
    for r in sample[["BBL", "acq"]].to_dict("records"):
        by = r["acq"] - 1
        if not (YMIN <= by <= YMAX): continue
        base = np.log1p(idx.get((r["BBL"], by), 0))
        for t in range(-4, 5):
            cy = r["acq"] + t
            if YMIN <= cy <= YMAX:
                out.setdefault(t, []).append(np.log1p(idx.get((r["BBL"], cy), 0)) - base)
    ts = list(range(-4, 5))
    return ts, [np.mean(out[t]) for t in ts], [np.std(out[t]) / np.sqrt(len(out[t])) for t in ts], len(out[-1])

fig, ax = plt.subplots(figsize=(9, 5.5))
for sample, colour, marker, label in [
    (lot[(lot.grp == "hidden") & lot.llc], "#c0392b", "s", "HIDDEN-BIG owner (\u226510, looks small)"),
    (lot[(lot.grp == "small")  & lot.llc], "#2e6da4", "o", "SMALL owner (<10 bldgs)"),
]:
    ts, m, se, n = path_series(sample)
    ax.errorbar(ts, m, yerr=[1.96 * s for s in se], marker=marker, capsize=3, color=colour, label=f"{label}, n={n:,}")
ax.axvline(-0.5, ls=":", color="red"); ax.axhline(0, ls="--", color="grey")
ax.set_xlabel("Years relative to purchase")
ax.set_ylabel("Change in ln(1+hazardous) vs year before sale")
ax.set_title("What happens to a building after a small vs hidden-big landlord buys it")
ax.legend()
fig.tight_layout()
show_png(fig, p("fork_hidden_vs_small_hazard.png"))
