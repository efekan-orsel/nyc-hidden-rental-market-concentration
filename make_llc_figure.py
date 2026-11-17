"""
Figure: LLC/corporate share of NYC rental-building purchases, 2003-2024 (ACRIS).
Publication-style single-series line chart. Saves PNG (300dpi) + PDF (vector).

Set LABELS below:
  "none"  -> no point labels (clean; axis carries values)   [recommended]
  "all"   -> a value on every point
  "peaks" -> label only the first point and the peak
"""
import os, re
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
LABELS = "none"                                              # "none" | "all" | "peaks"
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- compute the series ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); sample = set(df["BBL"].unique())
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
ac["yr"] = pd.to_datetime(ac["date"], errors="coerce").dt.year
ac = ac[ac["BBL"].isin(sample)].dropna(subset=["yr"]).copy(); ac["yr"] = ac["yr"].astype(int)
corp = (r"(?:\bLLC\b|L\.L\.C|\bINC\b|CORP|\bLP\b|\bLTD\b|\bLLP\b|COMPANY|HOLDING|"
        r"REALTY|MGMT|MANAGEMENT|PARTNER|PROPERT|ASSOC|GROUP|EQUITIES|CAPITAL|ENTERPRISE)")
ac["corp_buyer"] = ac["buyer"].fillna("").str.upper().str.contains(corp, regex=True)
g = (ac[(ac["yr"] >= 2003) & (ac["yr"] <= 2024)].groupby("yr")["corp_buyer"].mean() * 100)

# ---- plot ----
plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.edgecolor": "#444444", "axes.linewidth": 0.8})
INK = "#1b4965"
fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.plot(g.index, g.values, color=INK, linewidth=1.8, marker="o", markersize=5,
        markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.set_ylim(0, 82); ax.set_xlim(2002.5, 2024.5); ax.set_xticks(range(2003, 2025, 3))
ax.set_ylabel("Purchases with an LLC / corporate buyer (%)")
ax.set_xlabel("Year of purchase")
ax.set_title("Rise of LLC / corporate ownership in NYC rental buildings", fontsize=12, pad=10)

if LABELS == "all":
    for yr, v in g.items():
        ax.annotate(f"{v:.0f}", (yr, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK)
elif LABELS == "peaks":
    peak_yr = g.idxmax()
    ax.annotate(f"{g.loc[2003]:.0f}%", (2003, g.loc[2003]), textcoords="offset points",
                xytext=(20, -4), ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.annotate(f"{g.loc[peak_yr]:.0f}%", (peak_yr, g.loc[peak_yr]), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=10, color=INK, fontweight="bold")
# "none" -> add nothing

fig.tight_layout()
fig.savefig(p("fig_llc_trend.png"), dpi=300, bbox_inches="tight")
fig.savefig(p("fig_llc_trend.pdf"), bbox_inches="tight")
print(f"saved fig_llc_trend.png and .pdf  (LABELS={LABELS})")
