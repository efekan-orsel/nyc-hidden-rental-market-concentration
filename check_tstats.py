"""
Check the t-stats yourself: hidden-big vs genuinely-small harm gaps,
post-acquisition (t=0..4, acquired subsample), linear probability model.

For each outcome: gap (percentage points) = hidden - small,
with SEs clustered TWO ways so you can see the difference:
  - by BUILDING (BBL)  -> overstated (treats correlated buildings as independent)
  - by OWNER (fingerprint) -> the honest one (treatment is at owner level)
Eviction also shown with controls (age, units, class, borough), owner-clustered.

Needs statsmodels:  pip install statsmodels
"""
import os, re
import pandas as pd
import statsmodels.formula.api as smf

FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"   # <-- edit to your path
T, K = 10, 4
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
df["unitsres_n"] = pd.to_numeric(df["unitsres_n"], errors="coerce")
df["yearbuilt"]  = pd.to_numeric(df["yearbuilt"],  errors="coerce")
lot = df.drop_duplicates("BBL").copy(); lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
def grp(r):
    if r["true_sz"] < T: return "small"
    return "hidden" if r["app_sz"] < T else "visible"
lot["grp"] = lot.apply(grp, axis=1)

ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm); ac["yr"] = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="left")
al = lot.dropna(subset=["acq"]).copy()

def add_post(fn, bc, yc, lab):
    d = pd.read_csv(p(fn), dtype=str, low_memory=False)
    d["BBL"] = d[bc].map(bblnorm); d["yr"] = pd.to_numeric(d[yc], errors="coerce")
    j = d.dropna(subset=["yr"]).merge(al[["BBL","acq"]], on="BBL"); j["t"] = j["yr"] - j["acq"]
    al[lab] = al["BBL"].isin(set(j[(j["t"]>=0)&(j["t"]<=K)]["BBL"])).astype(int)
for fn, bc, yc, lab in [("hpd_haz_byyear.csv","bbl","yr","haz"),
                        ("omo_bybbl_year.csv","BBL","year","omo"),
                        ("evictions_bybbl_year.csv","BBL","year","ev")]:
    add_post(fn, bc, yc, lab)

# two-group comparison: hidden vs small
d2 = al[al["grp"].isin(["small","hidden"])].copy()
d2["hidden"] = (d2["grp"]=="hidden").astype(int)
d2["age"] = 2026 - d2["yearbuilt"].where(d2["yearbuilt"]>1800)
d2["cls1"] = d2["bldgclass"].str[0]

def tstat(formula, data, lab):
    m = smf.ols(formula, data=data).fit()
    rows = m.model.data.row_labels
    b = m.params["hidden"]; i = list(m.params.index).index("hidden")
    se_b = m.get_robustcov_results(cov_type="cluster", groups=data.loc[rows,"BBL"]).bse[i]
    se_o = m.get_robustcov_results(cov_type="cluster", groups=data.loc[rows,"fingerprint"]).bse[i]
    print(f"  {lab:28s} gap={100*b:5.2f}pp   t(building)={b/se_b:5.1f}   t(OWNER)={b/se_o:5.1f}")

print("HIDDEN-BIG vs GENUINELY-SMALL, post-acquisition (LPM):\n")
tstat("haz ~ hidden", d2, "hazardous violation")
tstat("omo ~ hidden", d2, "emergency repair")
tstat("ev  ~ hidden", d2, "eviction (raw)")
print()
dc = d2.dropna(subset=["age","unitsres_n","cls1","borough"])
tstat("ev ~ hidden + age + unitsres_n + C(cls1) + C(borough)", dc, "eviction (+ controls)")
print(f"\ndistinct owners: hidden={d2[d2['hidden']==1]['fingerprint'].nunique()}, "
      f"small={d2[d2['hidden']==0]['fingerprint'].nunique()}")
