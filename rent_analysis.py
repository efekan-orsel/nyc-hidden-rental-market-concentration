"""
Table 3 rent analysis, hidden-big vs genuinely-small.
(1) rent-stabilized share by owner group (lot level)
(2) listing-level log-rent gap for hidden-big (vs genuinely-small)
(3) same gap after dropping rent-stabilized listings (the null)
Stabilized flag = ever registered stabilized 2007-2017 (uc>0) in the DHCR taxbill file.
"""
import os, re
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))
T = 10

# ---- lot classification (small/hidden/visible) ----
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy()
lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
lot["grp"] = np.where(lot["true_sz"]<T,"small",np.where(lot["app_sz"]<T,"hidden","visible"))

# ---- rent-stabilized flag (ever uc>0, 2007-2017) ----
st = pd.read_csv(p("DiD Data/joined.csv"), low_memory=False)
st["BBL"] = st["ucbbl"].map(bblnorm)
uc_cols = [c for c in st.columns if re.fullmatch(r"20(0[7-9]|1[0-7])uc", str(c))]
st["stab"] = (st[uc_cols].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(axis=1)
stab = st.groupby("BBL")["stab"].any()
lot = lot.merge(stab.rename("stab"), left_on="BBL", right_index=True, how="left")
lot["stab"] = lot["stab"].fillna(False)

# acquisition year (for the acquired-lot variant of the stab share)
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
acqset = set(ac["BBL"].unique())
lot["acq"] = lot["BBL"].isin(acqset)

print("(1) rent-stabilized share by group:")
for g in ("small","hidden"):
    a = 100*lot[lot['grp']==g]['stab'].mean()
    b = 100*lot[(lot['grp']==g) & lot['acq']]['stab'].mean()
    print(f"    {g:7s}: all lots {a:.1f}%   acquired lots {b:.1f}%")

# ---- listings ----
L = pd.read_csv(p("master_listings.csv"), low_memory=False)
L["BBL"] = L["BBL"].map(bblnorm)
L["price"] = pd.to_numeric(L["price"], errors="coerce")
L = L[(L["price"]>0)].merge(lot[["BBL","grp","stab"]], on="BBL", how="inner")
L = L[L["grp"].isin(["small","hidden"])].copy()
L["hidden"] = (L["grp"]=="hidden").astype(int)
L["lnprice"] = np.log(L["price"])
L["bed"] = pd.to_numeric(L["bedrooms"], errors="coerce").fillna(-1)
L["zip"] = L["zip5"].astype(str)

SPECS = {
    "raw (no controls)"      : "lnprice ~ hidden",
    "bedrooms"               : "lnprice ~ hidden + C(bed)",
    "bedrooms + zip"         : "lnprice ~ hidden + C(bed) + C(zip)",
    "bedrooms + zip + units" : "lnprice ~ hidden + C(bed) + C(zip) + lnunits",
}
def gap(frame, formula):
    m = smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame["BBL"]})
    return 100*m.params["hidden"], m.tvalues["hidden"], len(frame)

L["lnunits"] = np.log(pd.to_numeric(L["unitsres_n"], errors="coerce").clip(lower=1))
nonstab = L[L["stab"]==False]
print("\n(2)/(3) listing-level log-rent gap (hidden vs genuinely-small):")
print(f"    {'spec':24s} {'ALL listings':>22s}   {'NON-STABILIZED':>22s}")
for name, f in SPECS.items():
    ba,ta,na = gap(L, f); bn,tn,nn = gap(nonstab, f)
    print(f"    {name:24s}  {ba:+5.1f}% t={ta:+5.2f} n={na:6d}   {bn:+5.1f}% t={tn:+5.2f} n={nn:6d}")
