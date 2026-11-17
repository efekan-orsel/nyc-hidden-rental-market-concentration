"""
Table 5 robustness battery, hidden-big vs genuinely-small.
Panel A: threshold sweep (T = 5 / 10 / 20)
Panel B: post-acquisition window length (K = 2 / 4 / 8)
Panel C: public/non-profit vs private hidden-big owners
Panel D: borough split (Staten Island excluded, no post-acq lots)
COVID:   eviction ratio with vs without the 2020-2022 freeze years
Dose-response: among apparent-small lots, harm ~ ln(true portfolio size), with/without controls
Ratio column = hidden / genuinely-small.
"""
import os, re
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
p = lambda f: os.path.join(FOLDER, f)
def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---------- sample (keep-first dedup, resolved fingerprints) ----------
df = pd.read_csv(p("rentals_owner.csv"), dtype=str, low_memory=False)
df["BBL"] = df["BBL"].map(bblnorm); df["ownername"] = df["ownername"].fillna(df["BBL"])
lot = df.drop_duplicates("BBL").copy()
lot = lot[lot["fingerprint"].notna()].copy()
lot["true_sz"] = lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]  = lot.groupby("ownername")["BBL"].transform("nunique")
lot["units"]   = pd.to_numeric(lot["unitsres_n"], errors="coerce")
lot["boro"]    = lot["BBL"].str[0]
PUB = {"C","M","O","X"}
lot["public"]  = lot["ownertype"].fillna("").str.strip().str.upper().isin(PUB)

def classify(T):
    g = np.where(lot["true_sz"] < T, "small", np.where(lot["app_sz"] < T, "hidden", "visible"))
    return pd.Series(g, index=lot.index)

# ---------- acquisition year ----------
ac = pd.read_csv(p("acris_sales_clean.csv"), dtype=str, low_memory=False)
ac["BBL"] = ac["BBL"].map(bblnorm)
ac["yr"]  = pd.to_datetime(ac["date"], errors="coerce").dt.year
acq = ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot = lot.merge(acq.rename("acq"), left_on="BBL", right_index=True, how="left")

OUT = [("hpd_haz_byyear.csv","bbl","yr","hazard"),
       ("omo_bybbl_year.csv","BBL","year","emergency"),
       ("evictions_bybbl_year.csv","BBL","year","eviction")]

# event (BBL, calendar year) long tables
EV = {}
for fn, bc, yc, lab in OUT:
    d = pd.read_csv(p(fn), dtype=str, low_memory=False)
    d["BBL"] = d[bc].map(bblnorm); d["cy"] = pd.to_numeric(d[yc], errors="coerce")
    EV[lab] = d.dropna(subset=["cy"])[["BBL","cy"]]

def postacq_hit(lab, K, drop_years=None):
    """set of BBLs with >=1 event in t=0..K after acquisition (optionally dropping calendar years)."""
    d = EV[lab]
    if drop_years: d = d[~d["cy"].isin(drop_years)]
    j = d.merge(lot.dropna(subset=["acq"])[["BBL","acq"]], on="BBL")
    j["t"] = j["cy"] - j["acq"]
    return set(j[(j["t"]>=0)&(j["t"]<=K)]["BBL"].unique())

def ratio(sub, hitset):
    sub = sub.assign(hit=sub["BBL"].isin(hitset).astype(int))
    s = sub[sub["g"]=="small"]["hit"].mean(); h = sub[sub["g"]=="hidden"]["hit"].mean()
    return 100*s, 100*h, (h/s if s>0 else float("nan"))

# ================= PANEL A: threshold =================
print("="*60); print("PANEL A  threshold sweep (K=4)        haz   emer   evic")
for T in (5,10,20):
    g = classify(T); acq_lot = lot.assign(g=g).dropna(subset=["acq"])
    rr = [ratio(acq_lot, postacq_hit(lab,4))[2] for _,_,_,lab in OUT]
    print(f"  T={T:<3}                        {rr[0]:5.2f}x {rr[1]:4.2f}x {rr[2]:4.2f}x")

# ================= PANEL B: window =================
print("="*60); print("PANEL B  window length (T=10)        haz   emer   evic")
g = classify(10); base = lot.assign(g=g).dropna(subset=["acq"])
for K in (2,4,8):
    rr = [ratio(base, postacq_hit(lab,K))[2] for _,_,_,lab in OUT]
    print(f"  K={K:<3}                        {rr[0]:5.2f}x {rr[1]:4.2f}x {rr[2]:4.2f}x")

# ================= PANEL C: public/private hidden =================
print("="*60); print("PANEL C  hidden split (T=10,K=4)     haz   emer   evic")
g = classify(10); acq_lot = lot.assign(g=g).dropna(subset=["acq"])
for which, mask in (("private hidden", ~acq_lot["public"]), ("public  hidden", acq_lot["public"])):
    sub = acq_lot[(acq_lot["g"]=="small") | ((acq_lot["g"]=="hidden") & mask)]
    rr = [ratio(sub, postacq_hit(lab,4))[2] for _,_,_,lab in OUT]
    print(f"  {which}               {rr[0]:5.2f}x {rr[1]:4.2f}x {rr[2]:4.2f}x")
full = lot.assign(g=classify(10))
print(f"  public share of hidden (full sample): {100*full[full['g']=='hidden']['public'].mean():.1f}%")

# ================= PANEL D: borough =================
print("="*60); print("PANEL D  borough (T=10,K=4)          haz   emer   evic")
BNAME = {"1":"Manhattan","2":"Bronx","3":"Brooklyn","4":"Queens","5":"StatenIs"}
g = classify(10); acq_lot = lot.assign(g=g).dropna(subset=["acq"])
for b in ("1","2","3","4","5"):
    sub = acq_lot[acq_lot["boro"]==b]
    if (sub["g"]=="hidden").sum()==0:
        print(f"  {BNAME[b]:<10} (skipped, no post-acq lots)"); continue
    rr = [ratio(sub, postacq_hit(lab,4))[2] for _,_,_,lab in OUT]
    print(f"  {BNAME[b]:<10}                 {rr[0]:5.2f}x {rr[1]:4.2f}x {rr[2]:4.2f}x")

# ================= COVID eviction freeze =================
print("="*60); print("COVID  eviction ratio (T=10,K=4)")
g = classify(10); acq_lot = lot.assign(g=g).dropna(subset=["acq"])
base_evic = ratio(acq_lot, postacq_hit("eviction",4))[2]
nocovid   = ratio(acq_lot, postacq_hit("eviction",4,drop_years={2020,2021,2022}))[2]
print(f"  with freeze years: {base_evic:.2f}x   without 2020-2022: {nocovid:.2f}x")

# ================= DOSE-RESPONSE =================
print("="*60); print("DOSE-RESPONSE  among apparent-small lots, harm(0/1 post-acq) ~ ln(true size)")
print("  (SEs clustered by owner/fingerprint; class FE = coarse C/D/S)")
g = classify(10); acq_lot = lot.assign(g=g).dropna(subset=["acq"]).copy()
dd = acq_lot[acq_lot["app_sz"]<10].copy()
dd["lntrue"]  = np.log(dd["true_sz"].astype(float))
dd["lnunits"] = np.log(dd["units"].clip(lower=1))
dd["cls"]     = dd["bldgclass"].str[0]
cl = {"groups": dd["fingerprint"]}
for _,_,_,lab in OUT:
    hs = postacq_hit(lab,4)
    dd["y"] = 100*dd["BBL"].isin(hs).astype(int)
    m0 = smf.ols("y ~ lntrue", data=dd).fit(cov_type="cluster", cov_kwds=cl)
    m1 = smf.ols("y ~ lntrue + C(boro) + C(cls) + lnunits", data=dd).fit(cov_type="cluster", cov_kwds=cl)
    print(f"  {lab:10s}  no-ctrl {m0.params['lntrue']:5.2f}pp (t={m0.tvalues['lntrue']:4.1f}) "
          f"| ctrl {m1.params['lntrue']:5.2f}pp (t={m1.tvalues['lntrue']:4.1f})")
