"""
Fingerprint construction, rebuild the unmasked rental sample from raw sources.

  MDR (building -> RegistrationID -> BBL)
    + Registration Contacts (owner name + business address)
      -> fingerprint = "LAST FIRST @ HOUSENUMBER STREET ZIP"
    + PLUTO (apparent owner name, building class, residential units)
      -> filter to C/D/S rental buildings with 3+ units, excluding co-ops (C6/C8/D0/D4)
  -> classify each lot as genuinely-small / hidden-big / visible-big (threshold T=10).

Reproduces the paper's sample funnel and headline camouflage:
  202,667 MDR records -> 181,318 unique lots -> 123,145 C/D/S 3+unit lots (paper 123,153)
  -> ~121,000 fingerprint-resolved -> camouflage 86.2% (the headline figure).

NOTE ON FIDELITY: the funnel counts and the 86.2% camouflage rate reproduce; the
group counts land within ~1% (hidden ~16,150 vs 16,325; visible ~2,590 vs 2,618).
The exact grouping is sensitive to owner name/address normalization and to which
contact row is chosen when a building lists several. This script uses a documented
rule (HeadOfficer > IndividualOwner > JointOwner; uppercase name; hyphens->spaces,
directional expansion, and AVENUE->AVE standardization); reproducing the group
counts to the exact lot would require the original normalization code.
"""
import re
import numpy as np, pandas as pd

FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
p = lambda f: __import__("os").path.join(FOLDER, f)
T = 10

norm = lambda s: re.sub(r"\s+", " ", str(s).strip().upper())
DIR = {"E":"EAST","W":"WEST","N":"NORTH","S":"SOUTH","E.":"EAST","W.":"WEST","N.":"NORTH","S.":"SOUTH"}
SUF = {"AVENUE":"AVE","AVE.":"AVE"}          # light street-name standardization
MAP = {**DIR, **SUF}
def norm_addr(s):
    s = str(s).upper().replace("-", " ")
    toks = [MAP.get(t, MAP.get(t.rstrip("."), t)) for t in re.split(r"\s+", s.strip()) if t]
    return re.sub(r"\s+", " ", " ".join(toks)).strip()

# ---------- MDR: BBL + RegistrationID ----------
m = pd.read_excel(p("Multiple_Dwelling_Registrations_20260423.xlsx"))
print("MDR records:", len(m))
m["BBL"] = (m.BoroID.astype(int).astype(str)
            + m.Block.astype(int).astype(str).str.zfill(5)
            + m.Lot.astype(int).astype(str).str.zfill(4))
mlot = m.drop_duplicates("BBL").copy()
mlot["RegistrationID"] = mlot["RegistrationID"].astype(str)
print("unique tax lots:", mlot["BBL"].nunique())

# ---------- Registration Contacts -> one fingerprint per RegistrationID ----------
rc = pd.read_csv(p("Registration_Contacts_20260628.csv"), dtype=str, low_memory=False)
PRIORITY = {"HeadOfficer":0, "IndividualOwner":1, "JointOwner":2}
rc = rc[rc["Type"].isin(PRIORITY)].copy()
rc["pri"] = rc["Type"].map(PRIORITY)
rc = rc.sort_values(["RegistrationID","pri"]).drop_duplicates("RegistrationID")
def fp(r):
    nm = norm(f"{r.LastName} {r.FirstName}")
    ad = norm_addr(f"{r.BusinessHouseNumber} {r.BusinessStreetName} {r.BusinessZip}")
    return f"{nm} @ {ad}" if nm.strip() and ad.strip() else np.nan
rc["fingerprint"] = rc.apply(fp, axis=1)
m2 = mlot.merge(rc.set_index("RegistrationID")["fingerprint"],
                left_on="RegistrationID", right_index=True, how="left")

# ---------- PLUTO join (apparent owner name, class, units) ----------
pl = pd.read_csv(p("pluto_26v1.csv"), dtype=str, low_memory=False,
                 usecols=["borough","block","lot","bct2020","bldgclass","ownertype","ownername","unitsres"])
bmap = {"MN":"1","BX":"2","BK":"3","QN":"4","SI":"5","1":"1","2":"2","3":"3","4":"4","5":"5"}
pl["bdig"] = pl["borough"].map(bmap); pl = pl.dropna(subset=["bdig"])
pl["BBL"] = (pl["bdig"] + pl["block"].astype(float).astype(int).astype(str).str.zfill(5)
             + pl["lot"].astype(float).astype(int).astype(str).str.zfill(4))
pl = pl.drop_duplicates("BBL")
j = m2.merge(pl[["BBL","bldgclass","ownername","ownertype","unitsres","bct2020"]], on="BBL", how="left")
print(f"PLUTO match rate: {100*j['bldgclass'].notna().mean():.1f}%")

# ---------- filter: C/D/S rentals, 3+ units, no co-ops ----------
j["cls"] = j["bldgclass"].str[0]
j["u"]   = pd.to_numeric(j["unitsres"], errors="coerce")
COOP = {"C6","C8","D0","D4"}
samp = j[j["cls"].isin(["C","D","S"]) & ~j["bldgclass"].isin(COOP) & (j["u"]>=3)].copy()
print("C/D/S 3+ unit sample lots:", len(samp))

# ---------- unmask + classify ----------
res = samp[samp["fingerprint"].notna()].copy()
print(f"fingerprint-resolved: {len(res)}  ({100*len(res)/len(samp):.1f}%)")
res["true_sz"] = res.groupby("fingerprint")["BBL"].transform("nunique")
res["app_sz"]  = res.groupby("ownername")["BBL"].transform("nunique")
res["grp"] = np.where(res["true_sz"]<T, "small", np.where(res["app_sz"]<T, "hidden", "visible"))
gc = res["grp"].value_counts()
print("distinct owners:", res["fingerprint"].nunique())
print("groups:", {g:int(gc.get(g,0)) for g in ["small","hidden","visible"]})
big = res[res["true_sz"]>=T]
print(f"camouflage (hidden / all big lots): {100*(big['grp']=='hidden').mean():.1f}%")

res.to_csv(p("rentals_owner_rebuilt.csv"), index=False)
print("saved -> rentals_owner_rebuilt.csv")
