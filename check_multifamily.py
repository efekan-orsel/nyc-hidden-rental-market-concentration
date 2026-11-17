"""
Verify the "~85% of NYC renters are in 3+ unit buildings" claim.
Source: ACS 2024 5-Year, Table B25032 (Tenure by Units in Structure), NYC place.
Reads the Census 'row-labeled' download format.
"""
import os, pandas as pd

FOLDER = r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"   # <-- edit to your path
FILE   = "ACSDT5Y2024.B25032-2026-09-06T164444.csv"          # <-- your B25032 file

d = pd.read_csv(os.path.join(FOLDER, FILE), dtype=str)
d.columns = ["label", "est", "moe"]
d["label"] = d["label"].str.strip()
d["n"] = pd.to_numeric(d["est"].str.replace(",", ""), errors="coerce")

# isolate the RENTER section (rows after "Renter-occupied")
r_start = d.index[d["label"].str.startswith("Renter-occupied")][0]
renter = d.loc[r_start:].copy()
renter_total = renter.iloc[0]["n"]                 # "Renter-occupied housing units:"

def val(lbl): return renter[renter["label"] == lbl]["n"].iloc[0]

one_two = val("1, detached") + val("1, attached") + val("2")
three_plus = val("3 or 4") + val("5 to 9") + val("10 to 19") + val("20 to 49") + val("50 or more")
five_plus  = val("5 to 9") + val("10 to 19") + val("20 to 49") + val("50 or more")

print(f"Renter-occupied households (NYC):       {renter_total:>12,.0f}")
print(f"  in 1-2 unit structures:               {one_two:>12,.0f}  ({100*one_two/renter_total:5.1f}%)")
print(f"  in 3+  unit structures (MULTIFAMILY): {three_plus:>12,.0f}  ({100*three_plus/renter_total:5.1f}%)")
print(f"  in 5+  unit structures:               {five_plus:>12,.0f}  ({100*five_plus/renter_total:5.1f}%)")
