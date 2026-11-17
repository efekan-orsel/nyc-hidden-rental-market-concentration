"""
Filter the raw 6GB HPD violations file down to a small Class-C-by-year summary.
Reads in chunks (low memory), never loads the whole thing.

Outputs two small files into your Thesis folder:
  hpd_classC_byyear.csv   <- Class C (immediately hazardous) only, columns: bbl, yr, n
  hpd_BC_check_byyear.csv <- B+C rebuilt, to confirm it matches your existing hpd_haz_byyear.csv

Just run it (needs pandas, which you already have). Takes a few minutes for 6GB.
When it finishes, tell me the three totals it prints.
"""
import os, re
import pandas as pd
from collections import Counter

FOLDER = "PATH/TO/DATA"   # <-- set to your data folder
RAW = os.path.join(FOLDER, "wvxf-dwi5_version_3357.csv")

def bblnorm(s): return re.sub(r"\D", "", str(s).replace(".0", ""))

# ---- 1. detect the columns we need (names can vary in case) ----
cols = pd.read_csv(RAW, nrows=0).columns.tolist()
low = {c.lower(): c for c in cols}
def find(names):
    for n in names:
        if n in low: return low[n]
    for c in cols:
        if any(n in c.lower() for n in names): return c
    return None

class_col = find(["class"])
bbl_col   = find(["bbl"])
date_col  = None
for pref in ["inspectiondate", "approveddate", "novissueddate", "certifieddate", "currentstatusdate"]:
    date_col = find([pref])
    if date_col: break

print("ALL COLUMNS:", cols)
print(f"USING -> class: {class_col} | bbl: {bbl_col} | date: {date_col}")
assert class_col and date_col, "could not find class/date columns - paste the ALL COLUMNS line to Claude"

usecols = [c for c in [class_col, bbl_col, date_col] if c]

# ---- 2. stream the file, count C and B+C per (bbl, year) ----
cC, cBC = Counter(), Counter()
rows_seen = 0
for chunk in pd.read_csv(RAW, usecols=usecols, dtype=str, chunksize=500_000, low_memory=False):
    rows_seen += len(chunk)
    yr = pd.to_datetime(chunk[date_col], errors="coerce").dt.year
    keep = yr.notna()
    chunk = chunk[keep]; yr = yr[keep].astype(int)
    bbl = chunk[bbl_col].map(bblnorm) if bbl_col else ""
    cl = chunk[class_col].astype(str).str.strip().str.upper()
    isC, isBC = cl.eq("C"), cl.isin(["B", "C"])
    for b, y in zip(bbl[isC], yr[isC]):   cC[(b, y)]  += 1
    for b, y in zip(bbl[isBC], yr[isBC]): cBC[(b, y)] += 1
    print(f"  ...processed {rows_seen:,} rows", end="\r")

# ---- 3. write the two small summaries ----
pd.DataFrame([(b, y, n) for (b, y), n in cC.items()], columns=["bbl", "yr", "n"]) \
  .to_csv(os.path.join(FOLDER, "hpd_classC_byyear.csv"), index=False)
pd.DataFrame([(b, y, n) for (b, y), n in cBC.items()], columns=["bbl", "yr", "n"]) \
  .to_csv(os.path.join(FOLDER, "hpd_BC_check_byyear.csv"), index=False)

print("\n\nDONE.")
print(f"  total rows read      : {rows_seen:,}")
print(f"  Class C total        : {sum(cC.values()):,}")
print(f"  Class B+C total      : {sum(cBC.values()):,}   (your existing hpd_haz_byyear summed to 7,803,404)")
print("  wrote hpd_classC_byyear.csv and hpd_BC_check_byyear.csv to your Thesis folder")
