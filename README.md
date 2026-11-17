# Hidden Ownership Concentration and Unobserved Harmful Outcomes for Tenants in NYC's Multifamily Rental Market

Replication code for my Columbia University MA thesis.

The project unmasks LLC-held NYC rental buildings back to their true owners, measures how much concentrated ("big") ownership is hidden behind apparently-small LLCs, and estimates the gap in tenant harm, hazardous violations, city emergency repairs, and marshal evictions, between hidden-big owners and genuinely-small ones.

## Method in brief

Each rental lot is linked to a registered owner **name + address** (a "fingerprint"). Lots that share a fingerprint are collapsed into a single true owner, which recovers an owner's true portfolio size even when it is split across many separate LLCs. Owners are then classified by their true size (from the fingerprint) and their apparent size (from the PLUTO owner name), using a threshold of **T = 10** lots:

- **genuinely-small**: true < 10 and apparent < 10
- **hidden-big**: true ≥ 10 but apparent < 10 (big, but looks small)
- **visible-big**: true ≥ 10 and apparent ≥ 10

The headline comparison is hidden-big vs genuinely-small, which holds apparent size roughly fixed and isolates the effect of hidden concentration.

## Scripts

### Data construction
- **construction.py** — Rebuilds the unmasked rental sample from raw sources: merges MDR with Registration Contacts to build the owner name+address fingerprint, joins PLUTO for building class, units, and the apparent owner name, and filters to C/D/S rental buildings with 3+ units (excluding co-ops). Reproduces the sample funnel (202,667 MDR records → 181,318 lots → ~123,153 C/D/S 3+unit lots → ~120,869 resolved, camouflage ~86%). See the fidelity note in the file header: the funnel and camouflage reproduce closely, while the exact owner grouping is sensitive to name normalization.
- **make_classC_byyear.py** — Streams the raw ~6 GB HPD violations file in 500k-row chunks and counts Class-C and Class-(B+C) violations per building-year.
- **funnel_robustness_check.py** — Audits the sample-construction funnel (record/building/lot counts, class composition, leakage checks, fingerprint-resolution rates).

### Descriptive tables and trends
- **check_multifamily.py** — Verifies from Census ACS 2024 that ~85% of NYC renters live in 3+ unit buildings.
- **check_llc_trend.py** / **make_llc_figure.py** — The LLC-buyer share of rental-building purchases over time (Figure 1).
- **table1_final.py** / **table1_summary_stats.py** — Descriptive Table 1 by ownership group (characteristics + portfolio camouflage).

### The harm gap (Table 2)
- **check_ratios.py** — Hidden/small harm ratio for each outcome, full-universe and post-acquisition.
- **check_tstats.py** — The hidden-big vs genuinely-small gap with t-stats clustered by building and by owner; eviction also with controls.

### Rents (Table 3)
- **rent_analysis.py** — Listing-level log-rent gap for hidden-big vs genuinely-small, the rent-stabilized share by group (65.4% vs 24.8% on acquired lots), and the null once rent-stabilized listings are dropped.

### Heterogeneity (Table 4)
- **heterogeneity.py** — The harm gap across neighbourhood income terciles and racial-composition terciles (ACS B19013 + B03002 joined on PLUTO bct2020), reporting both the relative ratio per tercile and how much larger the absolute gap is in the most-disadvantaged tercile.

### Robustness (Table 5)
- **robustness_battery.py** — Threshold sweep (T = 5/10/20), post-acquisition window length (K = 2/4/8), public/non-profit vs private hidden-big split, borough split, the COVID eviction-freeze check, and the continuous dose-response (harm ~ ln true portfolio size, with and without controls).
- **classC_battery.py** — The Class-C (immediately hazardous) robustness: binary DiD, pre-trend Wald test, and the behavioral fork.

### Event-study DiD and the causal result (Figures 2–6)
- **eventstudy_did.py** / **eventstudy_show.py** — The core event-study DiD for all three outcomes (pre-slope honesty check, pooled DiD, two-way fixed-effects DiD).
- **pretrend_show.py** — The three-group pre-trend/selection figure around acquisition (Figure 2).
- **make_5p5_eventstudies.py** / **make_5p5_figures.py** — The Section 5.5 event-study figures, annotated with DiD and pre-trend Wald tests.
- **eviction_figure_ppml.py** — The eviction event study with the binary DiD, joint pre-trend Wald χ²(4), and the per-unit PPML rate (Figure 3).
- **ppml_eviction_check.py** — Contrasts three eviction specifications: the baseline binary DiD (about +0.99pp), the "broken" per-unit OLS test (null), and the correct PPML count model (about +14.5%).
- **fork_figure.py** / **fork_show.py** / **make_fork_figure.py** — The hazard behavioral fork (Figure 6).

## Data

The scripts read public NYC and Census data. The large raw files are **not** included in this repo; download them from the sources below and point the scripts at them.

| File | Source |
|------|--------|
| HPD Multiple Dwelling Registrations (MDR) + Registration Contacts | NYC Dept. of Housing Preservation & Development (NYC Open Data) |
| PLUTO (tax-lot characteristics) | NYC Dept. of City Planning |
| ACRIS deed sales | NYC Dept. of Finance (ACRIS) |
| HPD violations (`wvxf-dwi5`) | NYC Open Data |
| HPD emergency repair / OMO orders | NYC Open Data |
| Marshal evictions (`6z8x-wfk4`) | NYC Open Data |
| ACS 5-Year, Tables B25032 / B19013 / B03002 | U.S. Census Bureau |
| DHCR rent-stabilization registrations (taxbills) | NYC / "Taxbills" rent-stabilization data |

Key derived inputs the analysis scripts expect: `rentals_owner.csv` (the final rental sample, rebuilt by `construction.py`), `acris_sales_clean.csv` (deed sales), the per-building-year outcome files (`evictions_bybbl_year.csv`, `omo_bybbl_year.csv`, `hpd_haz_byyear.csv`, `hpd_classC_byyear.csv`), `master_listings.csv` (rent listings), and `DiD Data/joined.csv` (stabilization).

## Running

Requires Python 3 with: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `linearmodels`, `pyfixest`, and `openpyxl` (for the `.xlsx` MDR file).

```
pip install pandas numpy matplotlib statsmodels linearmodels pyfixest openpyxl
```

Each script sets a `FOLDER` constant near the top pointing at the data directory. Edit it to your local path before running.

## Key parameters

- **T = 10** lots — the big-owner threshold
- **WIN = 5** / **K = 4** — event-study window and post-acquisition harm window
- **LLC-only** — the DiD comparison is restricted to LLC-held lots so the two groups share a legal form
- Acquisition year (t = 0) is dropped from the DiD; post = t ≥ 1
- Dose-response and clustered inference use the owner (fingerprint) as the cluster

---

Efekan Orsel · Columbia University
