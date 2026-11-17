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

### Data preparation
- **make_classC_byyear.py** — Streams the raw roughly 6 GB HPD violations file (`wvxf-dwi5_version_3357.csv`) in 500k-row chunks and counts Class-C and Class-(B+C) violations per building-year, writing `hpd_classC_byyear.csv` and `hpd_BC_check_byyear.csv`.
- **funnel_robustness_check.py** — Rebuilds the sample-construction funnel from the raw MDR (`.xlsx`), the final sample, and the registration contacts, and prints a scoreboard of record/building/lot counts, class composition, leakage checks, and fingerprint-resolution rates.

### Descriptive tables and trends
- **check_multifamily.py** — Verifies from Census ACS 2024 (Table B25032) that most NYC renters live in 3+ unit buildings.
- **check_llc_trend.py** — Computes the yearly share of rental-building purchases (2003–2024) made by an LLC/corporate buyer and writes `llc_buyer_trend.csv`.
- **make_llc_figure.py** — Publication-quality line chart of the LLC-buyer share over time (`fig_llc_trend.png` / `.pdf`).
- **table1_final.py** — Builds the final descriptive Table 1 by ownership group (characteristics + portfolio-camouflage rows), writes `table1_final.csv`.
- **table1_summary_stats.py** — An earlier/fuller Table 1 that also adds full-panel "% with ≥1 event ever" harm rows, writes `table1_summary_stats.csv`.

### The harm gap
- **check_ratios.py** — Classifies lots into the three groups and prints the hidden/small harm ratio for each outcome, computed both full-universe (≥1 event ever) and post-acquisition (≥1 event in t = 0..4).
- **check_tstats.py** — Linear-probability models of the hidden-big vs genuinely-small harm gap (post-acquisition), with t-stats clustered by building and by owner; eviction also run with controls (age, units, class, borough).

### Event-study DiD (the causal section)
- **eventstudy_did.py** — The core event-study / difference-in-differences script. Builds a building × event-year panel (t = −5..+5, hidden-big vs small, LLC-only) and reports, per outcome, a differential pre-slope, a pooled DiD, and a two-way fixed-effects DiD (the reported estimate).
- **eventstudy_show.py** — Same analysis, rendered via the matplotlib Agg backend (display-robustness variant).
- **pretrend_show.py** — The three-group (hidden / all-big / small) pre-trend selection figure around acquisition.
- **make_5p5_eventstudies.py** — Produces the three Section 5.5 event-study figures (eviction, emergency repair, hazardous), each annotated with a DiD estimate and a joint pre-trend Wald test; eviction additionally marked causal via PPML, the other two marked descriptive.
- **make_5p5_figures.py** — Master driver that regenerates all Section 5.5 figures with consistent figure numbering.

### Eviction: the causal result
- **eviction_figure_ppml.py** — Builds the eviction event study and reports the three headline statistics: a binary DiD, a joint pre-trend Wald χ²(4) test, and a per-unit PPML rate (Poisson with units as exposure). Saves Figure 3.
- **ppml_eviction_check.py** — Contrasts three eviction specifications: the baseline binary DiD (about +0.99pp), the "broken" per-unit OLS test (null), and the correct PPML count model (about +14.5%).

### Robustness
- **classC_battery.py** — Repeats the hidden-vs-small design on Class-C (immediately hazardous) violations only: a binary DiD, a joint pre-trend Wald test, and the behavioral fork in ln(1 + Class C).
- **fork_figure.py** / **fork_show.py** / **make_fork_figure.py** — The hazard "behavioral fork": within-building change in ln(1 + hazardous violations) anchored to the year before sale, hidden-big vs small (the last is the polished, publication-styled version).

## Data

The scripts read public NYC and Census data. The large raw files are **not** included in this repo; download them from the sources below and point the scripts at them.

| File | Source |
|------|--------|
| HPD Multiple Dwelling Registrations (MDR) + Registration Contacts | NYC Dept. of Housing Preservation & Development (NYC Open Data) |
| PLUTO (tax-lot characteristics; used upstream) | NYC Dept. of City Planning |
| ACRIS deed sales | NYC Dept. of Finance (ACRIS) |
| HPD violations (`wvxf-dwi5`) | NYC Open Data |
| HPD emergency repair / OMO orders | NYC Open Data |
| Marshal evictions (`6z8x-wfk4`) | NYC Open Data |
| ACS 5-Year, Table B25032 | U.S. Census Bureau |

Key derived inputs the scripts expect: `rentals_owner.csv` (the final rental sample with `BBL`, `ownername`, `fingerprint`, `unitsres_n`), `acris_sales_clean.csv` (deed sales), and the per-building-year outcome files `evictions_bybbl_year.csv`, `omo_bybbl_year.csv`, `hpd_haz_byyear.csv`, `hpd_classC_byyear.csv`.

## Running

Requires Python 3 with: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `linearmodels`, `pyfixest`, and `openpyxl` (for the `.xlsx` MDR file).

```
pip install pandas numpy matplotlib statsmodels linearmodels pyfixest openpyxl
```

Each script sets a `FOLDER` constant near the top pointing at the data directory. Edit it to your local path before running.

## Key parameters

- **T = 10** lots — the big-owner threshold
- **WIN = 5** — event-study window (t = −5..+5)
- **K = 4** — post-acquisition harm window (t = 0..4)
- **LLC-only** — the hidden-vs-small comparison is restricted to LLC-held lots so the two groups share a legal form
- Acquisition year (t = 0) is dropped from the DiD; post = t ≥ 1
- Hazardous / Class-C data filtered to year ≥ 2010; LLC trend covers 2003–2024
- Pre-trend Wald tests use leads t = −5, −4, −3, −2 (the assumption holds when p > 0.10)

---

Efekan Orsel · Columbia University
