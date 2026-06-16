# Analysis code: Mean per-site sequence entropy predicts epistatic regime in the Rogers–Ramanujan partition model of protein families

This repository contains the Python analysis scripts for the paper. The pipeline runs in the order listed below, with each script producing output that feeds into subsequent steps.

---

## Requirements

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
biopython >= 1.81
pandas >= 2.0
```

Install with:

```bash
pip install numpy scipy matplotlib biopython pandas
```

All scripts are self-contained single files with no inter-script imports. Each can be run independently provided its input files are present.

---

## Data

PFAM seed alignments in Stockholm format are required for most scripts. Download from InterPro:

```bash
wget "https://www.ebi.ac.uk/interpro/entry/pfam/PF00017/?annotation=alignment:seed" \
     -O alignments/PF00017.alignment.seed
```

Repeat for each PFAM accession listed in Table 1 of the paper. Alignment files are not included in this repository due to size and licensing constraints.

PDB structure files for the contact density analysis can be downloaded from RCSB:

```bash
wget https://files.rcsb.org/download/1SHC.pdb  # SH2
wget https://files.rcsb.org/download/1SRL.pdb  # SH3
wget https://files.rcsb.org/download/1ATP.pdb  # Protein kinase
wget https://files.rcsb.org/download/1NA0.pdb  # Immunoglobulin
wget https://files.rcsb.org/download/1M40.pdb  # PDZ
wget https://files.rcsb.org/download/1UBQ.pdb  # Ubiquitin
wget https://files.rcsb.org/download/1BE9.pdb  # RRM
```

The Turjanski et al. (2016) supplementary MR coverage data used by `mr_coherence_analysis.py` are hard-coded in that script and require no external files.

---

## Pipeline

### Step 1 — Regime classification across PFAM families

**`pfam_regime_analysis.py`**

Computes the partition model conservation ratio $r_\mathrm{cons} = S_1/S_\mathrm{rand}$ for each family from a PFAM seed alignment, classifies families as finite-$L$ or asymptotic relative to the Rogers–Ramanujan threshold $\sqrt{2/5} \approx 0.6325$, and finds the optimal partition length $L^*$ for finite-$L$ families.

```bash
python pfam_regime_analysis.py \
    --sto-dir ./alignments \
    --out-dir ./results
```

**Outputs:** `pfam_regime_results.csv`, `pfam_regime_plot.png` (Fig 2), `pfam_rcons_distribution.png`

---

### Step 2 — MPSI correlation (S1 Fig)

**`rcons_identity_correlation.py`**

Computes mean pairwise sequence identity (MPSI) from each seed alignment and regresses it against $r_\mathrm{cons}$. Produces the three-panel MPSI correlation figure.

```bash
python rcons_identity_correlation.py \
    --alignment_dir ./alignments \
    --csv results/rcons_identity_correlation.csv \
    --plot results/rcons_identity_correlation.png
```

**Inputs:** PFAM seed alignments  
**Outputs:** `rcons_identity_correlation.csv`, `rcons_identity_correlation.png` (S1 Fig)

---

### Step 3 — Partition model grid search for ANK, LRR, TPR (S2 Fig)

**`partition_L_gridsearch.py`**

Scans $L = 1 \ldots L_\mathrm{max}$ and computes exact finite-$L$ partition model entropy ratios for ANK, LRR, and TPR using dynamic programming. Finds $L^*$ and reports deviation from the Marchi et al. (2019) biological values. Biological values are hard-coded from Table 1 of that paper; no alignment input is required.

```bash
python partition_L_gridsearch.py --L_max 500
```

**Outputs:** `partition_L_gridsearch.csv`, `partition_L_gridsearch.png` (S2 Fig)

---

### Step 4 — L* tautology analysis (S3 Fig)

**`lstar_correlation.py`**

Shows that $L^*$ is a monotone transformation of $r_\mathrm{cons}$ (Spearman $r = 1.0$) and carries no independent information, justifying the use of $r_\mathrm{cons}$ as the primary dependent variable. Family data is hard-coded from `pfam_regime_results.csv`; edit the `FINITE_L_DATA` table at the top of the script to update.

```bash
python lstar_correlation.py
```

**Outputs:** `lstar_correlation.csv`, `lstar_correlation.png` (S3 Fig)

---

### Step 5 — MR coherence length universality (S4 Fig)

**`mr_coherence_analysis.py`**

Extracts the sequence coherence length $\ell^*$ from the Turjanski et al. (2016) supplementary MR coverage data for ANK (natural and synthetic), DEH, and WD40. Produces the median coverage curves showing $\ell^* = 4$ universally across all four families. All data are hard-coded in the script; no external input files are required. Outputs are written to the current working directory.

```bash
cd results && python ../mr_coherence_analysis.py
```

**Inputs:** none (Turjanski et al. Tables S1–S3 data are hard-coded)  
**Outputs:** `mr_coherence_summary.csv`, `mr_coherence_raw.csv`, `mr_coherence_curves.png` (S4 Fig)

---

### Step 6 — Structural contact density (Methods)

**`contact_density_analysis.py`**

Computes per-residue Cβ–Cβ contact density $\rho$ from PDB crystal structures for the 7 finite-$L$ families. Contact defined as Cβ–Cβ distance $\leq 8$ Å with sequence separation $|i-j| \geq 5$; Cα used for glycine.

```bash
python contact_density_analysis.py \
    --local-dir ./pdb \
    --out-dir ./results
```

**Inputs:** PDB files (local or fetched from RCSB with `--local-dir` omitted)  
**Outputs:** `contact_density_results.csv`, `contact_density_results.txt`

**Note:** The 1ATP structure uses chain E (not A) for the kinase catalytic subunit. The script auto-detects the most-populated chain if chain A is empty.

---

### Step 7 — Contact density regression (Methods)

**`rcons_structural_regression.py`**

Multiple OLS regression of $r_\mathrm{cons}$ on MPSI and contact density $\rho$ for the 7 finite-$L$ families. Reports $\Delta R^2$ (incremental variance explained by $\rho$ given MPSI) and residuals. $r_\mathrm{cons}$ and MPSI values are hard-coded in the `KNOWN_DATA` dict at the top of the script.

```bash
python rcons_structural_regression.py \
    --contact-csv results/contact_density_results.csv \
    --out-dir ./results
```

**Inputs:** `contact_density_results.csv`  
**Outputs:** `rcons_structural_regression_results.txt`

---

### Step 8 — Mean per-site entropy analysis (main result)

**`site_entropy_analysis.py`**

Core analysis script. Computes per-site Shannon entropy $H_i$ at each alignment column, family-level mean $\langle H_i \rangle$ and $\sigma(H_i)$, and runs the regression suite: $r_\mathrm{cons} \sim \langle H_i \rangle$, $r_\mathrm{cons} \sim \mathrm{MPSI}$, and joint models. Produces the full regression output table and leave-one-out analysis.

```bash
python site_entropy_analysis.py \
    --sto-dir ./alignments \
    --rcons-csv results/rcons_identity_correlation.csv \
    --out-dir ./results
```

**Inputs:** PFAM seed alignments (`.alignment.seed` or `.alignment.full`; gzip accepted), `rcons_identity_correlation.csv`  
**Outputs:** `site_entropy_results.csv`, `site_entropy_regression.txt`, `site_entropy_regression_data.csv`

**Notes:**

- Alignment columns with $> 50$% gap characters are excluded before entropy computation.
- Gap characters and non-standard residues are excluded from per-position frequency counts.
- The three Marchi et al. repeat families (ANK/PF00023, LRR/PF13516, TPR/PF00515) require their full PFAM alignments (not seed). Download as `.alignment.full` and place in the same directory. Their $r_\mathrm{cons}$ values are hard-coded in the `RCONS_OVERRIDE` dict at the top of the script.
- Glucosidase (PF00232, $n = 9$) is computed but flagged as small-$n$ and excluded from the primary regression.

---

### Step 9 — Main figure (Fig 1)

**`plot_entropy_regression.py`**

Produces the two-panel publication figure: (A) $\langle H_i \rangle$ vs $r_\mathrm{cons}$ with OLS line and 95% CI, points coloured by $\log_{10}(n_\mathrm{seq})$; (B) residuals vs $\log_{10}(n_\mathrm{seq})$ to assess alignment depth bias.

```bash
python plot_entropy_regression.py \
    --entropy-csv results/site_entropy_results.csv \
    --rcons-csv   results/rcons_identity_correlation.csv \
    --out-dir     ./figures
```

**Inputs:** `site_entropy_results.csv`, `rcons_identity_correlation.csv`  
**Outputs:** `entropy_regression_figure.pdf` (for journal submission), `entropy_regression_figure.png` (Fig 1)

---

## Output file index

| File | Produced by | Used in paper |
|---|---|---|
| `pfam_regime_results.csv` | `pfam_regime_analysis.py` | Table 1, Fig 2 |
| `pfam_regime_plot.png` | `pfam_regime_analysis.py` | Fig 2 |
| `rcons_identity_correlation.csv` | `rcons_identity_correlation.py` | S1 Fig, Methods |
| `rcons_identity_correlation.png` | `rcons_identity_correlation.py` | S1 Fig |
| `partition_L_gridsearch.csv` | `partition_L_gridsearch.py` | S2 Fig |
| `partition_L_gridsearch.png` | `partition_L_gridsearch.py` | S2 Fig |
| `lstar_correlation.csv` | `lstar_correlation.py` | S3 Fig |
| `lstar_correlation.png` | `lstar_correlation.py` | S3 Fig |
| `mr_coherence_curves.png` | `mr_coherence_analysis.py` | S4 Fig |
| `contact_density_results.csv` | `contact_density_analysis.py` | Methods |
| `site_entropy_results.csv` | `site_entropy_analysis.py` | Table 1, Methods |
| `site_entropy_regression.txt` | `site_entropy_analysis.py` | Results |
| `site_entropy_regression_data.csv` | `site_entropy_analysis.py` | Methods |
| `entropy_regression_figure.pdf` | `plot_entropy_regression.py` | Fig 1 |

---

## Reproducing all results

```bash
# Create output directories
mkdir -p alignments pdb results figures

# Download alignments (repeat for each accession in Table 1)
# See "Data" section above

# Run pipeline in order
python pfam_regime_analysis.py      --sto-dir ./alignments --out-dir ./results
python rcons_identity_correlation.py --alignment_dir ./alignments \
    --csv results/rcons_identity_correlation.csv \
    --plot results/rcons_identity_correlation.png
python partition_L_gridsearch.py    --L_max 500
python lstar_correlation.py
cd results && python ../mr_coherence_analysis.py && cd ..
python contact_density_analysis.py  --local-dir ./pdb --out-dir ./results
python rcons_structural_regression.py \
    --contact-csv results/contact_density_results.csv --out-dir ./results
python site_entropy_analysis.py     --sto-dir ./alignments \
    --rcons-csv results/rcons_identity_correlation.csv --out-dir ./results
python plot_entropy_regression.py   \
    --entropy-csv results/site_entropy_results.csv \
    --rcons-csv   results/rcons_identity_correlation.csv \
    --out-dir     ./figures
```

Total runtime on a standard laptop: approximately 10–15 minutes, dominated by the sequence reweighting step in `rcons_identity_correlation.py` for the large-$n$ families (WD40, HATPase, HEAT).

---

## Notes on reproducibility

All random operations use deterministic algorithms (dynamic programming, OLS). Results are fully reproducible given the same alignment files. Minor numerical differences ($< 10^{-6}$ in $r_\mathrm{cons}$) may arise from differences in PFAM alignment versions; the seed alignments used in this paper were downloaded in October 2025.

The `partition_L_gridsearch.py` and `lstar_correlation.py` scripts hard-code biological values from Marchi et al. (2019) Table 1 and partition model results from `pfam_regime_results.csv` respectively. These are not recomputed from raw data and will not change unless the hard-coded values are edited.

---

## Citation

[Citation to be added on acceptance]

## Licence

MIT License

Copyright (c) 2026 Paul M. King

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
