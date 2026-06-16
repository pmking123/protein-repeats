#!/usr/bin/env python3
"""
site_entropy_analysis.py
=========================
Computes per-site Shannon entropy statistics from PFAM seed alignments
and regresses them against r_cons and MPSI in the regime classification
framework.

Quantities computed per family
-------------------------------
  H_i     = -sum_a f_ia * log2(f_ia)   (single-site entropy at position i)
             gap characters excluded from frequency calculation
             positions with >50% gaps dropped (same filter as DCA pipeline)

  mean_H  = mean(H_i) over positions    -- average variability
  std_H   = std(H_i)  over positions    -- heterogeneity of constraint
  frac_conserved = fraction of positions with H_i < 0.5 bits
                   (roughly: <10% of positions mutated)

Regression models
-----------------
  (1) r_cons ~ MPSI                      (baseline, n=14 or 13)
  (2) r_cons ~ MPSI + mean_H             (does mean entropy add to MPSI?)
  (3) r_cons ~ MPSI + std_H              (does positional heterogeneity add?)
  (4) r_cons ~ MPSI + mean_H + std_H     (full entropy model)

The key question: does std_H or mean_H add explanatory power (dR2) beyond
MPSI, and is the added predictor significant given n=13-14?

INPUTS
------
  Stockholm alignment files: <sto_dir>/PF00042.alignment.seed etc.
  rcons_identity_correlation.csv

OUTPUTS
-------
  site_entropy_results.csv     -- per-family entropy statistics
  site_entropy_regression.txt  -- regression table

USAGE
-----
  python site_entropy_analysis.py \
      --sto-dir ./alignments \
      --rcons-csv rcons_identity_correlation.csv \
      --out-dir ./results
"""

import argparse
import csv
import os
import sys
from typing import Optional

import numpy as np
from scipy import stats

# -- Alphabet --------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"   # 20 standard; gaps handled separately
AA_SET = set(AA)

# -- Stockholm parser (same as dca_coupling_analysis.py) ------------------

def parse_stockholm(path: str) -> tuple[list[str], list[str]]:
    """
    Parse Stockholm alignment. Handles both plain text and gzip-compressed files.
    Skips markup lines (#, //). Concatenates multi-block sequences.
    """
    import gzip
    seqs: dict[str, list[str]] = {}
    order: list[str] = []

    # Detect gzip by magic bytes
    with open(path, "rb") as fb:
        magic = fb.read(2)
    is_gz = (magic == b"\x1f\x8b")

    opener = gzip.open(path, "rt", encoding="utf-8", errors="replace") if is_gz              else open(path, encoding="utf-8", errors="replace")

    with opener as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or line.startswith("//") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, seq = parts[0], parts[1]
            # Skip lines where seq field looks like a non-sequence annotation
            if any(c.isdigit() for c in seq) and not any(
                    c in "ACDEFGHIKLMNPQRSTVWY-." for c in seq.upper()):
                continue
            if name not in seqs:
                seqs[name] = []
                order.append(name)
            seqs[name].append(seq)

    sequences = ["".join(seqs[n]) for n in order]

    # Validate: all sequences same length (required for column indexing)
    if sequences:
        L = len(sequences[0])
        sequences = [s for s in sequences if len(s) == L]
        if not sequences:
            raise ValueError("No sequences with consistent length found")

    return order[:len(sequences)], sequences


# -- Per-site entropy ------------------------------------------------------

def site_entropy(col: list[str]) -> Optional[float]:
    """
    Shannon entropy (bits) at one alignment column.
    Gaps and non-standard characters are excluded from the frequency
    calculation. Returns None if the column has no standard AA characters.
    """
    counts: dict[str, int] = {}
    for aa in col:
        aa = aa.upper()
        if aa in AA_SET:
            counts[aa] = counts.get(aa, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return None

    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * np.log2(p)
    return H


def compute_site_entropies(sequences: list[str],
                           max_gap_frac: float = 0.5
                           ) -> tuple[np.ndarray, int]:
    """
    Compute per-site entropy for all columns passing the gap filter.
    Returns (H_array, n_cols_dropped).
    """
    if not sequences:
        return np.array([]), 0

    L = len(sequences[0])
    M = len(sequences)

    H_vals = []
    n_dropped = 0

    for j in range(L):
        col = [s[j] for s in sequences]
        gap_frac = sum(1 for c in col if c not in AA_SET) / M
        if gap_frac > max_gap_frac:
            n_dropped += 1
            continue
        h = site_entropy(col)
        if h is not None:
            H_vals.append(h)

    return np.array(H_vals), n_dropped


# -- PFAM family panel -----------------------------------------------------

FAMILY_PFAM = {
    "TEM-1 (PF13354)":       "PF13354",
    "PDZ (PF00595)":         "PF00595",
    "RRM (PF00076)":         "PF00076",
    "Glucosidase (PF00232)": "PF00232",
    "Prot.kinase (PF00069)": "PF00069",
    "SH2 (PF00017)":         "PF00017",
    "SH3 (PF00018)":         "PF00018",
    "Globin (PF00042)":      "PF00042",
    "Ig (PF00047)":          "PF00047",
    "Ferredoxin (PF00111)":  "PF00111",
    "Ubiquitin (PF00240)":   "PF00240",
    "HATPase (PF02518)":     "PF02518",
    "WD40 (PF00400)":        "PF00400",
    "HEAT (PF02985)":        "PF02985",
    # Repeat families from Marchi et al. -- provide full alignments
    # Download: wget "https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00023/?annotation=alignment:full&download" -O PF00023.alignment.full
    "ANK (PF00023)":         "PF00023",
    "LRR (PF13516)":         "PF13516",
    "TPR (PF00515)":         "PF00515",
}

# r_cons from bio_cons values in partition_L_gridsearch.py (Marchi et al. alignments).
# MPSI set to None -- not computed for these families (full alignments too large for
# pairwise identity calculation; use mean_H as the primary predictor instead).
RCONS_OVERRIDE = {
    "ANK (PF00023)": {"r_cons": 0.6241, "mpsi": None, "source": "computed_marchi"},
    "LRR (PF13516)": {"r_cons": 0.6161, "mpsi": None, "source": "computed_marchi"},
    "TPR (PF00515)": {"r_cons": 0.5652, "mpsi": None, "source": "computed_marchi"},
}


# -- OLS helpers -----------------------------------------------------------

def ols(X_pred: np.ndarray, y: np.ndarray):
    """OLS with intercept. Returns (beta, R2, residuals, SE, t, p)."""
    n = len(y)
    Xb = np.column_stack([np.ones(n), X_pred])
    beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
    yhat = Xb @ beta
    resid = y - yhat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = n - Xb.shape[1]
    if dof > 0:
        s2 = ss_res / dof
        XtXinv = np.linalg.pinv(Xb.T @ Xb)
        se = np.sqrt(np.diag(XtXinv) * s2)
        t_stat = beta / se
        p_val = np.array([2 * stats.t.sf(abs(t), df=dof) for t in t_stat])
    else:
        se = np.full_like(beta, np.nan)
        t_stat = np.full_like(beta, np.nan)
        p_val = np.full_like(beta, np.nan)
    return beta, r2, resid, se, t_stat, p_val


def print_model(label: str, predictor_names: list[str],
                beta, r2, resid, se, t_stat, p_val,
                families: list[str], y: np.ndarray, dof: int):
    print(f"\n  Model: {label}")
    print(f"  R2 = {r2:.4f}   RMSE = {np.sqrt(np.sum(resid**2)/max(dof,1)):.4f}")
    print(f"  {'Param':<14} {'beta':>9} {'SE':>9} {'t':>7} {'p':>8}")
    for name, b, s, t, p in zip(["intercept"] + predictor_names,
                                  beta, se, t_stat, p_val):
        sig = " *" if p < 0.05 else ("  " if np.isnan(p) else "  ")
        print(f"  {name:<14} {b:>9.4f} {s:>9.4f} {t:>7.3f} {p:>8.4f}{sig}")


# -- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-site entropy analysis for regime classification")
    parser.add_argument("--sto-dir",    default="./alignments")
    parser.add_argument("--rcons-csv",  default="rcons_identity_correlation.csv")
    parser.add_argument("--out-dir",    default=".")
    parser.add_argument("--max-gap",    type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -- Load r_cons / MPSI ---------------------------------------------
    rcons_data = {}
    if os.path.exists(args.rcons_csv):
        with open(args.rcons_csv) as f:
            for row in csv.DictReader(f):
                rcons_data[row["name"]] = {
                    "r_cons":  float(row["r_cons"]),
                    "mpsi":    float(row["mpsi"]) if row["mpsi"] != "" else None,
                    "source":  row.get("source", ""),
                }
        print(f"Loaded {len(rcons_data)} families from {args.rcons_csv}")
    else:
        print(f"WARNING: {args.rcons_csv} not found")
    # Merge hardcoded r_cons values for repeat families not in rcons_identity_correlation.csv
    for fam, vals in RCONS_OVERRIDE.items():
        if fam not in rcons_data or rcons_data[fam]["source"] == "estimated":
            rcons_data[fam] = vals
            print(f"  Override: {fam} r_cons={vals['r_cons']} ({vals['source']})")

    # -- Compute entropy statistics -------------------------------------
    print(f"\nComputing per-site entropies (max_gap={args.max_gap})\n")
    print(f"{'Family':<25} {'n_seq':>6} {'L_filt':>7} {'mean_H':>8} "
          f"{'std_H':>7} {'frac_cons':>10} {'status'}")
    print("-" * 75)

    entropy_results = []

    for family_name, pfam_id in FAMILY_PFAM.items():
        sto_path = None
        for fname in [f"{pfam_id}_seed.sto", f"{pfam_id}.sto",
                      f"{pfam_id.lower()}_seed.sto",
                      f"{pfam_id}.alignment.seed",
                      f"{pfam_id}.alignment.full"]:
            candidate = os.path.join(args.sto_dir, fname)
            if os.path.exists(candidate):
                sto_path = candidate
                break

        if sto_path is None:
            print(f"{family_name:<25}  MISSING: {pfam_id}.alignment.seed / .full not found in {args.sto_dir}")
            entropy_results.append({
                "family": family_name, "pfam": pfam_id,
                "n_seq": None, "L_filt": None,
                "mean_H": None, "std_H": None,
                "frac_conserved": None, "H_max": None,
                "H_min": None, "status": "missing"
            })
            continue

        try:
            names, seqs = parse_stockholm(sto_path)
            if not seqs:
                raise ValueError("Empty alignment")

            H_arr, n_dropped = compute_site_entropies(seqs, args.max_gap)
            if len(H_arr) == 0:
                raise ValueError("No columns passed gap filter")

            mean_H  = float(H_arr.mean())
            std_H   = float(H_arr.std(ddof=1)) if len(H_arr) > 1 else 0.0
            frac_cons = float((H_arr < 0.5).mean())

            print(f"{family_name:<25} {len(seqs):>6} {len(H_arr):>7} "
                  f"{mean_H:>8.4f} {std_H:>7.4f} {frac_cons:>10.4f}   ok")

            entropy_results.append({
                "family":          family_name,
                "pfam":            pfam_id,
                "n_seq":           len(seqs),
                "L_filt":          len(H_arr),
                "mean_H":          round(mean_H, 6),
                "std_H":           round(std_H, 6),
                "frac_conserved":  round(frac_cons, 6),
                "H_max":           round(float(H_arr.max()), 6),
                "H_min":           round(float(H_arr.min()), 6),
                "status":          "ok",
            })

        except Exception as e:
            # Print first raw bytes to help diagnose format issues
            if sto_path:
                with open(sto_path, "rb") as _fb:
                    _head = _fb.read(120)
                print(f"{family_name:<25}  ERROR: {e}")
                print(f"  File: {sto_path}")
                print(f"  First bytes: {_head[:80]}")
            else:
                print(f"{family_name:<25}  ERROR: {e}")
            entropy_results.append({
                "family": family_name, "pfam": pfam_id,
                "n_seq": None, "L_filt": None,
                "mean_H": None, "std_H": None,
                "frac_conserved": None,
                "H_max": None, "H_min": None,
                "status": f"error: {e}"
            })

    # -- Write entropy CSV ----------------------------------------------
    csv_path = os.path.join(args.out_dir, "site_entropy_results.csv")
    fieldnames = ["family","pfam","n_seq","L_filt","mean_H","std_H",
                  "frac_conserved","H_max","H_min","status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entropy_results)
    print(f"\nEntropy results written to {csv_path}")

    # -- Assemble regression dataset ------------------------------------
    ok = [r for r in entropy_results if r["status"] == "ok"]
    reg_rows = []
    for r in ok:
        fam = r["family"]
        if fam in rcons_data and rcons_data[fam]["source"] in ("computed", "computed_marchi"):
            reg_rows.append({
                "family":  fam,
                "r_cons":  rcons_data[fam]["r_cons"],
                "mpsi":    rcons_data[fam]["mpsi"],
                "mean_H":  r["mean_H"],
                "std_H":   r["std_H"],
                "frac_conserved": r["frac_conserved"],
                "n_seq":   r["n_seq"],
                "L_filt":  r["L_filt"],
            })

    n = len(reg_rows)
    if n < 4:
        print(f"\nOnly {n} families with complete data - skipping regression")
        return

    print(f"\n{n} families with complete data for regression")

    families = [r["family"]  for r in reg_rows]
    rc   = np.array([r["r_cons"]  for r in reg_rows])
    mpsi = np.array([r["mpsi"] if r["mpsi"] is not None else np.nan
                       for r in reg_rows], dtype=float)
    mH   = np.array([r["mean_H"]  for r in reg_rows])
    sH   = np.array([r["std_H"]   for r in reg_rows])
    fc   = np.array([r["frac_conserved"] for r in reg_rows])

    # -- Regression output ----------------------------------------------
    txt_path = os.path.join(args.out_dir, "site_entropy_regression.txt")
    import io
    buf = io.StringIO()

    def out(s=""):
        print(s)
        buf.write(s + "\n")

    out(f"\n{'='*65}")
    out(f"Per-site entropy regression: r_cons ~ predictors  (n={n})")
    out(f"{'='*65}")

    # Pearson correlations (skip MPSI rows where NaN present)
    mpsi_mask = ~np.isnan(mpsi)
    n_mpsi = mpsi_mask.sum()
    out(f"\nPearson correlations with r_cons:")
    for name, vec in [("MPSI", mpsi), ("mean_H", mH),
                       ("std_H", sH), ("frac_conserved", fc)]:
        if name == "MPSI":
            if n_mpsi < 4:
                out(f"  {name:<18} (skipped -- fewer than 4 non-NaN values)")
                continue
            r_val, p_val = stats.pearsonr(vec[mpsi_mask], rc[mpsi_mask])
            out(f"  {name:<18} r = {r_val:>7.4f}   p = {p_val:.4f}  (n={n_mpsi})")
        else:
            r_val, p_val = stats.pearsonr(vec, rc)
            out(f"  {name:<18} r = {r_val:>7.4f}   p = {p_val:.4f}")

    out(f"\nPredictor intercorrelations:")
    for (n1, v1), (n2, v2) in [
        (("MPSI", mpsi), ("mean_H", mH)),
        (("MPSI", mpsi), ("std_H",  sH)),
        (("mean_H", mH), ("std_H",  sH)),
    ]:
        if "MPSI" in (n1, n2):
            if n_mpsi < 4:
                continue
            mask = mpsi_mask
            r_val, p_val = stats.pearsonr(v1[mask], v2[mask])
            out(f"  {n1} ~ {n2:<14}  r = {r_val:>7.4f}   p = {p_val:.4f}  (n={n_mpsi})")
        else:
            r_val, p_val = stats.pearsonr(v1, v2)
            out(f"  {n1} ~ {n2:<14}  r = {r_val:>7.4f}   p = {p_val:.4f}")

    # Models -- MPSI models restricted to rows with non-NaN MPSI
    out(f"\n{'-'*65}")
    models = [
        ("MPSI only (baseline)",        ["MPSI"],           mpsi[mpsi_mask].reshape(-1,1),   rc[mpsi_mask]),
        ("mean_H only",                 ["mean_H"],         mH.reshape(-1,1),                rc),
        ("std_H only",                  ["std_H"],          sH.reshape(-1,1),                rc),
        ("MPSI + mean_H",               ["MPSI","mean_H"],  np.c_[mpsi[mpsi_mask],mH[mpsi_mask]], rc[mpsi_mask]),
        ("MPSI + std_H",                ["MPSI","std_H"],   np.c_[mpsi[mpsi_mask],sH[mpsi_mask]], rc[mpsi_mask]),
        ("MPSI + mean_H + std_H",       ["MPSI","mean_H","std_H"],
                                        np.c_[mpsi[mpsi_mask],mH[mpsi_mask],sH[mpsi_mask]], rc[mpsi_mask]),
    ]

    r2_baseline = None
    for label, pred_names, Xp, rc_mod in models:
        beta, r2, resid, se, t_stat, p_val = ols(Xp, rc_mod)
        n_mod = len(rc_mod)
        dof = n_mod - len(pred_names) - 1
        if r2_baseline is None:
            r2_baseline = r2
        delta = f"  dR2={r2-r2_baseline:+.4f}" if label != "MPSI only (baseline)" else ""
        fams_mod = families if n_mod == n else [f for f, m in zip(families, mpsi_mask) if m]
        print_model(label, pred_names, beta, r2, resid, se, t_stat, p_val,
                    fams_mod, rc_mod, dof)
        out(f"  R2={r2:.4f}{delta}")

    # Residual table for best model
    # Best = highest R2 among models with p(new predictor) < 0.10
    out(f"\n{'-'*65}")
    out(f"Per-family values:")
    out(f"  {'Family':<25} {'r_cons':>8} {'MPSI':>7} {'mean_H':>8} "
        f"{'std_H':>7} {'frac_cons':>10}")
    out(f"  {'-'*68}")
    for i, fam in enumerate(families):
        mpsi_str = f"{mpsi[i]:>7.4f}" if not np.isnan(mpsi[i]) else "    N/A"
        out(f"  {fam:<25} {rc[i]:>8.4f} {mpsi_str} {mH[i]:>8.4f} "
            f"{sH[i]:>7.4f} {fc[i]:>10.4f}")

    # Leave-one-out for mean_H only model (works for all n families)
    out(f"\n{'-'*65}")
    out(f"Leave-one-out (mean_H only model, n={n}):")
    Xfull = mH.reshape(-1,1)
    _, r2_full, _, _, _, _ = ols(Xfull, rc)
    out(f"  Full (n={n}): R2={r2_full:.4f}")
    for i, fam in enumerate(families):
        mask = np.ones(n, dtype=bool); mask[i] = False
        _, r2_loo, _, _, _, _ = ols(Xfull[mask], rc[mask])
        delta = r2_loo - r2_full
        flag = " <-- influential" if abs(delta) > 0.03 else ""
        out(f"  Drop {fam:<25}  R2={r2_loo:.4f}  d={delta:+.4f}{flag}")

    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nRegression output written to {txt_path}")

    # -- Write regression CSV -------------------------------------------
    reg_csv = os.path.join(args.out_dir, "site_entropy_regression_data.csv")
    with open(reg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["family","r_cons","mpsi","mean_H","std_H",
                        "frac_conserved","n_seq","L_filt"])
        writer.writeheader()
        writer.writerows(reg_rows)
    print(f"Regression data written to {reg_csv}")


if __name__ == "__main__":
    main()
