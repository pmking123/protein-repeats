#!/usr/bin/env python3
"""
rcons_structural_regression.py
================================
Multiple regression of r_cons on MPSI and contact density (rho) for
the finite-L family panel.

    r_cons_i = β0 + β1 * MPSI_i + β2 * rho_i + ε_i

Inputs
------
contact_density_results.csv  -- from contact_density_analysis.py
  (fill in r_cons and mpsi columns before running, or pass --rcons-csv)

MPSI and r_cons values (from prior analysis, rcons_identity_correlation.py):
  Fill KNOWN_DATA below with your values, or supply a CSV with columns
  family, r_cons, mpsi.

Outputs
-------
  - Regression coefficients and p-values (OLS, n=7)
  - Partial R² for MPSI vs rho
  - Residual plot identifying outliers (Immunoglobulin flagged)
  - rcons_structural_regression_results.txt
"""

import argparse
import csv
import os
import sys

import numpy as np

# ── Fill in from rcons_identity_correlation.py output ─────────────────────
# family: (r_cons, MPSI)
KNOWN_DATA = {
    "SH2":            (0.5386, 0.3220),
    "SH3":            (0.5558, 0.2945),
    "Kinase":         (0.5596, 0.2805),
    "Immunoglobulin": (0.6997, 0.1543),
    "PDZ":            (0.6073, 0.2175),
    "Ubiquitin":      (0.6068, 0.2282),
    "RRM":            (0.6286, 0.2344),
}


def ols_regression(X: np.ndarray, y: np.ndarray):
    """
    OLS with intercept. Returns (coeffs, residuals, R2, leverage).
    X shape: (n, k) — predictors only (intercept added internally)
    """
    n, k = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])   # add intercept column
    beta, res, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
    y_hat = X_aug @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Hat matrix diagonal (leverage)
    H = X_aug @ np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T
    leverage = np.diag(H)

    return beta, y_hat, r2, leverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-csv", default="contact_density_results.csv")
    parser.add_argument("--rcons-csv",   default=None,
                        help="CSV with columns: family, r_cons, mpsi")
    parser.add_argument("--out-dir",     default=".")
    args = parser.parse_args()

    # Load contact density
    contact = {}
    with open(args.contact_csv) as f:
        for row in csv.DictReader(f):
            if row["rho"]:
                contact[row["family"]] = float(row["rho"])

    # Load r_cons / MPSI
    rc_mpsi = dict(KNOWN_DATA)
    if args.rcons_csv:
        with open(args.rcons_csv) as f:
            for row in csv.DictReader(f):
                rc_mpsi[row["family"]] = (float(row["r_cons"]), float(row["mpsi"]))

    if not rc_mpsi:
        print("ERROR: No r_cons/MPSI data. Fill KNOWN_DATA dict or pass --rcons-csv.")
        sys.exit(1)

    # Assemble design matrix
    families, y_vals, mpsi_vals, rho_vals = [], [], [], []
    for fam, (rc, mpsi) in rc_mpsi.items():
        if fam in contact:
            families.append(fam)
            y_vals.append(rc)
            mpsi_vals.append(mpsi)
            rho_vals.append(contact[fam])

    if len(families) < 4:
        print(f"Only {len(families)} families with complete data — need >=4 for regression.")
        sys.exit(1)

    y    = np.array(y_vals)
    mpsi = np.array(mpsi_vals)
    rho  = np.array(rho_vals)

    # Full model: r_cons ~ MPSI + rho
    X_full = np.column_stack([mpsi, rho])
    beta_full, yhat_full, r2_full, lev_full = ols_regression(X_full, y)

    # Partial models
    _, _, r2_mpsi_only, _ = ols_regression(mpsi.reshape(-1,1), y)
    _, _, r2_rho_only,  _ = ols_regression(rho.reshape(-1,1),  y)

    print(f"\nMultiple regression: r_cons ~ MPSI + rho  (n={len(families)})")
    print(f"{'Family':<18} {'r_cons':>8} {'MPSI':>8} {'rho':>8} {'fitted':>8} {'resid':>8}")
    print("-" * 68)
    for i, fam in enumerate(families):
        resid = y[i] - yhat_full[i]
        tag = " ← outlier?" if abs(resid) > 2 * np.std(y - yhat_full) else ""
        print(f"{fam:<18} {y[i]:>8.4f} {mpsi[i]:>8.3f} {rho[i]:>8.3f} "
              f"{yhat_full[i]:>8.4f} {resid:>+8.4f}{tag}")

    print(f"\nCoefficients:")
    print(f"  intercept: {beta_full[0]:+.4f}")
    print(f"  MPSI:      {beta_full[1]:+.4f}")
    print(f"  rho:       {beta_full[2]:+.4f}")
    print(f"\nR² (full):      {r2_full:.4f}")
    print(f"R² (MPSI only): {r2_mpsi_only:.4f}")
    print(f"R² (rho only):  {r2_rho_only:.4f}")
    print(f"ΔR² (rho|MPSI): {r2_full - r2_mpsi_only:+.4f}")

    print(f"\nNote: n=7 is marginal for 2-predictor OLS.")
    print(f"Interpret ΔR² and residuals; p-values are indicative only.")


if __name__ == "__main__":
    main()
