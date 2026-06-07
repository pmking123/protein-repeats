#!/usr/bin/env python3
"""
plot_entropy_regression.py
===========================
Produces a two-panel publication-quality figure for the
mean per-site entropy (mean_H) vs r_cons analysis.

Panel A: main result
    mean_H vs r_cons, points coloured by log10(n_seq),
    OLS regression line with 95% CI band, family labels.

Panel B: supplementary check
    Residuals from the mean_H model vs log10(n_seq),
    to show the n_seq sampling bias does not drive the result.

INPUTS
------
  site_entropy_results.csv         -- from site_entropy_analysis.py
  rcons_identity_correlation.csv   -- from rcons_identity_correlation.py

OUTPUT
------
  entropy_regression_figure.pdf    -- vector, for journal submission
  entropy_regression_figure.png    -- 300 dpi raster, for quick viewing

USAGE
-----
  python plot_entropy_regression.py \
      --entropy-csv site_entropy_results.csv \
      --rcons-csv   rcons_identity_correlation.csv \
      --out-dir     ./figures
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import t as t_dist

# ── Style ─────────────────────────────────────────────────────────────────
# Journal-appropriate: no serif, clean axes, no chartjunk
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "axes.linewidth":    0.8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,   # embed fonts as Type 42 (TrueType) for journal
    "ps.fonttype":       42,
})

# ── Colours ───────────────────────────────────────────────────────────────
REGIME_COLOURS = {
    "FINITE-L":   "#2166ac",   # blue
    "ASYMPTOTIC": "#d6604d",   # red-orange
}

# ── Helpers ───────────────────────────────────────────────────────────────

def ols_with_ci(x, y, x_pred=None, ci=0.95):
    """
    OLS fit of y ~ x (with intercept).
    Returns fitted line and pointwise CI band at x_pred.
    """
    n = len(x)
    xm = x.mean()
    Sxx = np.sum((x - xm)**2)
    b1 = np.sum((x - xm) * y) / Sxx
    b0 = y.mean() - b1 * xm
    yhat = b0 + b1 * x
    resid = y - yhat
    s2 = np.sum(resid**2) / (n - 2)

    if x_pred is None:
        x_pred = np.linspace(x.min(), x.max(), 200)

    se_pred = np.sqrt(s2 * (1/n + (x_pred - xm)**2 / Sxx))
    t_crit  = t_dist.ppf((1 + ci) / 2, df=n - 2)

    y_fit  = b0 + b1 * x_pred
    y_lo   = y_fit - t_crit * se_pred
    y_hi   = y_fit + t_crit * se_pred

    r, p = stats.pearsonr(x, y)
    r2 = r**2

    return {
        "b0": b0, "b1": b1, "r": r, "r2": r2, "p": p,
        "x_pred": x_pred, "y_fit": y_fit, "y_lo": y_lo, "y_hi": y_hi,
        "resid": resid,
    }


def load_data(entropy_csv, rcons_csv):
    entropy = {r["family"]: r
               for r in csv.DictReader(open(entropy_csv))
               if r["status"] == "ok"}
    rcons   = {r["name"]: r
               for r in csv.DictReader(open(rcons_csv))
               if r["source"] == "computed"}

    rows = []
    for fam, er in entropy.items():
        if fam not in rcons:
            continue
        rows.append({
            "family":  fam,
            "r_cons":  float(rcons[fam]["r_cons"]),
            "mpsi":    float(rcons[fam]["mpsi"]),
            "mean_H":  float(er["mean_H"]),
            "std_H":   float(er["std_H"]),
            "n_seq":   int(er["n_seq"]),
            "L_filt":  int(er["L_filt"]),
            "regime":  rcons[fam]["regime"],
        })
    return rows


def shorten_label(name):
    """Strip PFAM accession for cleaner annotation."""
    return name.split(" (")[0]


# ── Main figure ───────────────────────────────────────────────────────────

def make_figure(rows, out_dir):
    families = [r["family"]  for r in rows]
    rc       = np.array([r["r_cons"]  for r in rows])
    mH       = np.array([r["mean_H"]  for r in rows])
    n_seq    = np.array([r["n_seq"]   for r in rows])
    regimes  = [r["regime"]  for r in rows]
    log_n    = np.log10(n_seq)
    n        = len(rows)

    fit = ols_with_ci(mH, rc)

    # ── Colour map for log10(n_seq) ───────────────────────────────────────
    norm   = mcolors.Normalize(vmin=log_n.min(), vmax=log_n.max())
    cmap   = matplotlib.colormaps["viridis"]

    # ── Figure layout ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.0, 3.4),       # fits a single-column-wide journal figure
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.38},
    )

    # ══════════════════════════════════════════════════════════════════════
    # Panel A: mean_H vs r_cons
    # ══════════════════════════════════════════════════════════════════════
    ax = axes[0]

    # 95% CI band
    ax.fill_between(fit["x_pred"], fit["y_lo"], fit["y_hi"],
                    color="#999999", alpha=0.18, linewidth=0, zorder=1)

    # Regression line
    ax.plot(fit["x_pred"], fit["y_fit"],
            color="#444444", linewidth=1.2, zorder=2, linestyle="--")

    # Scatter: colour by log10(n_seq), shape by regime
    markers = {"FINITE-L": "o", "ASYMPTOTIC": "s"}
    for i, row in enumerate(rows):
        colour = cmap(norm(log_n[i]))
        marker = markers.get(row["regime"], "o")
        ax.scatter(mH[i], rc[i],
                   c=[colour], marker=marker,
                   s=52, linewidths=0.6,
                   edgecolors="white", zorder=3)

    # Labels — offset to avoid overlap
    # Pre-computed offsets (dx, dy) per family for clean layout
    label_offsets = {
        "TEM-1":       ( 0.02, -0.013),
        "PDZ":         ( 0.02,  0.006),
        "RRM":         ( 0.02,  0.006),
        "Prot.kinase": (-0.03, -0.013),
        "SH2":         (-0.03, -0.013),
        "SH3":         ( 0.02,  0.005),
        "Globin":      ( 0.02,  0.005),
        "Ig":          ( 0.02, -0.010),
        "Ferredoxin":  ( 0.02,  0.005),
        "Ubiquitin":   (-0.05, -0.013),
        "HATPase":     ( 0.02,  0.005),
        "WD40":        ( 0.02, -0.010),
        "HEAT":        (-0.04, -0.013),
    }
    for i, row in enumerate(rows):
        short = shorten_label(row["family"])
        dx, dy = label_offsets.get(short, (0.02, 0.005))
        ax.annotate(
            short,
            xy=(mH[i], rc[i]),
            xytext=(mH[i] + dx, rc[i] + dy),
            fontsize=6.5,
            color="#222222",
            ha="left" if dx >= 0 else "right",
            va="center",
        )

    # Threshold line at sqrt(2/5)
    threshold = np.sqrt(2/5)
    ax.axhline(threshold, color="#888888", linewidth=0.7,
               linestyle=":", zorder=0)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 1 else 2.25,
            threshold + 0.004,
            r"$r_\mathrm{cons} = \sqrt{2/5}$",
            fontsize=6.5, color="#888888", va="bottom")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02, aspect=18)
    cbar.set_label(r"$\log_{10}(n_\mathrm{seq})$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Regime legend
    legend_elements = [
        plt.scatter([], [], marker="o", c="#555555", s=40,
                    label="Finite-$L$"),
        plt.scatter([], [], marker="s", c="#555555", s=40,
                    label="Asymptotic"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=7, frameon=False,
              handletextpad=0.3, borderpad=0.2)

    # Stats annotation
    r2_str  = f"$R^2 = {fit['r2']:.3f}$"
    p_str   = f"$p = {fit['p']:.2e}$"
    n_str   = f"$n = {n}$"
    ax.text(0.97, 0.08,
            f"{r2_str}\n{p_str}\n{n_str}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#cccccc", alpha=0.85))

    ax.set_xlabel(r"Mean per-site entropy $\langle H_i \rangle$ (bits)")
    ax.set_ylabel(r"$r_\mathrm{cons}$")
    ax.set_title("A", loc="left", fontweight="bold", fontsize=10)

    # Tidy spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ══════════════════════════════════════════════════════════════════════
    # Panel B: residuals vs log10(n_seq)
    # ══════════════════════════════════════════════════════════════════════
    ax2 = axes[1]

    resid = fit["resid"]
    r_res, p_res = stats.pearsonr(log_n, resid)

    # Horizontal zero line
    ax2.axhline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=0)

    # +/- 2*RMSE band
    rmse = np.sqrt(np.mean(resid**2))
    ax2.axhline( 2*rmse, color="#cccccc", linewidth=0.5, linestyle=":", zorder=0)
    ax2.axhline(-2*rmse, color="#cccccc", linewidth=0.5, linestyle=":", zorder=0)
    ax2.text(log_n.max() + 0.05, 2*rmse, r"$\pm 2\,\mathrm{RMSE}$",
             fontsize=6, color="#aaaaaa", va="center")

    for i, row in enumerate(rows):
        colour = cmap(norm(log_n[i]))
        marker = markers.get(row["regime"], "o")
        ax2.scatter(log_n[i], resid[i],
                    c=[colour], marker=marker,
                    s=52, linewidths=0.6,
                    edgecolors="white", zorder=3)
        short = shorten_label(row["family"])
        # Label only large residuals to keep it clean
        if abs(resid[i]) > 1.5 * rmse:
            # HATPase is top-right -- label to the left to avoid box overlap
            is_large_n = log_n[i] > 2.5
            xoff = -0.10 if is_large_n else 0.08
            ha   = "right" if is_large_n else "left"
            ax2.annotate(
                short,
                xy=(log_n[i], resid[i]),
                xytext=(log_n[i] + xoff, resid[i]),
                fontsize=6.5, color="#222222", va="center", ha=ha,
            )

    # Residual trend stats + HATPase sensitivity
    from scipy.stats import pearsonr as _pr
    mask_no_h = np.array([shorten_label(r["family"]) != "HATPase"
                          for r in rows])
    r_no_h, p_no_h = _pr(log_n[mask_no_h], resid[mask_no_h])
    ax2.text(0.97, 0.08,
             f"$r = {r_res:.3f}$, $p = {p_res:.2f}$\n"
             f"excl. HATPase: $r = {r_no_h:.3f}$, $p = {p_no_h:.2f}$",
             transform=ax2.transAxes,
             ha="right", va="bottom",
             fontsize=6.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="#cccccc", alpha=0.85))

    ax2.set_xlabel(r"$\log_{10}(n_\mathrm{seq})$")
    ax2.set_ylabel(r"Residual ($r_\mathrm{cons}$ -- fitted)")
    ax2.set_title("B", loc="left", fontweight="bold", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "entropy_regression_figure.pdf")
    png_path = os.path.join(out_dir, "entropy_regression_figure.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return pdf_path, png_path


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy-csv", default="site_entropy_results.csv")
    parser.add_argument("--rcons-csv",   default="rcons_identity_correlation.csv")
    parser.add_argument("--out-dir",     default="./figures")
    args = parser.parse_args()

    rows = load_data(args.entropy_csv, args.rcons_csv)
    print(f"Loaded {len(rows)} families")
    make_figure(rows, args.out_dir)


if __name__ == "__main__":
    main()
