"""
partition_L_gridsearch.py
=========================
Finds the partition length L* that minimises the total deviation between
the partition model entropy ratio predictions and the biological values
from Marchi et al. (2019), for each of ANK, LRR, and TPR.

Scans L = 1 ... L_max (default 100) and computes exact finite-L ratios:
  r_conservation = log p_RR(L) / log p(L)   [target: S_1/S_rand]
  r_interaction  = log p_RR(L) / log p_d(L) [target: S_full/S_1]

Loss function: sum of squared percentage deviations from biological values.
Also reports the L* that minimises each ratio separately.

Outputs:
  - Console summary
  - partition_L_gridsearch.csv  — full grid of deviations for all families
  - partition_L_gridsearch.png  — deviation curves with L* marked

Usage:
  python partition_L_gridsearch.py [--L_max 100]

Dependencies: numpy, pandas, matplotlib
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache

# ---------------------------------------------------------------------------
# Biological data: Marchi et al. (2019) Table 1
# ---------------------------------------------------------------------------

MARCHI = {
    'ANK': dict(two_L=66,  S_rand=290, S_1=181.0, S_ir=176.7, S_full=167.2),
    'LRR': dict(two_L=48,  S_rand=211, S_1=130.0, S_ir=123.1, S_full=113.2),
    'TPR': dict(two_L=68,  S_rand=299, S_1=169.0, S_ir=157.6, S_full=141.4),
}

for fam, d in MARCHI.items():
    d['L_res']          = d['two_L'] // 2
    d['r_conservation'] = d['S_1']   / d['S_rand']
    d['r_interaction']  = d['S_full']/ d['S_1']

# Asymptotic (L -> inf) predictions
ALPHA_0  = np.pi * np.sqrt(2 / 3)
ALPHA_RR = 2 * np.pi / np.sqrt(15)
ALPHA_D  = np.pi / np.sqrt(3)
PRED_CONS = ALPHA_RR / ALPHA_0   # sqrt(2/5) = 0.6325
PRED_INT  = ALPHA_RR / ALPHA_D   # 2/sqrt(5) = 0.8944

# ---------------------------------------------------------------------------
# Exact partition counts via dynamic programming
# ---------------------------------------------------------------------------

# We build tables up to L_max once, then index.

def build_partition_tables(L_max):
    """
    Returns three arrays of length L_max+1:
      p_unres[L]  = number of unrestricted partitions of L
      p_rr[L]     = number of Rogers-Ramanujan partitions of L
      p_dist[L]   = number of distinct-part partitions of L
    """
    # --- Unrestricted partitions (Euler's recurrence via pentagonal numbers) ---
    p_unres = [0] * (L_max + 1)
    p_unres[0] = 1
    for n in range(1, L_max + 1):
        k = 1
        while True:
            p1 = k * (3 * k - 1) // 2
            p2 = k * (3 * k + 1) // 2
            if p1 > n:
                break
            sign = 1 if k % 2 == 1 else -1
            p_unres[n] += sign * p_unres[n - p1]
            if p2 <= n:
                p_unres[n] += sign * p_unres[n - p2]
            k += 1

    # --- RR partitions: parts ≡ ±1 (mod 5), each part used any number of times ---
    p_rr = [0] * (L_max + 1)
    p_rr[0] = 1
    for part in range(1, L_max + 1):
        if part % 5 in (1, 4):
            for j in range(part, L_max + 1):
                p_rr[j] += p_rr[j - part]

    # --- Distinct-part partitions: each part used at most once ---
    p_dist = [0] * (L_max + 1)
    p_dist[0] = 1
    for part in range(1, L_max + 1):
        for j in range(L_max, part - 1, -1):
            p_dist[j] += p_dist[j - part]

    return np.array(p_unres, dtype=float), np.array(p_rr, dtype=float), np.array(p_dist, dtype=float)


def entropy_ratios(L, p_unres, p_rr, p_dist):
    """Finite-L entropy ratios at a given L."""
    lp  = np.log(p_unres[L]) if p_unres[L] > 0 else np.nan
    lrr = np.log(p_rr[L])   if p_rr[L]   > 0 else np.nan
    ld  = np.log(p_dist[L]) if p_dist[L] > 0 else np.nan
    r_cons = lrr / lp  if (not np.isnan(lp)  and lp  > 0) else np.nan
    r_int  = lrr / ld  if (not np.isnan(ld)  and ld  > 0) else np.nan
    return r_cons, r_int


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_gridsearch(L_max=100):
    print(f"Building partition tables up to L = {L_max} ...")
    p_unres, p_rr, p_dist = build_partition_tables(L_max)
    print("Done.\n")

    print(f"Asymptotic predictions (L -> inf):")
    print(f"  r_conservation = sqrt(2/5) = {PRED_CONS:.4f}")
    print(f"  r_interaction  = 2/sqrt(5) = {PRED_INT:.4f}")
    print()

    all_rows = []

    colours = {'ANK': '#2166ac', 'LRR': '#d73027', 'TPR': '#4dac26'}

    # Collect all data first, then plot — so all panels use the same L_max
    family_data = {}

    for fam in ['ANK', 'LRR', 'TPR']:
        bio       = MARCHI[fam]
        bio_cons  = bio['r_conservation']
        bio_int   = bio['r_interaction']
        L_res     = bio['L_res']

        Ls, r_cons_arr, r_int_arr = [], [], []
        loss_arr, dev_cons_arr, dev_int_arr = [], [], []

        for L in range(2, L_max + 1):
            rc, ri = entropy_ratios(L, p_unres, p_rr, p_dist)
            if np.isnan(rc) or np.isnan(ri):
                continue
            dc = 100 * (rc - bio_cons) / bio_cons
            di = 100 * (ri - bio_int)  / bio_int
            loss = dc**2 + di**2

            Ls.append(L)
            r_cons_arr.append(rc)
            r_int_arr.append(ri)
            dev_cons_arr.append(dc)
            dev_int_arr.append(di)
            loss_arr.append(loss)

            all_rows.append({
                'family':      fam,
                'L':           L,
                'r_cons':      round(rc, 4),
                'r_int':       round(ri, 4),
                'bio_cons':    round(bio_cons, 4),
                'bio_int':     round(bio_int, 4),
                'dev_cons_%':  round(dc, 2),
                'dev_int_%':   round(di, 2),
                'loss':        round(loss, 4),
            })

        Ls           = np.array(Ls)
        loss_arr     = np.array(loss_arr)
        dev_cons_arr = np.array(dev_cons_arr)
        dev_int_arr  = np.array(dev_int_arr)

        L_star_joint = Ls[np.argmin(loss_arr)]
        L_star_cons  = Ls[np.argmin(np.abs(dev_cons_arr))]
        L_star_int   = Ls[np.argmin(np.abs(dev_int_arr))]

        family_data[fam] = dict(
            Ls=Ls, loss_arr=loss_arr,
            dev_cons_arr=dev_cons_arr, dev_int_arr=dev_int_arr,
            L_star_joint=L_star_joint, L_star_cons=L_star_cons,
            L_star_int=L_star_int, L_res=L_res,
            bio_cons=bio_cons, bio_int=bio_int,
        )

        # Console output
        rc_at_res,  ri_at_res  = entropy_ratios(L_res,        p_unres, p_rr, p_dist)
        rc_at_star, ri_at_star = entropy_ratios(L_star_joint, p_unres, p_rr, p_dist)

        print(f"{'='*65}")
        print(f"  {fam}  |  L_res = {L_res}  |  bio_cons = {bio_cons:.4f}  "
              f"bio_int = {bio_int:.4f}")
        print(f"  {'':12}  {'L':>5}  {'r_cons':>8}  {'dev%':>8}  "
              f"{'r_int':>8}  {'dev%':>8}  {'loss':>8}")
        print(f"  {'-'*60}")

        def fmt_row(label, L_val, rc, ri):
            dc = 100*(rc - bio_cons)/bio_cons
            di = 100*(ri - bio_int)/bio_int
            lo = dc**2 + di**2
            print(f"  {label:<12}  {L_val:>5}  {rc:>8.4f}  {dc:>+8.2f}%  "
                  f"{ri:>8.4f}  {di:>+8.2f}%  {lo:>8.2f}")

        fmt_row('L_res',      L_res,        rc_at_res,  ri_at_res)
        fmt_row('L* (joint)', L_star_joint, rc_at_star, ri_at_star)
        fmt_row('L* (cons)',  L_star_cons,
                *entropy_ratios(L_star_cons, p_unres, p_rr, p_dist))
        fmt_row('L* (int)',   L_star_int,
                *entropy_ratios(L_star_int,  p_unres, p_rr, p_dist))
        fmt_row('asymptotic', '∞',           PRED_CONS,  PRED_INT)
        print()

    # --- Plot all panels after all data is collected ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # x-axis runs to L_max + 5% margin so lines at the boundary are visible
    x_right = L_max * 1.05

    for row_idx, fam in enumerate(['ANK', 'LRR', 'TPR']):
        d  = family_data[fam]
        c  = colours[fam]
        ax_dev  = axes[row_idx, 0]
        ax_loss = axes[row_idx, 1]

        L_star = d['L_star_joint']
        at_boundary = (L_star == L_max)

        # Label: flag boundary-hitting optima explicitly
        if at_boundary:
            lstar_label = f"L*≥{L_star} (boundary)"
            lstar_color = 'darkorange'
            lstar_ls    = '-.'   # distinct style for boundary case
        else:
            lstar_label = f"L*={L_star}"
            lstar_color = 'orange'
            lstar_ls    = '--'

        # --- Deviation panel ---
        # Clip y to [-25, 25] to show structure near zero; small-L extremes
        # are not informative
        clip_lo, clip_hi = -25, 25
        ax_dev.plot(d['Ls'], np.clip(d['dev_cons_arr'], clip_lo, clip_hi),
                    '-',  color=c, lw=1.5, label='conservation dev%')
        ax_dev.plot(d['Ls'], np.clip(d['dev_int_arr'],  clip_lo, clip_hi),
                    '--', color=c, lw=1.5, label='interaction dev%')
        ax_dev.axhline(0, color='k', lw=0.8)
        ax_dev.axvline(d['L_res'], color='grey', lw=1, ls=':',
                       label=f"L_res={d['L_res']}")
        ax_dev.axvline(L_star, color=lstar_color, lw=1.5, ls=lstar_ls,
                       label=lstar_label)
        # Annotate asymptotic target values
        ax_dev.axhline(100*(PRED_CONS - d['bio_cons'])/d['bio_cons'],
                       color='k', lw=0.6, ls=':', alpha=0.4,
                       label=f"asymptotic cons dev")
        ax_dev.axhline(100*(PRED_INT - d['bio_int'])/d['bio_int'],
                       color='k', lw=0.6, ls='--', alpha=0.4,
                       label=f"asymptotic int dev")
        ax_dev.set_title(f'{fam}: deviation from biological ratios')
        ax_dev.set_xlabel('L')
        ax_dev.set_ylabel('Deviation (%)')
        ax_dev.set_ylim(clip_lo - 1, clip_hi + 1)
        ax_dev.legend(fontsize=7, loc='lower right')
        ax_dev.set_xlim(2, x_right)

        # --- Loss panel ---
        # Clip loss y-axis to show structure above small-L spike
        loss_max_display = min(np.max(d['loss_arr']), 500)
        ax_loss.plot(d['Ls'], np.clip(d['loss_arr'], 0, loss_max_display),
                     '-', color=c, lw=1.5)
        ax_loss.axvline(d['L_res'], color='grey', lw=1, ls=':',
                        label=f"L_res={d['L_res']}")
        ax_loss.axvline(L_star, color=lstar_color, lw=1.5, ls=lstar_ls,
                        label=lstar_label)
        ax_loss.set_title(f'{fam}: joint loss (sum of squared dev%)')
        ax_loss.set_xlabel('L')
        ax_loss.set_ylabel('Loss (clipped at 500)')
        ax_loss.set_ylim(0, loss_max_display * 1.05)
        ax_loss.legend(fontsize=7)
        ax_loss.set_xlim(2, x_right)

    fig.suptitle(
        f'Partition model L grid search vs biological entropy ratios\n'
        f'(Marchi et al. 2019; L_max={L_max}; '
        f'orange dashed = finite L*; dash-dot = boundary L*)',
        fontsize=11)
    fig.tight_layout()
    fig.savefig('partition_L_gridsearch.png', dpi=150, bbox_inches='tight')
    print("Saved: partition_L_gridsearch.png")

    df = pd.DataFrame(all_rows)
    df.to_csv('partition_L_gridsearch.csv', index=False)
    print("Saved: partition_L_gridsearch.csv")

    # Final summary table
    print(f"\n{'='*50}")
    print("SUMMARY: optimal L per family")
    print(f"{'='*50}")
    print(f"{'Family':<8} {'L_res':>6} {'L*(joint)':>10} "
          f"{'L*(cons)':>9} {'L*(int)':>8}")
    print(f"{'-'*50}")
    for fam in ['ANK', 'LRR', 'TPR']:
        sub = df[df['family'] == fam]
        L_res     = MARCHI[fam]['L_res']
        L_j       = int(sub.loc[sub['loss'].idxmin(), 'L'])
        L_c       = int(sub.loc[sub['dev_cons_%'].abs().idxmin(), 'L'])
        L_i       = int(sub.loc[sub['dev_int_%'].abs().idxmin(), 'L'])
        print(f"{fam:<8} {L_res:>6} {L_j:>10} {L_c:>9} {L_i:>8}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--L_max', type=int, default=100,
                        help='Maximum L to scan (default: 100)')
    args = parser.parse_args()
    run_gridsearch(L_max=args.L_max)
