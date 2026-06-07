"""
pfam_regime_analysis.py
=======================
Reads PFAM seed alignments in Stockholm format, computes the conservation
ratio r_cons = S_1/S_rand for each family, and classifies each family
into the partition model regime (FINITE-L or ASYMPTOTIC).

For FINITE-L families, finds the optimal partition length L* by matching
the biological r_cons to the finite-L partition model prediction
log p_RR(L) / log p(L).

The partition model asymptotic threshold is sqrt(2/5) = 0.6325, derived
from the Rogers-Ramanujan entropy coefficient ratio alpha_RR/alpha_0.
Families with r_cons < threshold have a finite-L optimum; families with
r_cons >= threshold are in the asymptotic regime.

USAGE
-----
1. Edit the ALIGNMENTS list below to add your families.
2. Run:  python pfam_regime_analysis.py [--L_max 500] [--occ 0.10]
         [--alignment_dir /path/to/seeds]
         [--csv results.csv] [--plot1 curve.png] [--plot2 dist.png]

For families where you have pre-computed entropy ratios (e.g. from
published Table 1 values), set 'r_cons_override' instead of 'file'.

INPUTS
------
Stockholm-format seed alignments (.seed or .sto files) as downloaded
from InterPro/PFAM. Each alignment must have all sequences the same
width (standard PFAM format).

OUTPUTS
-------
  pfam_regime_results.csv      -- full results table
  pfam_regime_plot.png         -- r_cons vs L* on model curve
  pfam_rcons_distribution.png  -- r_cons by structural class, coloured by regime

DEPENDENCIES
------------
  numpy, pandas, matplotlib
  pip install numpy pandas matplotlib

NOTE ON SMALL SAMPLES
---------------------
Entropy estimates from seed alignments with fewer than ~30 sequences
are unreliable. For such families, download the full alignment
(not the seed) from PFAM for a more accurate r_cons estimate.
The script flags families with n_seq < 20 automatically.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path

# ===========================================================================
# INPUT: edit this list to add families
# ===========================================================================
#
# Each entry is a dict with:
#   name             : display name
#   type             : one of 'repeat', 'single-domain', 'PFAM-wide' (for colouring)
#   file             : Stockholm alignment filename (resolved vs ALIGNMENT_DIR), or None
#   r_cons_override  : pre-computed r_cons = S_1/S_rand, or None
#   L_res            : known repeat-unit or domain length (optional, for reference)
#
# If 'r_cons_override' is set, 'file' is ignored.

ALIGNMENTS = [
    # --- Repeat proteins: r_cons from Marchi et al. (2019) Table 1 ---
    {
        'name':            'ANK',
        'type':            'repeat',
        'file':            None,
        'r_cons_override': 181.0 / 290,   # 0.6241
        'L_res':           33,
    },
    {
        'name':            'LRR',
        'type':            'repeat',
        'file':            None,
        'r_cons_override': 130.0 / 211,   # 0.6161
        'L_res':           24,
    },
    {
        'name':            'TPR',
        'type':            'repeat',
        'file':            None,
        'r_cons_override': 169.0 / 299,   # 0.5652
        'L_res':           34,
    },
    # --- Single-domain proteins: computed from PFAM seed alignments ---
    {
        'name':            'TEM-1 (PF13354)',
        'type':            'single-domain',
        'file':            'PF13354_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'PDZ (PF00595)',
        'type':            'single-domain',
        'file':            'PF00595_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'RRM (PF00076)',
        'type':            'single-domain',
        'file':            'PF00076_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'Glucosidase (PF00232)',
        'type':            'single-domain',
        'file':            'PF00232_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'Prot.kinase (PF00069)',
        'type':            'single-domain',
        'file':            'PF00069_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'SH2 (PF00017)',
        'type':            'single-domain',
        'file':            'PF00017_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'SH3 (PF00018)',
        'type':            'single-domain',
        'file':            'PF00018_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'Globin (PF00042)',
        'type':            'single-domain',
        'file':            'PF00042_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'Immunoglobulin (PF00047)',
        'type':            'single-domain',
        'file':            'PF00047_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'Ferredoxin (PF00111)',
        'type':            'single-domain',
        'file':            'PF00111_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'Ubiquitin (PF00240)',
        'type':            'single-domain',
        'file':            'PF00240_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'HATPase (PF02518)',
        'type':            'single-domain',
        'file':            'PF02518_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    # --- Repeat proteins from PFAM seed alignments ---
    {
        'name':            'WD40 (PF00400)',
        'type':            'repeat',
        'file':            'PF00400_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    {
        'name':            'HEAT (PF02985)',
        'type':            'repeat',
        'file':            'PF02985_alignment.seed',
        'r_cons_override': None,
        'L_res':           None,
    },
    # --- PFAM-wide average: King (2026) ---
    {
        'name':            'PFAM HMM mean',
        'type':            'PFAM-wide',
        'file':            None,
        'r_cons_override': 0.613,
        'L_res':           None,
    },
]

# Directory containing alignment files (overridden by --alignment_dir)
ALIGNMENT_DIR = Path('.')

# ===========================================================================
# Stockholm parser
# ===========================================================================

AA = set('ACDEFGHIKLMNPQRSTVWY')

def parse_stockholm(path):
    """
    Parse a Stockholm-format alignment.
    Returns {seq_name: aligned_sequence_string}.
    Multi-block alignments are joined correctly.
    """
    seqs = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#') or line.startswith('//') or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2:
                seqs.setdefault(parts[0], []).append(parts[1])
    return {k: ''.join(v) for k, v in seqs.items()}


def compute_r_cons(path, occupancy_threshold=0.10):
    """
    Compute r_cons = S_1 / S_rand from a Stockholm alignment.

    S_1    = sum over match columns of per-site Shannon entropy (bits)
    S_rand = L_eff * log2(21)
    L_eff  = columns where >= occupancy_threshold fraction have a standard AA

    Returns (r_cons, L_eff, n_sequences).
    """
    seqs  = parse_stockholm(path)
    aln   = list(seqs.values())
    n_seq = len(aln)
    L_aln = len(aln[0])

    match_cols = [
        j for j in range(L_aln)
        if sum(1 for s in aln if s[j].upper() in AA) / n_seq >= occupancy_threshold
    ]

    S1 = 0.0
    for j in match_cols:
        col    = [s[j].upper() for s in aln if s[j].upper() in AA]
        counts = Counter(col)
        total  = sum(counts.values())
        for cnt in counts.values():
            p   = cnt / total
            S1 -= p * np.log2(p)

    L_eff = len(match_cols)
    return S1 / (L_eff * np.log2(21)), L_eff, n_seq


# ===========================================================================
# Partition tables (dynamic programming)
# ===========================================================================

def build_partition_tables(L_max):
    """
    Returns arrays p_unres, p_rr, p_dist of length L_max+1.
    p_unres: unrestricted partitions (Euler pentagonal recurrence)
    p_rr:    Rogers-Ramanujan partitions (parts == +-1 mod 5)
    p_dist:  distinct-part partitions
    """
    p_u = [0] * (L_max + 1); p_u[0] = 1
    for n in range(1, L_max + 1):
        k = 1
        while True:
            p1 = k * (3*k - 1) // 2
            p2 = k * (3*k + 1) // 2
            if p1 > n: break
            sign = 1 if k % 2 == 1 else -1
            p_u[n] += sign * p_u[n - p1]
            if p2 <= n: p_u[n] += sign * p_u[n - p2]
            k += 1

    p_r = [0] * (L_max + 1); p_r[0] = 1
    for part in range(1, L_max + 1):
        if part % 5 in (1, 4):
            for j in range(part, L_max + 1):
                p_r[j] += p_r[j - part]

    p_d = [0] * (L_max + 1); p_d[0] = 1
    for part in range(1, L_max + 1):
        for j in range(L_max, part - 1, -1):
            p_d[j] += p_d[j - part]

    return (np.array(p_u, dtype=float),
            np.array(p_r, dtype=float),
            np.array(p_d, dtype=float))


# Asymptotic entropy-ratio predictions
ALPHA_0   = np.pi * np.sqrt(2/3)
ALPHA_RR  = 2 * np.pi / np.sqrt(15)
ALPHA_D   = np.pi / np.sqrt(3)
ASYM_CONS = ALPHA_RR / ALPHA_0    # sqrt(2/5) = 0.6325
ASYM_INT  = ALPHA_RR / ALPHA_D    # 2/sqrt(5) = 0.8944


def rc_model(L, p_u, p_r):
    """Finite-L model conservation ratio log p_RR(L) / log p_unres(L)."""
    if p_u[L] <= 1 or p_r[L] <= 0:
        return np.nan
    return np.log(float(p_r[L])) / np.log(float(p_u[L]))


def find_Lstar(r_cons, p_u, p_r, L_max):
    """
    L* = argmin_L |rc_model(L) - r_cons|.
    Returns (L_star, rc_at_Lstar).
    """
    best_L, best_dev = 2, np.inf
    for L in range(2, L_max + 1):
        rcc = rc_model(L, p_u, p_r)
        if np.isnan(rcc): continue
        dev = abs(rcc - r_cons)
        if dev < best_dev:
            best_dev = dev; best_L = L
    return best_L, rc_model(best_L, p_u, p_r)


# ===========================================================================
# Helpers
# ===========================================================================

def _isnan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))

def _fmt(v, fmt_str=None, default='--'):
    if _isnan(v): return default
    return (fmt_str % v) if fmt_str else str(int(v))


# ===========================================================================
# Main
# ===========================================================================

def run(L_max=500, occupancy_threshold=0.10,
        csv_out='pfam_regime_results.csv',
        plot1_out='pfam_regime_plot.png',
        plot2_out='pfam_rcons_distribution.png'):

    print(f"Building partition tables to L={L_max} ...")
    p_u, p_r, p_d = build_partition_tables(L_max)
    print("Done.\n")

    print(f"Partition model asymptotic threshold: sqrt(2/5) = {ASYM_CONS:.4f}")
    print(f"  FINITE-L   : r_cons < {ASYM_CONS:.4f} AND L* < {L_max}")
    print(f"  ASYMPTOTIC : r_cons >= {ASYM_CONS:.4f} OR  L* = {L_max}\n")

    rows = []
    for entry in ALIGNMENTS:
        name  = entry['name']
        ftype = entry.get('type', 'unknown')
        L_res = entry.get('L_res')

        if entry.get('r_cons_override') is not None:
            r_cons, L_eff, n_seq = entry['r_cons_override'], np.nan, np.nan
        else:
            fpath = ALIGNMENT_DIR / entry['file']
            r_cons, L_eff, n_seq = compute_r_cons(fpath, occupancy_threshold)

        Lstar, rc_star = find_Lstar(r_cons, p_u, p_r, L_max)

        if r_cons >= ASYM_CONS or Lstar >= L_max:
            regime = 'ASYMPTOTIC'
            Lstar_out, rcs_out = np.nan, np.nan
            dev = 100 * (ASYM_CONS - r_cons) / r_cons
        else:
            regime = 'FINITE-L'
            Lstar_out, rcs_out = float(Lstar), rc_star
            dev = 100 * (rc_star - r_cons) / r_cons

        rows.append({
            'name':        name,
            'type':        ftype,
            'r_cons':      round(r_cons, 4),
            'n_seq':       n_seq,
            'L_eff':       L_eff,
            'L_res':       float(L_res) if L_res is not None else np.nan,
            'regime':      regime,
            'L_star':      Lstar_out,
            'rc_at_Lstar': rcs_out,
            'dev_pct':     round(dev, 2),
        })

    df = pd.DataFrame(rows)

    # --- Console output ---
    hdr = (f"{'Name':<28} {'Type':<14} {'r_cons':>7} {'n_seq':>5} "
           f"{'L_eff':>6} {'Regime':<12} {'L*':>5} {'rc@L*':>7} {'dev%':>8}")
    print(hdr); print('-' * len(hdr))

    for regime_group in ['FINITE-L', 'ASYMPTOTIC']:
        sub = df[df['regime'] == regime_group].sort_values('r_cons')
        if sub.empty: continue
        print(f"\n  --- {regime_group} ---")
        for _, r in sub.iterrows():
            flag    = ' !! small n' if not _isnan(r['n_seq']) and r['n_seq'] < 20 else ''
            rcs_str = _fmt(r['rc_at_Lstar'], '%.4f') if regime_group == 'FINITE-L' \
                      else f"{ASYM_CONS:.4f}"
            dev_str = (f"{r['dev_pct']:>+.2f}%" if regime_group == 'FINITE-L'
                       else f"{r['dev_pct']:>+.2f}%*")
            print(f"  {r['name']:<28} {r['type']:<14} {r['r_cons']:>7.4f} "
                  f"{_fmt(r['n_seq']):>5} {_fmt(r['L_eff']):>6} {r['regime']:<12} "
                  f"{_fmt(r['L_star']):>5} {rcs_str:>7} {dev_str:>8}{flag}")

    print(f"\n  * deviation from asymptote {ASYM_CONS:.4f}")

    # Summary counts
    print()
    for reg in ['FINITE-L', 'ASYMPTOTIC']:
        sub = df[df['regime'] == reg]
        types = sub['type'].value_counts().to_dict()
        print(f"{reg} ({len(sub)}): " +
              ", ".join(f"{t}: {n}" for t, n in types.items()))

    # --- Save CSV ---
    df.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    # --- Plot 1: r_cons vs L* on model curve ---
    Ls_arr = np.arange(2, L_max + 1)
    rc_arr = np.array([rc_model(L, p_u, p_r) for L in Ls_arr])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(Ls_arr, rc_arr, '-', color='#cccccc', lw=2, zorder=1,
            label='model r_cons(L)')
    ax.axhline(ASYM_CONS, color='#888888', ls='--', lw=1.2,
               label=f'asymptote sqrt(2/5) = {ASYM_CONS:.4f}')

    colour = {'repeat': '#2166ac', 'single-domain': '#d73027', 'PFAM-wide': '#4dac26'}
    marker = {'repeat': 'o', 'single-domain': 's', 'PFAM-wide': 'D'}
    x_asym = L_max + 40

    for _, r in df.iterrows():
        c = colour.get(r['type'], 'grey')
        m = marker.get(r['type'], 'o')
        x = r['L_star'] if r['regime'] == 'FINITE-L' else x_asym
        ax.scatter(x, r['r_cons'], color=c, marker=m, s=90, zorder=4,
                   edgecolors='k', linewidths=0.5)
        ax.annotate(r['name'], (x, r['r_cons']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
        if r['regime'] == 'FINITE-L' and not _isnan(r['L_star']):
            idx = int(r['L_star']) - 2
            if 0 <= idx < len(rc_arr):
                ax.plot([r['L_star'], r['L_star']], [r['r_cons'], rc_arr[idx]],
                        ':', color=c, alpha=0.4, lw=1)

    ax.axvline(x_asym - 8, color='#888888', ls=':', lw=0.8, alpha=0.5)
    ax.text(x_asym - 6, 0.32, 'ASYMPTOTIC', fontsize=8, color='#888888',
            rotation=90, va='bottom')

    handles = [mpatches.Patch(color=v, label=k) for k, v in colour.items()]
    handles += [
        plt.Line2D([0], [0], color='#cccccc', lw=2, label='model r_cons(L)'),
        plt.Line2D([0], [0], color='#888888', ls='--',
                   label=f'asymptote {ASYM_CONS:.4f}'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.set_xlabel('L*  (partition length matching biological r_cons)', fontsize=11)
    ax.set_ylabel('r_conservation  =  S1 / S_rand', fontsize=11)
    ax.set_title('Partition model regime classification\n'
                 'FINITE-L families shown at L*; ASYMPTOTIC families at right margin',
                 fontsize=10)
    ax.set_xlim(0, L_max + 80)
    ax.set_ylim(0.30, 0.78)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot1_out, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot1_out}")

    # --- Plot 2: r_cons distribution by structural class ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ypos = {'repeat': 0, 'single-domain': 1, 'PFAM-wide': 2}
    rng  = np.random.default_rng(42)

    for _, r in df.iterrows():
        c    = colour.get(r['type'], 'grey')
        m    = marker.get(r['type'], 'o')
        y    = ypos.get(r['type'], 1) + rng.uniform(-0.15, 0.15)
        edge = 'black' if r['regime'] == 'FINITE-L' else 'none'
        ax2.scatter(r['r_cons'], y, color=c, marker=m, s=100,
                    edgecolors=edge, linewidths=1.2, zorder=3)
        ax2.annotate(r['name'], (r['r_cons'], y),
                     textcoords='offset points', xytext=(4, 4), fontsize=7.5)

    ax2.axvline(ASYM_CONS, color='#888888', ls='--', lw=1.5,
                label=f'threshold {ASYM_CONS:.4f}')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['repeat', 'single-domain', 'PFAM-wide'], fontsize=10)
    ax2.set_xlabel('r_conservation  =  S1 / S_rand', fontsize=11)
    ax2.set_title('r_cons by structural class\n'
                  'Black edge = FINITE-L;  no edge = ASYMPTOTIC', fontsize=10)
    ax2.set_xlim(0.30, 0.78)
    ax2.grid(True, axis='x', alpha=0.25)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(plot2_out, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot2_out}")

    return df


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify protein families by partition model regime.')
    parser.add_argument('--L_max', type=int, default=500,
                        help='Maximum L for partition DP tables (default: 500)')
    parser.add_argument('--occ', type=float, default=0.10,
                        help='Minimum column occupancy for match columns (default: 0.10)')
    parser.add_argument('--alignment_dir', type=str, default='.',
                        help='Directory containing .seed alignment files')
    parser.add_argument('--csv', type=str, default='pfam_regime_results.csv',
                        help='Output CSV filename')
    parser.add_argument('--plot1', type=str, default='pfam_regime_plot.png',
                        help='Output filename for r_cons vs L* plot')
    parser.add_argument('--plot2', type=str, default='pfam_rcons_distribution.png',
                        help='Output filename for r_cons distribution plot')
    args = parser.parse_args()

    ALIGNMENT_DIR = Path(args.alignment_dir)
    run(L_max=args.L_max,
        occupancy_threshold=args.occ,
        csv_out=args.csv,
        plot1_out=args.plot1,
        plot2_out=args.plot2)
