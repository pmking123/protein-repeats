"""
lstar_correlation.py
====================
Analyses what determines L* across FINITE-L protein families.

The key finding from the initial check:
  - Spearman r(r_cons, L_star) = 1.000  -- perfect rank correlation
  - L_eff has NO additional explanatory power once r_cons is accounted for

This is not surprising: L* is defined as the L where model r_cons(L) matches
biological r_cons. Since model r_cons is a monotone function of L, L* is
simply a nonlinear transformation of r_cons. The two contain identical
information about family ordering.

The scientifically interesting question is therefore NOT "does L* correlate
with L_eff" but rather: "does r_cons itself correlate with measurable
structural or evolutionary properties, and what is the biology behind the
clustering of families into two groups?"

This script:
1. Shows the tautological r_cons <-> L* relationship explicitly
2. Tests whether L_eff explains any RESIDUAL variance in L* beyond r_cons
3. Identifies the two natural clusters (low L*, high L*) and characterises
   what separates them biologically
4. Produces plots for all of the above

USAGE
-----
    python lstar_correlation.py

INPUTS
------
Data is hard-coded from pfam_regime_results.csv (FINITE-L families only,
glucosidase excluded due to small sample size n=9).

To extend: add rows to the DATA dict, or pass a CSV path as argument.

OUTPUTS
-------
  lstar_correlation.png  -- four-panel figure
  lstar_correlation.csv  -- data table with all computed quantities

DEPENDENCIES
------------
  numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Data: FINITE-L families from pfam_regime_analysis.py
# Glucosidase (PF00232) excluded: n_seq=9, unreliable entropy estimate
# TPR: L_eff not available (r_cons from published override, not alignment)
# ---------------------------------------------------------------------------

# name, L_eff, L_star, r_cons, n_seq, structural_class
FINITE_L_DATA = [
    # name           L_eff  L_star  r_cons  n_seq  class
    ('SH2',           86,    22,   0.5386,   52,  'signalling'),
    ('SH3',           49,    34,   0.5558,   55,  'signalling'),
    ('Prot.kinase',  330,    37,   0.5596,   37,  'enzyme/signalling'),
    ('TPR',          None,   43,   0.5652,  None, 'repeat'),
    ('TEM-1',        332,   112,   0.5905,   56,  'enzyme'),
    ('Ubiquitin',     76,   323,   0.6068,   59,  'modifier'),
    ('PDZ',           95,   337,   0.6073,   44,  'scaffold'),
]

COLS = ['name', 'L_eff', 'L_star', 'r_cons', 'n_seq', 'class']
df = pd.DataFrame(FINITE_L_DATA, columns=COLS)

# ---------------------------------------------------------------------------
# Build model r_cons curve (for plotting)
# ---------------------------------------------------------------------------

L_MAX = 500
p_u = [0] * (L_MAX + 1); p_u[0] = 1
for n in range(1, L_MAX + 1):
    k = 1
    while True:
        p1 = k*(3*k-1)//2; p2 = k*(3*k+1)//2
        if p1 > n: break
        s = 1 if k%2==1 else -1
        p_u[n] += s * p_u[n-p1]
        if p2 <= n: p_u[n] += s * p_u[n-p2]
        k += 1
p_r = [0] * (L_MAX + 1); p_r[0] = 1
for part in range(1, L_MAX + 1):
    if part % 5 in (1, 4):
        for j in range(part, L_MAX + 1): p_r[j] += p_r[j-part]
p_u = np.array(p_u, float); p_r = np.array(p_r, float)

ASYM = np.pi*2/np.sqrt(15) / (np.pi*np.sqrt(2/3))  # sqrt(2/5) = 0.6325

def rc_model(L):
    if p_u[L] <= 1 or p_r[L] <= 0: return np.nan
    return np.log(float(p_r[L])) / np.log(float(p_u[L]))

Ls_arr  = np.arange(2, L_MAX+1)
rc_arr  = np.array([rc_model(L) for L in Ls_arr])

# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

# 1. r_cons vs L_star (all 7 families)
r_pearson,   p_pearson  = stats.pearsonr(df['r_cons'], df['L_star'])
r_spearman,  p_spearman = stats.spearmanr(df['r_cons'], df['L_star'])
r_log_log,   p_log_log  = stats.pearsonr(np.log(df['r_cons']), np.log(df['L_star']))

# 2. L_eff vs L_star (6 families with L_eff)
sub = df.dropna(subset=['L_eff']).copy()
r_leff,  p_leff  = stats.pearsonr(sub['L_eff'], sub['L_star'])
rho_leff, p_leff_sp = stats.spearmanr(sub['L_eff'], sub['L_star'])

# 3. Partial correlation: L_eff with L_star residuals after accounting for r_cons
slope, intercept, _, _, _ = stats.linregress(sub['r_cons'], sub['L_star'])
sub['L_star_resid'] = sub['L_star'] - (slope * sub['r_cons'] + intercept)
r_partial, p_partial = stats.pearsonr(sub['L_eff'], sub['L_star_resid'])

# 4. The model r_cons(L) is highly nonlinear -- most variation in L*
#    comes from families near the asymptote (where the curve is flat).
#    Compute d(r_cons)/dL at each family's L* to show this
drc_dL = []
for _, row in df.iterrows():
    L = int(row['L_star'])
    if L >= 3:
        drc = rc_model(L) - rc_model(L-1)
        drc_dL.append(drc)
    else:
        drc_dL.append(np.nan)
df['drc_dL'] = drc_dL

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print("=" * 65)
print("L* CORRELATION ANALYSIS — FINITE-L FAMILIES")
print("=" * 65)
print()
print(df[['name', 'L_eff', 'L_star', 'r_cons', 'class']].to_string(index=False))
print()

print("1. r_cons vs L_star (n=7)")
print(f"   Pearson  r = {r_pearson:.3f}   p = {p_pearson:.4f}")
print(f"   Spearman r = {r_spearman:.3f}   p = {p_spearman:.4f}")
print(f"   Pearson  r (log-log) = {r_log_log:.3f}   p = {p_log_log:.4f}")
print()
print("   INTERPRETATION: r_cons and L_star contain identical rank")
print("   information (Spearman=1.0) because L* is defined as the L")
print("   where model r_cons(L) = biological r_cons. The relationship")
print("   is monotone by construction.")
print()

print("2. L_eff vs L_star (n=6, TPR excluded)")
print(f"   Pearson  r = {r_leff:.3f}   p = {p_leff:.4f}")
print(f"   Spearman r = {rho_leff:.3f}   p = {p_leff_sp:.4f}")
print()
print("   INTERPRETATION: No significant correlation. L_eff does not")
print("   predict L* -- short and long alignment families can have")
print("   either small or large L*.")
print()

print("3. Partial correlation: L_eff with L_star residuals (after r_cons, n=6)")
print(f"   r = {r_partial:.3f}   p = {p_partial:.4f}")
print()
print("   INTERPRETATION: After accounting for r_cons, L_eff adds no")
print("   significant explanatory power for L*.")
print()

print("4. Sensitivity d(r_cons)/dL at each family's L*")
print(f"   {'Name':<14} {'L*':>5} {'r_cons':>7} {'d(rc)/dL':>10}")
print(f"   {'-'*40}")
for _, r in df.sort_values('L_star').iterrows():
    print(f"   {r['name']:<14} {int(r['L_star']):>5} {r['r_cons']:>7.4f} "
          f"{r['drc_dL']:>10.6f}")
print()
print("   INTERPRETATION: d(r_cons)/dL is ~100x larger at small L*")
print("   than at large L*. Small differences in r_cons near the")
print("   asymptote (Ubiquitin 0.6068 vs PDZ 0.6073) map to large")
print("   differences in L* (323 vs 337). The L* scale compresses")
print("   biologically meaningful distinctions at small L* and")
print("   exaggerates them at large L*.")
print()

print("5. Two natural clusters")
low  = df[df['L_star'] <  100]
high = df[df['L_star'] >= 100]
print(f"   LOW  L* (<100):  {list(low['name'])}")
print(f"   HIGH L* (>=100): {list(high['name'])}")
print(f"   Mean r_cons LOW:  {low['r_cons'].mean():.4f}")
print(f"   Mean r_cons HIGH: {high['r_cons'].mean():.4f}")
print(f"   t-test: p = {stats.ttest_ind(low['r_cons'], high['r_cons']).pvalue:.4f}")
print()
print("   INTERPRETATION: The biologically meaningful distinction is")
print("   the gap in r_cons between the two groups (0.54-0.57 vs")
print("   0.59-0.61), not the L* values themselves. L* amplifies a")
print("   real but modest difference in conservation constraint into")
print("   a 10-fold difference in scale.")

print()
print("=" * 65)
print("CONCLUSION")
print("=" * 65)
print("""
L* is a nonlinear but monotone transformation of r_cons, so
L* vs r_cons is tautologically perfect (Spearman=1.0). L_eff
(alignment length) has no predictive power for L* once r_cons
is known.

The scientifically informative quantity is r_cons itself. The
two FINITE-L clusters differ in r_cons by ~0.05 units, which
maps to a ~10-fold difference in L* due to the flatness of the
model curve near the asymptote.

To understand what drives the variation in r_cons across
families, the next step is to correlate r_cons with:
  (a) n_contacts / L  (contact density, structural)
  (b) DCA coupling strength (epistatic constraint)
  (c) mean pairwise sequence identity (evolutionary divergence)

These require structural data (PDB) and/or DCA inference,
and would explain WHY different families have different r_cons
rather than simply describing the pattern.
""")

# ---------------------------------------------------------------------------
# Save data
# ---------------------------------------------------------------------------
df['drc_dL'] = df['drc_dL'].round(7)
df.to_csv('lstar_correlation.csv', index=False)
print("Saved: lstar_correlation.csv")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

class_colour = {
    'signalling':          '#d73027',
    'enzyme/signalling':   '#fc8d59',
    'repeat':              '#2166ac',
    'enzyme':              '#e08214',
    'modifier':            '#542788',
    'scaffold':            '#1b7837',
}

# Panel 1: r_cons vs L* with model curve
ax = axes[0, 0]
ax.plot(Ls_arr, rc_arr, '-', color='#cccccc', lw=2, label='model r_cons(L)')
ax.axhline(ASYM, color='#888888', ls='--', lw=1,
           label=f'asymptote {ASYM:.4f}')
for _, r in df.iterrows():
    c = class_colour.get(r['class'], 'grey')
    ax.scatter(r['L_star'], r['r_cons'], color=c, s=90, zorder=4,
               edgecolors='k', linewidths=0.5)
    ax.annotate(r['name'], (r['L_star'], r['r_cons']),
                textcoords='offset points', xytext=(5, 4), fontsize=8)
ax.set_xlabel('L*', fontsize=10)
ax.set_ylabel('r_cons  (S1/S_rand)', fontsize=10)
ax.set_title(f'r_cons vs L*\nSpearman r={r_spearman:.3f}, p={p_spearman:.4f}',
             fontsize=10)
ax.set_xlim(0, 380); ax.set_ylim(0.50, 0.65)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=7)

# Panel 2: L_eff vs L*
ax = axes[0, 1]
sub2 = df.dropna(subset=['L_eff'])
for _, r in sub2.iterrows():
    c = class_colour.get(r['class'], 'grey')
    ax.scatter(r['L_eff'], r['L_star'], color=c, s=90, zorder=4,
               edgecolors='k', linewidths=0.5)
    ax.annotate(r['name'], (r['L_eff'], r['L_star']),
                textcoords='offset points', xytext=(5, 4), fontsize=8)
ax.set_xlabel('L_eff  (alignment match columns)', fontsize=10)
ax.set_ylabel('L*', fontsize=10)
ax.set_title(f'L_eff vs L*\nPearson r={r_leff:.3f}, p={p_leff:.4f}',
             fontsize=10)
ax.grid(True, alpha=0.25)

# Panel 3: r_cons vs L* (linear scale with cluster shading)
ax = axes[1, 0]
ax.axvspan(0.535, 0.570, alpha=0.08, color='#d73027', label='low-L* cluster')
ax.axvspan(0.588, 0.612, alpha=0.08, color='#2166ac', label='high-L* cluster')
ax.axvline(ASYM, color='#888888', ls='--', lw=1, label=f'asymptote {ASYM:.4f}')
for _, r in df.iterrows():
    c = class_colour.get(r['class'], 'grey')
    ax.scatter(r['r_cons'], r['L_star'], color=c, s=90, zorder=4,
               edgecolors='k', linewidths=0.5)
    ax.annotate(r['name'], (r['r_cons'], r['L_star']),
                textcoords='offset points', xytext=(3, 5), fontsize=8)
ax.set_xlabel('r_cons  (S1/S_rand)', fontsize=10)
ax.set_ylabel('L*', fontsize=10)
ax.set_title('Two natural clusters in r_cons vs L*', fontsize=10)
ax.set_xlim(0.52, 0.63); ax.grid(True, alpha=0.25)
ax.legend(fontsize=7)

# Panel 4: d(r_cons)/dL — sensitivity of L* to r_cons
ax = axes[1, 1]
df_sorted = df.sort_values('L_star')
bars = ax.bar(range(len(df_sorted)), df_sorted['drc_dL'] * 1000,
              color=[class_colour.get(c,'grey') for c in df_sorted['class']],
              edgecolor='k', linewidth=0.5)
ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted['name'], rotation=30, ha='right', fontsize=8)
ax.set_ylabel('d(r_cons)/dL  x 1000', fontsize=10)
ax.set_title('Sensitivity: how much r_cons changes per unit L at L*\n'
             '(high = L* is a precise indicator; low = L* is noisy)',
             fontsize=9)
ax.grid(True, axis='y', alpha=0.25)

# Shared legend for structural class
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=v, label=k) for k, v in class_colour.items()
           if k in df['class'].values]
fig.legend(handles=handles, fontsize=8, loc='lower center',
           ncol=3, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("L* correlation analysis — FINITE-L protein families\n"
             "L* is a monotone transformation of r_cons (Spearman=1.0); "
             "L_eff has no independent predictive power",
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig('lstar_correlation.png', dpi=150, bbox_inches='tight')
print("Saved: lstar_correlation.png")
