#!/usr/bin/env python3
"""
contact_density_analysis.py
============================
Computes per-residue contact density from PDB structures for the seven
FINITE-L protein families in the regime classification framework.

Contact density definition
--------------------------
    rho = n_contacts / N_residues

where a contact is a Cb-Cb pair (Ca for Gly) satisfying:
    distance <= D_CUTOFF  AND  |i - j| >= SEQ_SEP

D_CUTOFF = 8.0 Å   (Bhaskara & Bhattacharyya 2011; standard for globular proteins)
SEQ_SEP  = 5       (excludes local backbone / helix geometry; common choice)

USAGE
-----
# Fetch structures automatically from RCSB:
    python contact_density_analysis.py

# Use locally downloaded PDB files (place them in ./pdb/ or current dir):
    python contact_density_analysis.py --local-dir ./pdb

# Override cutoff parameters:
    python contact_density_analysis.py --cutoff 8.5 --sep 4

OBTAINING PDB FILES
-------------------
If running locally, download with:
    wget https://files.rcsb.org/download/1UBQ.pdb   (repeat for each accession)
or:
    python -c "import Bio.PDB; Bio.PDB.PDBList().retrieve_pdb_file('1UBQ', file_type='pdb')"

INPUTS (Finite-L family panel)
------
Family          PDB     Representative domain / notes
SH2             1SHC    SH2 domain chain A
SH3             1SRL    SH3 domain chain A
Kinase          1ATP    cAMP-dependent PKA catalytic subunit
Immunoglobulin  1NA0    Ig VH domain — expected outlier (positive selection)
PDZ             1M40    PDZ domain chain A
Ubiquitin       1UBQ    Ubiquitin (76 aa)
RRM             1BE9    RNA recognition motif chain A

OUTPUT
------
contact_density_results.csv   — family, PDB, chain, N_res, n_contacts, rho
contact_density_results.txt   — human-readable summary

DOWNSTREAM
----------
Feed rho values into rcons_structural_regression.py for:
    r_cons ~ MPSI + rho   (multiple regression, n=7)
"""

import argparse
import csv
import os
import sys
import urllib.request
from typing import Optional

import numpy as np

# ── Parameters (overridable via CLI) ──────────────────────────────────────
D_CUTOFF = 8.0
SEQ_SEP  = 5

# ── Family panel ──────────────────────────────────────────────────────────
# (family_name, pdb_id, chain_id)
# r_cons / MPSI columns left None here; fill in rcons_structural_regression.py
FAMILIES = [
    ("SH2",            "1SHC", "A"),
    ("SH3",            "1SRL", "A"),
    ("Kinase",         "1ATP", "A"),
    ("Immunoglobulin", "1NA0", "A"),
    ("PDZ",            "1M40", "A"),
    ("Ubiquitin",      "1UBQ", "A"),
    ("RRM",            "1BE9", "A"),
]

# ── PDB fetching ──────────────────────────────────────────────────────────

def fetch_pdb_remote(pdb_id: str) -> str:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  Fetching {pdb_id} from RCSB...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    print(f"OK ({len(text)//1024} KB)")
    return text


def load_pdb_local(pdb_id: str, local_dir: str) -> str:
    """Search local_dir for <pdb_id>.pdb (case-insensitive)."""
    for fname in [f"{pdb_id}.pdb", f"{pdb_id.lower()}.pdb", f"{pdb_id.upper()}.pdb",
                  f"pdb{pdb_id.lower()}.ent"]:
        fpath = os.path.join(local_dir, fname)
        if os.path.exists(fpath):
            print(f"  Loading {pdb_id} from {fpath}...", end=" ", flush=True)
            with open(fpath) as f:
                text = f.read()
            print(f"OK ({len(text)//1024} KB)")
            return text
    raise FileNotFoundError(
        f"{pdb_id} not found in {local_dir}. "
        f"Expected: {pdb_id}.pdb / {pdb_id.lower()}.pdb"
    )


# ── PDB parsing ──────────────────────────────────────────────────────────

class Residue:
    __slots__ = ("chain", "res_seq", "res_name", "cb_coord")
    def __init__(self, chain, res_seq, res_name):
        self.chain    = chain
        self.res_seq  = res_seq
        self.res_name = res_name
        self.cb_coord: Optional[np.ndarray] = None


def parse_chain(pdb_text: str, chain_id: str, model: int = 1) -> list:
    """
    Extract Cb (Ca for Gly) coordinates from chain_id, model 1.
    Returns sorted list of Residue objects with cb_coord set.
    """
    residues: dict[int, Residue] = {}
    in_model = False
    model_count = 0

    for line in pdb_text.splitlines():
        rec = line[:6].strip()

        if rec == "MODEL":
            model_count += 1
            in_model = (model_count == model)
            continue
        if rec == "ENDMDL":
            if in_model:
                break
            continue
        if rec != "ATOM":
            continue
        if model_count == 0:
            in_model = True   # single-model file

        if not in_model:
            continue

        try:
            chain     = line[21]
            res_seq   = int(line[22:26].strip())
            ins_code  = line[26].strip()          # insertion code — ignore alt positions
            res_name  = line[17:20].strip()
            atom_name = line[12:16].strip()
            alt_loc   = line[16].strip()
            if alt_loc and alt_loc != "A":        # skip alternate conformers B,C,...
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        if chain != chain_id:
            continue

        key = (res_seq, ins_code)
        if key not in residues:
            residues[key] = Residue(chain, res_seq, res_name)

        res = residues[key]
        is_gly = (res_name == "GLY")

        if atom_name == "CB" and not is_gly:
            res.cb_coord = np.array([x, y, z], dtype=np.float32)
        elif atom_name == "CA" and res.cb_coord is None:
            # Gly: use Ca; others: Ca only as fallback if CB missing
            res.cb_coord = np.array([x, y, z], dtype=np.float32)

    result = [r for r in sorted(residues.values(), key=lambda r: r.res_seq)
              if r.cb_coord is not None]
    return result


# ── Contact density ──────────────────────────────────────────────────────

def compute_contact_density(residues: list, d_cutoff: float, seq_sep: int) -> dict:
    n = len(residues)
    if n == 0:
        return {"n_residues": 0, "n_contacts": 0, "rho": 0.0}

    coords = np.stack([r.cb_coord for r in residues])   # (n, 3)

    # Vectorised distance matrix — feasible for domain-sized proteins
    diff = coords[:, None, :] - coords[None, :, :]      # (n, n, 3)
    dmat = np.sqrt((diff**2).sum(axis=-1))               # (n, n)

    # Upper triangle, excluding seq_sep band
    i_idx, j_idx = np.triu_indices(n, k=seq_sep)
    close = dmat[i_idx, j_idx] <= d_cutoff
    n_contacts = int(close.sum())

    return {
        "n_residues": n,
        "n_contacts": n_contacts,
        "rho":        n_contacts / n,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute Cb contact density from PDB structures")
    parser.add_argument("--local-dir", default=None,
                        help="Directory containing local .pdb files (skips RCSB fetch)")
    parser.add_argument("--cutoff", type=float, default=D_CUTOFF,
                        help=f"Cb-Cb distance cutoff in Å (default {D_CUTOFF})")
    parser.add_argument("--sep", type=int, default=SEQ_SEP,
                        help=f"Minimum sequence separation (default {SEQ_SEP})")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for CSV and TXT results")
    args = parser.parse_args()

    d_cutoff = args.cutoff
    seq_sep  = args.sep

    print(f"\nContact density analysis")
    print(f"  D_CUTOFF = {d_cutoff} Å   SEQ_SEP = {seq_sep}")
    if args.local_dir:
        print(f"  Mode: local files from {args.local_dir}")
    else:
        print(f"  Mode: remote fetch from RCSB")
    print()

    header = f"{'Family':<18} {'PDB':<6} {'Chain'} {'N_res':>6} {'N_contacts':>11} {'rho':>8}"
    print(header)
    print("-" * 58)

    results = []

    for (family, pdb_id, chain_id) in FAMILIES:
        try:
            if args.local_dir:
                pdb_text = load_pdb_local(pdb_id, args.local_dir)
            else:
                pdb_text = fetch_pdb_remote(pdb_id)

            residues = parse_chain(pdb_text, chain_id)

            if not residues:
                from collections import Counter
                chain_counts = Counter(
                    line[21] for line in pdb_text.splitlines()
                    if line[:4] == "ATOM"
                )
                best_chain = chain_counts.most_common(1)[0][0]
                residues = parse_chain(pdb_text, best_chain)
                chain_id = best_chain

            if not residues:
                raise ValueError(f"No residues with Cb/Ca found in {pdb_id}")

            stats = compute_contact_density(residues, d_cutoff, seq_sep)
            stats["family"]   = family
            stats["pdb_id"]   = pdb_id
            stats["chain"]    = chain_id
            results.append(stats)

            print(f"{family:<18} {pdb_id:<6} {chain_id:^5} "
                  f"{stats['n_residues']:>6} {stats['n_contacts']:>11} "
                  f"{stats['rho']:>8.3f}")

        except Exception as e:
            print(f"  ERROR {pdb_id}: {e}")
            results.append({"family": family, "pdb_id": pdb_id, "chain": chain_id,
                             "n_residues": None, "n_contacts": None, "rho": None})

    # ── Outputs ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "contact_density_results.csv")
    txt_path = os.path.join(args.out_dir, "contact_density_results.txt")

    fieldnames = ["family", "pdb_id", "chain", "n_residues", "n_contacts", "rho"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    with open(txt_path, "w") as f:
        f.write(f"Contact density analysis\n")
        f.write(f"D_CUTOFF={d_cutoff} Å  SEQ_SEP={seq_sep}\n\n")
        f.write(header + "\n" + "-"*58 + "\n")
        for r in results:
            if r.get("rho") is not None:
                f.write(f"{r['family']:<18} {r['pdb_id']:<6} {r['chain']:^5} "
                        f"{r['n_residues']:>6} {r['n_contacts']:>11} {r['rho']:>8.3f}\n")
            else:
                f.write(f"{r['family']:<18} {r['pdb_id']:<6}  ERROR\n")
        f.write("\nNotes:\n")
        f.write(f"  Contact: Cb-Cb <= {d_cutoff} Å, |i-j| >= {seq_sep}; Gly uses Ca\n")
        f.write("  rho = n_contacts / n_residues\n")
        f.write("  Chain A, model 1 (fallback to B/C/L/H if A empty)\n")
        f.write("\nNext: rcons_structural_regression.py\n")

    print(f"\nWritten: {csv_path}")
    print(f"Written: {txt_path}")

    # ── Immunoglobulin diagnostic ──────────────────────────────────────────
    good = [r for r in results if r.get("rho") is not None]
    if good:
        ig = next((r for r in good if r["family"] == "Immunoglobulin"), None)
        others = [r for r in good if r["family"] != "Immunoglobulin"]
        if ig and others:
            mean_rho_others = np.mean([r["rho"] for r in others])
            delta = ig["rho"] - mean_rho_others
            print(f"\nImmunoglobulin rho = {ig['rho']:.3f}")
            print(f"Mean rho (other 6)  = {mean_rho_others:.3f}")
            print(f"Δ (Ig − mean)       = {delta:+.3f} "
                  f"({'denser' if delta > 0 else 'sparser'} than panel mean)")
            print("\nInterpretation hint:")
            if delta < -0.5:
                print("  Low contact density in Ig is consistent with the open β-sandwich")
                print("  architecture and immunological need for accessible CDR loops.")
                print("  This structural sparsity may decouple sequence constraint from the")
                print("  packing-driven epistasis that drives finite L* in other families.")
            elif delta > 0.5:
                print("  Ig shows HIGHER contact density than the panel mean.")
                print("  If r_cons is nonetheless high, this would suggest that diversity-")
                print("  driven selection overrides packing constraint — revise hypothesis.")
            else:
                print("  Ig contact density is close to panel mean.")
                print("  Contact density alone may not explain the Ig outlier;")
                print("  consider loop flexibility or residue-depth metrics.")


if __name__ == "__main__":
    main()
