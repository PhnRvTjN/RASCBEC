#!/usr/bin/env python3

"""
Title: POSCAR Rotation Script for RASCBEC Finite-Difference Perturbations

Authors: Rui Zhang (original);
         Phani Ravi Teja Nunna (refactor)

Date: 11/06/2024 (original); 2026-03-04 (refactor)

License: MIT License

Description:
    Prepares the 8 POSCAR subdirectories required by RASCBEC_phonopy.py.

    The unrotated POSCAR is copied unchanged into ./1/ and ./m1/ (the
    reference ± E-field calculations along the unrotated axes).  Three
    45°-rotated variants are written into ./x/, ./y/, ./z/ and their
    negatives ./mx/, ./my/, ./mz/.  The rotated and unrotated POSCARs
    within each ± pair are identical — the sign of the field is set in
    the INCAR, not the POSCAR.

    Rotation matrices apply a 45° rotation of the lattice vectors so that
    the applied E-field projects equally onto two Cartesian axes, enabling
    the off-diagonal BEC derivative components to be recovered by finite
    differences (see RASCBEC paper, Zhang et al. 2025).

    Output layout (matches OUTCAR_MAP in RASCBEC_phonopy.py):
        ./1/POSCAR   ./m1/POSCAR   — unrotated reference
        ./x/POSCAR   ./mx/POSCAR   — 45° rotation in xy-plane
        ./y/POSCAR   ./my/POSCAR   — 45° rotation in yz-plane
        ./z/POSCAR   ./mz/POSCAR   — 45° rotation in xz-plane

Dependencies: numpy

Usage:
    python rotate.py            # reads ./POSCAR, creates all 8 subdirs
    python rotate.py --poscar path/to/POSCAR
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 45° rotation matrices for the three Cartesian planes.
# Each matrix rotates the lattice so the E-field projects onto two axes
# simultaneously, allowing off-diagonal BEC derivative recovery.
# ---------------------------------------------------------------------------

ROTATION = {
    'x': np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0],
                   [1/np.sqrt(2),  1/np.sqrt(2), 0],
                   [0,             0,             1]]),
    'y': np.array([[1, 0,              0            ],
                   [0, 1/np.sqrt(2), -1/np.sqrt(2) ],
                   [0, 1/np.sqrt(2),  1/np.sqrt(2) ]]),
    'z': np.array([[ 1/np.sqrt(2), 0, 1/np.sqrt(2)],
                   [ 0,            1, 0            ],
                   [-1/np.sqrt(2), 0, 1/np.sqrt(2)]]),
}

# ===========================================================================
# I. POSCAR reader / writer
# ===========================================================================

def read_poscar(path):
    """
    Read a VASP POSCAR and return raw lines plus parsed structural arrays.

    Only the lattice vectors and atomic positions are extracted for
    transformation; all other lines (comment, scale, species, counts,
    coordinate type) are preserved verbatim for writing.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    lines            : list[str] — all raw lines of the file
    lattice_vectors  : ndarray (3, 3) — one row per lattice vector (Å)
    atomic_positions : ndarray (nat, 3) — fractional coordinates
    """
    with open(path, 'r') as fh:
        lines = fh.readlines()
    lattice_vectors  = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])
    atom_counts      = list(map(int, lines[6].split()))
    total_atoms      = sum(atom_counts)
    atomic_positions = np.array([list(map(float, lines[8 + i].split()[:3]))
                                  for i in range(total_atoms)])
    return lines, lattice_vectors, atomic_positions

def write_poscar(dest, lines, lattice_vectors, atomic_positions):
    """
    Write a POSCAR to *dest*, replacing lattice vectors and atomic positions.

    The comment line, scale factor, species names, atom counts, and
    coordinate-type line are copied verbatim from *lines*.

    Parameters
    ----------
    dest             : str or Path — output file path (parent dir must exist)
    lines            : list[str]   — original POSCAR lines (for header/footer)
    lattice_vectors  : ndarray (3, 3)
    atomic_positions : ndarray (nat, 3)
    """
    with open(dest, 'w') as fh:
        fh.writelines(lines[:2])
        for vec in lattice_vectors:
            fh.write(f"  {'  '.join(f'{v:.16f}' for v in vec)}\n")
        fh.writelines(lines[5:8])
        for pos in atomic_positions:
            fh.write(f"  {'  '.join(f'{p:.16f}' for p in pos)}\n")

# ===========================================================================
# II. Directory setup helpers
# ===========================================================================

def make_subdir(label):
    """Create subdirectory ./<label>/ if it does not already exist."""
    d = Path(label)
    d.mkdir(exist_ok=True)
    return d

# ===========================================================================
# III. Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='Generate rotated POSCARs for RASCBEC finite-difference runs.')
    p.add_argument('--poscar', default='POSCAR',
        help='Source POSCAR file (default: ./POSCAR)')
    args = p.parse_args()

    lines, lattice_vectors, atomic_positions = read_poscar(args.poscar)

    # --- Unrotated reference: copy directly into ./1/ and ./m1/ ---
    for label in ('1', 'm1'):
        dest = make_subdir(label) / 'POSCAR'
        shutil.copy(args.poscar, dest)
        print(f"Copied  : {args.poscar} → {dest}")

    # --- Rotated variants: write into ./x/, ./mx/, ./y/, ./my/, ./z/, ./mz/ ---
    for axis, R in ROTATION.items():
        rotated_lattice = R @ lattice_vectors
        for label in (axis, f'm{axis}'):
            dest = make_subdir(label) / 'POSCAR'
            write_poscar(dest, lines, rotated_lattice, atomic_positions)
            print(f"Written : {dest}  (rotation {axis})")

    print("\nDone. Subdirectories ready for VASP LEPSILON runs.")

if __name__ == '__main__':
    main()
