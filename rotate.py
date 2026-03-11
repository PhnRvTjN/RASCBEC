#!/usr/bin/env python3

"""
Title: POSCAR Rotation Script for RASCBEC Finite-Difference Perturbations

Authors: Rui Zhang (original);
         Phani Ravi Teja Nunna (refactor)

Date: 11/06/2024 (original); 2026-03-04 (refactor); 2026-03-11 (expanded)

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

    In addition to the POSCAR, this script copies the VASP input files
    INCAR, KPOINTS, and POTCAR into every subdirectory.  Any *.sbatch
    job-submission scripts found in the working directory (or supplied
    explicitly via --sbatch) are also copied into each subdirectory.

    Important post-copy steps (must be done manually):
      1. Edit the INCAR in each subdirectory to set the correct EFIELD_PEAD
         direction and magnitude for that perturbation direction.
      2. Rename the *.sbatch file(s) in each subdirectory as needed for your
         scheduler (e.g. to reflect the subdirectory label in the job name).

    Output layout (matches OUTCAR_MAP in RASCBEC_phonopy.py):
        ./1/   {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — unrotated reference
        ./m1/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — unrotated reference
        ./x/   {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xy-plane
        ./mx/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xy-plane
        ./y/   {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in yz-plane
        ./my/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in yz-plane
        ./z/   {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xz-plane
        ./mz/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xz-plane

Dependencies: numpy

Usage:
    python rotate.py                          # auto-discovers all input files in ./
    python rotate.py --poscar  path/to/POSCAR
    python rotate.py --incar   path/to/INCAR
    python rotate.py --kpoints path/to/KPOINTS
    python rotate.py --potcar  path/to/POTCAR
    python rotate.py --sbatch  run1.sbatch run2.sbatch  # explicit sbatch list
    python rotate.py --sbatch                           # suppress sbatch copying
    python rotate.py --no-sbatch                        # suppress sbatch copying
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
    lines            : list[str]
        All raw lines of the file.
    lattice_vectors  : ndarray (3, 3)
        One row per lattice vector (Å).
    atomic_positions : ndarray (nat, 3)
        Fractional coordinates.
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
    dest             : str or Path
        Output file path (parent directory must already exist).
    lines            : list[str]
        Original POSCAR lines used to preserve header and coordinate-type tag.
    lattice_vectors  : ndarray (3, 3)
        Lattice vectors to write (may be rotated relative to source).
    atomic_positions : ndarray (nat, 3)
        Fractional coordinates to write (unchanged by lattice rotation).
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
    """
    Create subdirectory ./<label>/ if it does not already exist.

    Parameters
    ----------
    label : str
        Directory name (e.g. '1', 'm1', 'x', 'mx').

    Returns
    -------
    d : Path
        Path object pointing to the created (or existing) directory.
    """
    d = Path(label)
    d.mkdir(exist_ok=True)
    return d


def copy_support_files(subdir, incar, kpoints, potcar, sbatch_files):
    """
    Copy VASP input files and SLURM job scripts into a subdirectory.

    Called once per subdirectory after the POSCAR has been placed.
    Missing optional files are skipped with a warning rather than raising
    an exception, allowing the script to run even when, for example, no
    *.sbatch file is present.

    Note: The copied INCAR is identical across all subdirectories.  Each
    subdirectory's INCAR must be edited manually afterward to set the
    correct EFIELD_PEAD tag (direction and magnitude) for that perturbation.
    Similarly, any copied *.sbatch files should be renamed as appropriate
    for your scheduler (e.g. to embed the subdirectory label in the job
    name) before submission.

    Parameters
    ----------
    subdir : Path
        Destination directory (must already exist).
    incar : Path or None
        INCAR file to copy; skipped if None or the path does not exist.
    kpoints : Path or None
        KPOINTS file to copy; skipped if None or the path does not exist.
    potcar : Path or None
        POTCAR file to copy; skipped if None or the path does not exist.
    sbatch_files : list[Path]
        Zero or more *.sbatch files to copy.  Each is copied preserving
        its original filename.  An empty list is silently accepted.

    Notes
    -----
    Files are copied with shutil.copy (preserves permissions but not
    metadata timestamps), consistent with the POSCAR copy strategy used
    in main().
    """
    for src, label in [(incar, 'INCAR'), (kpoints, 'KPOINTS'), (potcar, 'POTCAR')]:
        if src is None:
            print(f"  Skipped : {label} (not provided)")
            continue
        src = Path(src)
        if not src.exists():
            print(f"  Warning : {label} not found at '{src}' — skipped")
            continue
        shutil.copy(src, subdir / src.name)
        print(f"  Copied  : {src} → {subdir / src.name}")

    for sbatch in sbatch_files:
        sbatch = Path(sbatch)
        if not sbatch.exists():
            print(f"  Warning : sbatch file '{sbatch}' not found — skipped")
            continue
        shutil.copy(sbatch, subdir / sbatch.name)
        print(f"  Copied  : {sbatch} → {subdir / sbatch.name}")


# ===========================================================================
# III. Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description=(
            'Generate rotated POSCARs for RASCBEC finite-difference runs '
            'and populate each subdirectory with INCAR, KPOINTS, POTCAR, '
            'and *.sbatch files.'
        )
    )
    p.add_argument('--poscar',  default='POSCAR',
                   help='Source POSCAR file (default: ./POSCAR)')
    p.add_argument('--incar',   default='INCAR',
                   help='INCAR file to copy into each subdirectory (default: ./INCAR)')
    p.add_argument('--kpoints', default='KPOINTS',
                   help='KPOINTS file to copy into each subdirectory (default: ./KPOINTS)')
    p.add_argument('--potcar',  default='POTCAR',
                   help='POTCAR file to copy into each subdirectory (default: ./POTCAR)')
    p.add_argument('--sbatch',  nargs='*', default=None, metavar='FILE',
                   help=(
                       'One or more *.sbatch files to copy into each subdirectory. '
                       'If omitted, all *.sbatch files in the working directory are '
                       'discovered automatically. Pass --sbatch with no arguments '
                       'to suppress sbatch copying.'
                   ))
    p.add_argument('--no-sbatch', action='store_true',
                   help='Disable automatic *.sbatch discovery and copying entirely.')
    args = p.parse_args()

    # Resolve sbatch file list
    if args.no_sbatch or args.sbatch == []:
        sbatch_files = []
    elif args.sbatch is None:
        sbatch_files = sorted(Path('.').glob('*.sbatch'))
        if sbatch_files:
            print(f"Auto-discovered sbatch files: {[str(f) for f in sbatch_files]}")
        else:
            print("No *.sbatch files found in working directory — skipping.")
    else:
        sbatch_files = [Path(f) for f in args.sbatch]

    lines, lattice_vectors, atomic_positions = read_poscar(args.poscar)

    # --- Unrotated reference: copy POSCAR directly into ./1/ and ./m1/ ---
    for label in ('1', 'm1'):
        subdir = make_subdir(label)
        dest   = subdir / 'POSCAR'
        shutil.copy(args.poscar, dest)
        print(f"\n[{label}]")
        print(f"  Copied  : {args.poscar} → {dest}")
        copy_support_files(subdir, args.incar, args.kpoints, args.potcar, sbatch_files)

    # --- Rotated variants: write into ./x/, ./mx/, ./y/, ./my/, ./z/, ./mz/ ---
    for axis, R in ROTATION.items():
        rotated_lattice = R @ lattice_vectors
        for label in (axis, f'm{axis}'):
            subdir = make_subdir(label)
            dest   = subdir / 'POSCAR'
            write_poscar(dest, lines, rotated_lattice, atomic_positions)
            print(f"\n[{label}]")
            print(f"  Written : {dest}  (rotation {axis})")
            copy_support_files(subdir, args.incar, args.kpoints, args.potcar, sbatch_files)

    print(
        "\nDone. Files copied into all 8 subdirectories.\n"
        "Remember to:\n"
        "  1. Edit INCAR in each subdirectory to set the correct EFIELD_PEAD\n"
        "     direction and magnitude for that perturbation.\n"
        "  2. Rename *.sbatch file(s) in each subdirectory as needed before\n"
        "     submitting to the scheduler."
    )


if __name__ == '__main__':
    main()
