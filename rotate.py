#!/usr/bin/env python3

"""
Title: POSCAR Rotation Script for RASCBEC Finite-Difference Perturbations

Authors: Rui Zhang (original);
         Phani Ravi Teja Nunna (refactor)

Date: 11/06/2024 (original); 2026-03-04 (refactor); 2026-03-11 (expanded); 2026-03-20 (row-vector fix); 2026-03-20 (row-order legacy routing)

License: MIT License

Description:
    Prepares the 8 POSCAR subdirectories required by RASCBEC_phonopy.py.

    The unrotated POSCAR is copied unchanged into ./1/ and ./m1/ (the
    reference ± E-field calculations along the unrotated axes). Three
    45°-rotated variants are written into ./x/, ./y/, ./z/ and their
    negatives ./mx/, ./my/, ./mz/. The rotated and unrotated POSCARs
    within each ± pair are identical — the sign of the field is set in
    the INCAR, not the POSCAR.

    Rotation matrices apply a 45° rotation of the lattice vectors so that
    the applied E-field projects equally onto two Cartesian axes, enabling
    the off-diagonal BEC derivative components to be recovered by finite
    differences (see RASCBEC paper, Zhang et al. 2025).

    Important convention correction:
    VASP POSCAR lattice vectors are written one vector per row. Therefore,
    a rigid Cartesian rotation that leaves fractional coordinates unchanged
    must update the lattice as A' = A @ R.T, not A' = R @ A. The latter
    mixes column-vector algebra with row-vector POSCAR storage and can
    shear/distort the written cell for non-cubic structures.

    Legacy naming behavior implemented here:
    The original RASCBEC rotate.py labels its three rotated outputs as x,
    y, and z, but those labels are tied to which two POSCAR lattice rows
    are mixed in the original implementation rather than to a fixed textbook
    Cartesian-axis naming convention. This corrected script preserves that
    legacy naming behavior while fixing the rigid-rotation algebra.

    Concretely, the corrected script routes legacy folders by row order:
    - x/ gets the rigid rotation about the physical axis associated with row a
    - y/ gets the rigid rotation about the physical axis associated with row b
    - z/ gets the rigid rotation about the physical axis associated with row c

    Example:
    If the input POSCAR rows correspond to a -> +y, b -> +z, c -> +x,
    then x/POSCAR receives the rigid rotation about physical Cartesian y,
    y/POSCAR receives the rigid rotation about physical Cartesian z, and
    z/POSCAR receives the rigid rotation about physical Cartesian x.

    In addition to the POSCAR, this script copies the VASP input files
    INCAR, KPOINTS, and POTCAR into every subdirectory. Any *.sbatch
    job-submission scripts found in the working directory (or supplied
    explicitly via --sbatch) are also copied into each subdirectory.

    Important post-copy steps (must be done manually):
    1. Edit the INCAR in each subdirectory to set the correct EFIELD_PEAD
       direction and magnitude for that perturbation direction.
    2. Rename the *.sbatch file(s) in each subdirectory as needed for your
       scheduler (e.g. to reflect the subdirectory label in the job name).

    Output layout (matches OUTCAR_MAP in RASCBEC_phonopy.py):
    ./1/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — unrotated reference
    ./m1/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — unrotated reference
    ./x/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — legacy x branch,
           routed to the physical axis associated with input row a
    ./mx/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — same POSCAR as ./x/
    ./y/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — legacy y branch,
           routed to the physical axis associated with input row b
    ./my/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — same POSCAR as ./y/
    ./z/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — legacy z branch,
           routed to the physical axis associated with input row c
    ./mz/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch} — same POSCAR as ./z/

    Dependencies: numpy

    Usage:
        python rotate.py
        python rotate.py --poscar path/to/POSCAR
        python rotate.py --incar path/to/INCAR
        python rotate.py --kpoints path/to/KPOINTS
        python rotate.py --potcar path/to/POTCAR
        python rotate.py --sbatch run1.sbatch run2.sbatch
        python rotate.py --sbatch
        python rotate.py --no-sbatch
        python rotate.py --axis-tol 0.95
        python rotate.py --strict-axis-report
"""

import argparse
import itertools
import shutil
from pathlib import Path

import numpy as np

PHYSICAL_ROTATION = {
    'x': np.array([
        [1, 0, 0],
        [0, 1/np.sqrt(2), -1/np.sqrt(2)],
        [0, 1/np.sqrt(2),  1/np.sqrt(2)],
    ], dtype=float),
    'y': np.array([
        [ 1/np.sqrt(2), 0, 1/np.sqrt(2)],
        [ 0,            1, 0],
        [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
    ], dtype=float),
    'z': np.array([
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [1/np.sqrt(2),  1/np.sqrt(2), 0],
        [0,             0,            1],
    ], dtype=float),
}

AXIS_VECTORS = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}


def read_poscar(path):
    """
    Read a VASP POSCAR and return raw lines plus parsed structural arrays.

    Only the lattice vectors and atomic positions are extracted for
    transformation; all other lines (comment, scale, species, counts,
    coordinate type) are preserved verbatim for writing.
    """
    with open(path, 'r') as fh:
        lines = fh.readlines()

    lattice_vectors = np.array(
        [list(map(float, lines[i].split())) for i in range(2, 5)],
        dtype=float,
    )
    atom_counts = list(map(int, lines[6].split()))
    total_atoms = sum(atom_counts)
    atomic_positions = np.array(
        [list(map(float, lines[8 + i].split()[:3])) for i in range(total_atoms)],
        dtype=float,
    )
    return lines, lattice_vectors, atomic_positions


def write_poscar(dest, lines, lattice_vectors, atomic_positions):
    """
    Write a POSCAR to *dest*, replacing lattice vectors and atomic positions.

    The comment line, scale factor, species names, atom counts, and
    coordinate-type line are copied verbatim from *lines*.
    """
    with open(dest, 'w') as fh:
        fh.writelines(lines[:2])
        for vec in lattice_vectors:
            fh.write(f" {' '.join(f'{v:.16f}' for v in vec)}\n")
        fh.writelines(lines[5:8])
        for pos in atomic_positions:
            fh.write(f" {' '.join(f'{p:.16f}' for p in pos)}\n")


def detect_axis_mapping(lattice_vectors):
    """
    Detect which lattice-vector rows are most closely aligned with the
    Cartesian x, y, and z axes.

    This helper does not permute the lattice vectors. It reports which
    physical Cartesian axis each POSCAR row is closest to, together with
    the sign and alignment score.
    """
    labels = ('x', 'y', 'z')
    row_norms = np.linalg.norm(lattice_vectors, axis=1)
    if np.any(row_norms == 0.0):
        raise ValueError('Encountered zero-length lattice vector; cannot detect Cartesian axis mapping.')

    unit_rows = lattice_vectors / row_norms[:, None]
    cart = np.vstack([AXIS_VECTORS[k] for k in labels])
    cosines = unit_rows @ cart.T
    scores = np.abs(cosines)

    best_perm = max(
        itertools.permutations(range(3)),
        key=lambda p: sum(scores[i, p[i]] for i in range(3))
    )

    mapping = {}
    for row_idx, axis_idx in enumerate(best_perm):
        signed_cos = cosines[row_idx, axis_idx]
        mapping[row_idx] = {
            'axis': labels[axis_idx],
            'sign': '+' if signed_cos >= 0 else '-',
            'score': float(abs(signed_cos)),
        }

    ambiguous = any(info['score'] < 0.90 for info in mapping.values())
    return mapping, ambiguous


def print_axis_mapping(mapping, ambiguous, tol):
    """
    Pretty-print the detected row-to-Cartesian-axis mapping.
    """
    print('Detected lattice-row orientation relative to Cartesian axes:')
    for row_idx in sorted(mapping):
        info = mapping[row_idx]
        flag = ' < weak>' if info['score'] < tol else ''
        print(
            f"  row {row_idx + 1} -> {info['sign']}{info['axis']} "
            f"(alignment = {info['score']:.6f}){flag}"
        )
    if ambiguous:
        print(
            ' Warning : row-to-axis mapping is not a clean Cartesian assignment. '
            'Routing will still proceed using the best row-order mapping, but '
            'you should verify whether the input cell is already oblique or '
            'otherwise not axis-aligned in the way you expect.'
        )


def print_legacy_routing(mapping):
    """
    Print how legacy x/y/z folders map onto physical Cartesian rotation axes.

    The routing rule is row-order based:
    - legacy x follows row a (row 1)
    - legacy y follows row b (row 2)
    - legacy z follows row c (row 3)
    """
    print('Legacy folder routing derived from detected row order:')
    for legacy_label, row_idx in [('x', 0), ('y', 1), ('z', 2)]:
        info = mapping[row_idx]
        print(
            f"  legacy {legacy_label} -> row {row_idx + 1} "
            f"({info['sign']}{info['axis']}) -> physical Cartesian {info['axis']}"
        )


def rigid_rotate_lattice(lattice_vectors, physical_axis):
    """
    Rigidly rotate POSCAR row-wise lattice vectors about a physical Cartesian axis.

    For POSCAR row-vector storage, the correct rigid-rotation update is
    A' = A @ R.T.
    """
    R = PHYSICAL_ROTATION[physical_axis]
    return lattice_vectors @ R.T


def make_subdir(label):
    """
    Create subdirectory ./<label>/ if it does not already exist.
    """
    d = Path(label)
    d.mkdir(exist_ok=True)
    return d


def copy_support_files(subdir, incar, kpoints, potcar, sbatch_files):
    """
    Copy VASP input files and SLURM job scripts into a subdirectory.
    """
    for src, label in [(incar, 'INCAR'), (kpoints, 'KPOINTS'), (potcar, 'POTCAR')]:
        if src is None:
            print(f" Skipped : {label} (not provided)")
            continue
        src = Path(src)
        if not src.exists():
            print(f" Warning : {label} not found at '{src}' — skipped")
            continue
        shutil.copy(src, subdir / src.name)
        print(f" Copied : {src} → {subdir / src.name}")

    for sbatch in sbatch_files:
        sbatch = Path(sbatch)
        if not sbatch.exists():
            print(f" Warning : sbatch file '{sbatch}' not found — skipped")
            continue
        shutil.copy(sbatch, subdir / sbatch.name)
        print(f" Copied : {sbatch} → {subdir / sbatch.name}")


def main():
    p = argparse.ArgumentParser(
        description=(
            'Generate rigidly rotated POSCARs for RASCBEC finite-difference runs '
            'and populate each subdirectory with INCAR, KPOINTS, POTCAR, and '
            '*.sbatch files. The script preserves legacy row-order naming '
            '(x->row a, y->row b, z->row c) while correcting the POSCAR '
            'row-vector rotation algebra.'
        )
    )
    p.add_argument('--poscar', default='POSCAR',
                   help='Source POSCAR file (default: ./POSCAR)')
    p.add_argument('--incar', default='INCAR',
                   help='INCAR file to copy into each subdirectory (default: ./INCAR)')
    p.add_argument('--kpoints', default='KPOINTS',
                   help='KPOINTS file to copy into each subdirectory (default: ./KPOINTS)')
    p.add_argument('--potcar', default='POTCAR',
                   help='POTCAR file to copy into each subdirectory (default: ./POTCAR)')
    p.add_argument('--sbatch', nargs='*', default=None, metavar='FILE',
                   help=(
                       'One or more *.sbatch files to copy into each subdirectory. '
                       'If omitted, all *.sbatch files in the working directory are '
                       'discovered automatically. Pass --sbatch with no arguments '
                       'to suppress sbatch copying.'
                   ))
    p.add_argument('--no-sbatch', action='store_true',
                   help='Disable automatic *.sbatch discovery and copying entirely.')
    p.add_argument('--axis-tol', type=float, default=0.90,
                   help='Reporting threshold for row-to-axis cosine alignment (default: 0.90).')
    p.add_argument('--strict-axis-report', action='store_true',
                   help='Abort if the reported row-to-axis mapping is weak or ambiguous.')
    args = p.parse_args()

    if args.no_sbatch or args.sbatch == []:
        sbatch_files = []
    elif args.sbatch is None:
        sbatch_files = sorted(Path('.').glob('*.sbatch'))
        if sbatch_files:
            print(f"Auto-discovered sbatch files: {[str(f) for f in sbatch_files]}")
        else:
            print('No *.sbatch files found in working directory — skipping.')
    else:
        sbatch_files = [Path(f) for f in args.sbatch]

    lines, lattice_vectors, atomic_positions = read_poscar(args.poscar)

    mapping, ambiguous = detect_axis_mapping(lattice_vectors)
    print_axis_mapping(mapping, ambiguous, args.axis_tol)
    print_legacy_routing(mapping)
    if ambiguous and args.strict_axis_report:
        raise ValueError(
            'Weak/ambiguous lattice-row report encountered under --strict-axis-report. '
            'Without that flag, the script would still proceed and route legacy '
            'folders according to the best row-order mapping available.'
        )

    for label in ('1', 'm1'):
        subdir = make_subdir(label)
        dest = subdir / 'POSCAR'
        shutil.copy(args.poscar, dest)
        print(f"\n[{label}]")
        print(f" Copied : {args.poscar} → {dest}")
        copy_support_files(subdir, args.incar, args.kpoints, args.potcar, sbatch_files)

    physical_cache = {}
    for physical_axis in ('x', 'y', 'z'):
        physical_cache[physical_axis] = rigid_rotate_lattice(lattice_vectors, physical_axis)

    legacy_to_physical = {
        'x': mapping[0]['axis'],
        'y': mapping[1]['axis'],
        'z': mapping[2]['axis'],
    }

    for legacy_label in ('x', 'y', 'z'):
        physical_axis = legacy_to_physical[legacy_label]
        rotated_lattice = physical_cache[physical_axis]
        for label in (legacy_label, f'm{legacy_label}'):
            subdir = make_subdir(label)
            dest = subdir / 'POSCAR'
            write_poscar(dest, lines, rotated_lattice, atomic_positions)
            print(f"\n[{label}]")
            print(
                f" Written : {dest} "
                f"(legacy {legacy_label} -> row-order physical Cartesian {physical_axis})"
            )
            copy_support_files(subdir, args.incar, args.kpoints, args.potcar, sbatch_files)

    print(
        '\nDone. Files copied into all 8 subdirectories.\n'
        'Remember to:\n'
        ' 1. Edit INCAR in each subdirectory to set the correct EFIELD_PEAD\n'
        '    direction and magnitude for that perturbation.\n'
        ' 2. Rename *.sbatch file(s) in each subdirectory as needed before\n'
        '    submitting to the scheduler.'
    )


if __name__ == '__main__':
    main()
