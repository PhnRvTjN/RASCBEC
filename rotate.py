#!/usr/bin/env python3

"""
Title: POSCAR Rotation Script for RASCBEC Finite-Difference Perturbations

Authors: Rui Zhang (original);
Phani Ravi Teja Nunna (refactor);
bug-fix simplification by preserving Cartesian rigid rotations

Date: 11/06/2024 (original); 2026-03-04 (refactor); 2026-03-11 (expanded); 2026-03-21 (bug-fix rewrite)

License: MIT License

Description
-----------
Prepares the 8 POSCAR subdirectories required by RASCBEC_phonopy.py.

The unrotated POSCAR is copied unchanged into ./1/ and ./m1/ (the
reference ± E-field calculations along the unrotated axes). Three
45°-rotated variants are written into ./x/, ./y/, ./z/ and their
negatives ./mx/, ./my/, ./mz/. The rotated and unrotated POSCARs
within each ± pair are identical — the sign of the field is set in
the INCAR, not the POSCAR.

Important bug-fix note
----------------------
Earlier versions applied the rotation as:

    rotated_lattice = R @ lattice_vectors

This is incorrect when the POSCAR lattice is stored in the normal VASP
layout where each lattice vector is written as a row. Left-multiplying
in that form mixes rows (a, b, c) with one another, so the code behaves
as if the row indices themselves were Cartesian axes. That only matches
the intended operation in the special case where the POSCAR rows happen
to align trivially with x, y, and z.

This script instead applies the rotation as:

    rotated_lattice = lattice_vectors @ R.T

Here each row is treated as a Cartesian row-vector and is rigidly rotated
through its x, y, z components. As a result:

- x rotation means a 45° rigid rotation in the Cartesian xy-plane
- y rotation means a 45° rigid rotation in the Cartesian yz-plane
- z rotation means a 45° rigid rotation in the Cartesian xz-plane

These labels refer to Cartesian component planes, not to specific POSCAR
rows. Therefore the rotation remains correct even if the lattice-vector
rows are permuted, e.g. a -> z, b -> x, c -> y. The row order is preserved;
the code rotates the vectors as written and does not reorder them.

Atomic-position convention
--------------------------
For POSCAR files written in Direct coordinates, the fractional coordinates
are left unchanged when the lattice is rotated. This is the correct rigid
rotation of the entire crystal, because the basis vectors themselves are
rotated while the coordinates in that basis remain the same.

For POSCAR files written in Cartesian coordinates, the atomic positions are
rotated with the same Cartesian rotation matrix used for the lattice.

In addition to the POSCAR, this script copies the VASP input files INCAR,
KPOINTS, and POTCAR into every subdirectory. Any *.sbatch job-submission
scripts found in the working directory (or supplied explicitly via
--sbatch) are also copied into each subdirectory.

Important post-copy steps (must be done manually)
-------------------------------------------------
1. Edit the INCAR in each subdirectory to set the correct EFIELD_PEAD
   direction and magnitude for that perturbation direction.
2. Rename the *.sbatch file(s) in each subdirectory as needed for your
   scheduler (for example to reflect the subdirectory label in the job name).

Output layout
-------------
./1/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — unrotated reference
./m1/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — unrotated reference
./x/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xy-plane
./mx/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xy-plane
./y/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in yz-plane
./my/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in yz-plane
./z/  {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xz-plane
./mz/ {POSCAR, INCAR, KPOINTS, POTCAR, *.sbatch}  — 45° rotation in xz-plane

Dependencies
------------
- numpy

Usage
-----
python rotate.py
python rotate.py --poscar path/to/POSCAR
python rotate.py --incar path/to/INCAR
python rotate.py --kpoints path/to/KPOINTS
python rotate.py --potcar path/to/POTCAR
python rotate.py --sbatch run1.sbatch run2.sbatch
python rotate.py --sbatch
python rotate.py --no-sbatch
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 45° Cartesian rigid rotations.
#
# The labels x/y/z are historical directory labels retained for compatibility
# with the existing RASCBEC workflow:
#   x -> rotate in the Cartesian xy-plane
#   y -> rotate in the Cartesian yz-plane
#   z -> rotate in the Cartesian xz-plane
#
# These rotations act on Cartesian components, not on POSCAR row indices.
# ---------------------------------------------------------------------------

ROTATION = {
    "x": np.array([
        [1 / np.sqrt(2),  1 / np.sqrt(2), 0],
        [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        [0,               0,              1],
    ]),
    "y": np.array([
        [1, 0,               0             ],
        [0,  1 / np.sqrt(2), 1 / np.sqrt(2)],
        [0, -1 / np.sqrt(2), 1 / np.sqrt(2)],
    ]),
    "z": np.array([
        [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
        [0,              1, 0              ],
        [1 / np.sqrt(2), 0,  1 / np.sqrt(2)],
    ]),
}


# ===========================================================================
# I. POSCAR reader / writer
# ===========================================================================

def read_poscar(path):
    """
    Read a VASP POSCAR and return the parsed structural data needed for
    rigid Cartesian rotation while preserving the original file layout.

    The function extracts:
    - the lattice vectors (stored by VASP as three row-vectors),
    - the atomic coordinates,
    - whether the coordinates are Direct or Cartesian,
    - whether a Selective dynamics line is present,
    - and any trailing per-atom flags such as T/F tags.

    The rest of the file is preserved verbatim so that the output POSCAR
    remains as close as possible to the input, except for the intended
    lattice and coordinate updates.

    Parameters
    ----------
    path : str or Path
        POSCAR file to parse.

    Returns
    -------
    data : dict
        Dictionary containing raw lines plus parsed lattice/coordinate data.
    """
    path = Path(path)

    with open(path, "r") as fh:
        lines = fh.readlines()

    lattice_vectors = np.array(
        [list(map(float, lines[i].split())) for i in range(2, 5)],
        dtype=float,
    )

    atom_counts = list(map(int, lines[6].split()))
    total_atoms = sum(atom_counts)

    coord_mode_idx = 7
    selective_dynamics = False

    if lines[coord_mode_idx].strip().lower().startswith("s"):
        selective_dynamics = True
        coord_mode_idx += 1

    coord_mode_line = lines[coord_mode_idx].strip()
    coordinate_type = coord_mode_line.lower()
    position_start = coord_mode_idx + 1

    atomic_positions = []
    trailing_tags = []

    for i in range(total_atoms):
        fields = lines[position_start + i].split()
        atomic_positions.append([float(fields[0]), float(fields[1]), float(fields[2])])
        trailing_tags.append(fields[3:])

    atomic_positions = np.array(atomic_positions, dtype=float)

    return {
        "lines": lines,
        "lattice_vectors": lattice_vectors,
        "atomic_positions": atomic_positions,
        "atom_counts": atom_counts,
        "total_atoms": total_atoms,
        "selective_dynamics": selective_dynamics,
        "coordinate_type": coordinate_type,
        "coord_mode_idx": coord_mode_idx,
        "position_start": position_start,
        "trailing_tags": trailing_tags,
    }


def write_poscar(dest, data, lattice_vectors, atomic_positions):
    """
    Write a POSCAR to *dest*, replacing the lattice vectors and atomic
    coordinates while preserving the original non-structural text.

    The comment line, scaling factor, species/count lines, optional
    Selective dynamics line, and coordinate-type line are copied from
    the original file. Any per-atom trailing tokens (for example T/F
    flags used with Selective dynamics) are also preserved.

    Parameters
    ----------
    dest : str or Path
        Output file path. The parent directory must already exist.
    data : dict
        Parsed POSCAR dictionary returned by read_poscar().
    lattice_vectors : ndarray, shape (3, 3)
        Rotated lattice vectors, still written as row-vectors in POSCAR order.
    atomic_positions : ndarray, shape (nat, 3)
        Atomic coordinates to write, either unchanged Direct coordinates
        or rotated Cartesian coordinates depending on the input file.
    """
    dest = Path(dest)

    lines = data["lines"]
    position_start = data["position_start"]
    total_atoms = data["total_atoms"]
    trailing_tags = data["trailing_tags"]

    with open(dest, "w") as fh:
        fh.writelines(lines[:2])

        for vec in lattice_vectors:
            fh.write(" " + " ".join(f"{v:.16f}" for v in vec) + "\n")

        fh.writelines(lines[5:position_start])

        for pos, tags in zip(atomic_positions, trailing_tags):
            if tags:
                fh.write(
                    " "
                    + " ".join(f"{x:.16f}" for x in pos)
                    + " "
                    + " ".join(tags)
                    + "\n"
                )
            else:
                fh.write(" " + " ".join(f"{x:.16f}" for x in pos) + "\n")

        trailing_start = position_start + total_atoms
        if trailing_start < len(lines):
            fh.writelines(lines[trailing_start:])


def rotate_structure(data, R):
    """
    Apply a rigid Cartesian rotation to the POSCAR structure.

    The lattice vectors in a POSCAR are stored as row-vectors. A rigid
    Cartesian rotation acting on row-vectors is therefore written as:

        L_rot = L @ R.T

    This rotates each row through its Cartesian x, y, z components and
    does not reorder rows. Consequently, the operation remains correct
    even when the POSCAR row order is permuted relative to intuitive
    axis labels.

    Coordinate handling follows VASP conventions:
    - Direct coordinates are left unchanged because the rotated lattice
      already carries the rigid-body transformation.
    - Cartesian coordinates are rotated with the same Cartesian matrix.

    Parameters
    ----------
    data : dict
        Parsed POSCAR dictionary returned by read_poscar().
    R : ndarray, shape (3, 3)
        Cartesian rotation matrix.

    Returns
    -------
    rotated_lattice : ndarray, shape (3, 3)
        Rigidly rotated lattice vectors.
    rotated_positions : ndarray, shape (nat, 3)
        Output coordinates in the same coordinate convention as the input.
    """
    lattice_vectors = data["lattice_vectors"]
    atomic_positions = data["atomic_positions"]
    coordinate_type = data["coordinate_type"]

    rotated_lattice = lattice_vectors @ R.T

    if coordinate_type.startswith("d"):
        rotated_positions = atomic_positions.copy()
    elif coordinate_type.startswith("c"):
        rotated_positions = atomic_positions @ R.T
    else:
        raise ValueError(
            "Unsupported POSCAR coordinate type line: "
            f"'{data['lines'][data['coord_mode_idx']].strip()}'"
        )

    return rotated_lattice, rotated_positions


# ===========================================================================
# II. Directory setup helpers
# ===========================================================================

def make_subdir(label):
    """
    Create a subdirectory if it does not already exist and return its Path.

    Parameters
    ----------
    label : str
        Directory name such as '1', 'm1', 'x', or 'mx'.

    Returns
    -------
    Path
        Path object pointing to the created or pre-existing directory.
    """
    d = Path(label)
    d.mkdir(exist_ok=True)
    return d


def copy_support_files(subdir, incar, kpoints, potcar, sbatch_files):
    """
    Copy VASP support files and optional scheduler scripts into a subdirectory.

    Missing optional files are skipped with a warning so that POSCAR generation
    can still proceed in partially prepared working directories.

    Parameters
    ----------
    subdir : Path
        Destination directory.
    incar, kpoints, potcar : str or Path or None
        Standard VASP input files to copy if present.
    sbatch_files : list[Path]
        Optional scheduler scripts to copy.
    """
    for src, label in [(incar, "INCAR"), (kpoints, "KPOINTS"), (potcar, "POTCAR")]:
        if src is None:
            print(f" Skipped : {label} (not provided)")
            continue

        src = Path(src)
        if not src.exists():
            print(f" Warning : {label} not found at '{src}' — skipped")
            continue

        shutil.copy(src, subdir / src.name)
        print(f" Copied : {src} -> {subdir / src.name}")

    for sbatch in sbatch_files:
        sbatch = Path(sbatch)
        if not sbatch.exists():
            print(f" Warning : sbatch file '{sbatch}' not found — skipped")
            continue

        shutil.copy(sbatch, subdir / sbatch.name)
        print(f" Copied : {sbatch} -> {subdir / sbatch.name}")


# ===========================================================================
# III. Main
# ===========================================================================

def main():
    """
    Parse command-line options, prepare the 8 RASCBEC subdirectories, write
    the rotated POSCAR files, and copy the requested support files.

    The directory naming and file layout are kept compatible with the existing
    RASCBEC workflow. Only the rotation implementation is simplified and fixed
    so that the operation is now a true rigid Cartesian rotation acting on the
    x/y/z components of the lattice vectors rather than on the POSCAR row order.
    """
    p = argparse.ArgumentParser(
        description=(
            "Generate rotated POSCARs for RASCBEC finite-difference runs "
            "and populate each subdirectory with INCAR, KPOINTS, POTCAR, "
            "and optional *.sbatch files."
        )
    )

    p.add_argument(
        "--poscar",
        default="POSCAR",
        help="Source POSCAR file (default: ./POSCAR)",
    )
    p.add_argument(
        "--incar",
        default="INCAR",
        help="INCAR file to copy into each subdirectory (default: ./INCAR)",
    )
    p.add_argument(
        "--kpoints",
        default="KPOINTS",
        help="KPOINTS file to copy into each subdirectory (default: ./KPOINTS)",
    )
    p.add_argument(
        "--potcar",
        default="POTCAR",
        help="POTCAR file to copy into each subdirectory (default: ./POTCAR)",
    )
    p.add_argument(
        "--sbatch",
        nargs="*",
        default=None,
        metavar="FILE",
        help=(
            "One or more *.sbatch files to copy into each subdirectory. "
            "If omitted, all *.sbatch files in the working directory are "
            "discovered automatically. Pass --sbatch with no arguments "
            "to suppress sbatch copying."
        ),
    )
    p.add_argument(
        "--no-sbatch",
        action="store_true",
        help="Disable automatic *.sbatch discovery and copying entirely.",
    )

    args = p.parse_args()

    if args.no_sbatch or args.sbatch == []:
        sbatch_files = []
    elif args.sbatch is None:
        sbatch_files = sorted(Path(".").glob("*.sbatch"))
        if sbatch_files:
            print(f"Auto-discovered sbatch files: {[str(f) for f in sbatch_files]}")
        else:
            print("No *.sbatch files found in working directory — skipping.")
    else:
        sbatch_files = [Path(f) for f in args.sbatch]

    data = read_poscar(args.poscar)

    for label in ("1", "m1"):
        subdir = make_subdir(label)
        dest = subdir / "POSCAR"
        shutil.copy(args.poscar, dest)
        print(f"\n[{label}]")
        print(f" Copied : {args.poscar} -> {dest}")
        copy_support_files(subdir, args.incar, args.kpoints, args.potcar, sbatch_files)

    for axis, R in ROTATION.items():
        rotated_lattice, rotated_positions = rotate_structure(data, R)

        for label in (axis, f"m{axis}"):
            subdir = make_subdir(label)
            dest = subdir / "POSCAR"
            write_poscar(dest, data, rotated_lattice, rotated_positions)
            print(f"\n[{label}]")
            print(f" Written : {dest} (rotation {axis})")
            copy_support_files(subdir, args.incar, args.kpoints, args.potcar, sbatch_files)

    print(
        "\nDone. Files copied into all 8 subdirectories.\n"
        "Remember to:\n"
        " 1. Edit INCAR in each subdirectory to set the correct EFIELD_PEAD\n"
        "    direction and magnitude for that perturbation.\n"
        " 2. Rename *.sbatch file(s) in each subdirectory as needed before\n"
        "    submitting to the scheduler."
    )


if __name__ == "__main__":
    main()