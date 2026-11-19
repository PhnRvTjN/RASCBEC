#!/usr/bin/env python3

"""
Title: Calculation of Raman Activities from VASP DFPT Output Files

Authors: Rui Zhang (original);
Phani Ravi Teja Nunna (refactor and enhancements)

Patches: auto atomic-mass lookup via pymatgen,
variable mode count (filtered imaginary modes),
subdirectory-based OUTCAR paths (./1/OUTCAR, ./m1/OUTCAR, ...),
chemistry-based output filenames derived from POSCAR composition,
CSV metadata header (formula, dopants, file label, E-field),
auto-read EFIELD_PEAD from OUTCAR,
plot output option,
auto-generated informative plot title,
basis-group and dopant roles (--bas, --dops) with base-formula reconstruction,
vectorised Raman activity loop via numpy einsum,
reverse-scan OUTCAR BEC reader,
temperature/laser-wavelength Stokes intensity options (--temp, --laser-wl),
--jobs parallel BEC file reads (thread pool, I/O bound).

Date: 2025-04-28 (original); 2026-02-23 (patched); 2026-03-04 (chemistry naming);
      2026-11-16 (parity with RASCBEC_phonopy.py: --bas/--dops roles, base-formula
                  reconstruction, vectorised activity loop, reverse-scan BEC reader,
                  --temp/--laser-wl, shared two-line plot title);
      2026-11-19 (--jobs parallel BEC file reads)

License: GNU GENERAL PUBLIC LICENSE

Cite: Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation
via born effective charge." Computer Physics Communications 307 (2025): 109425

Description:
Reads VASP DFPT output to calculate Raman activities per phonon mode
using the RASCBEC method.

Key distinctions from RASCBEC_phonopy.py:
- Phonon input is the flat VASP DFPT export (freqs_vasp.dat / eigvecs_vasp.dat),
  not phonopy YAML; no irrep labels, no NAC multi-direction support
  (use RASCBEC_phonopy.py for NAC).
- VASP outputs frequencies already in cm-1 (no THz conversion needed).
- VASP eigenvectors are already mass-normalised (no /sqrt(mass) step).
- All other routines (BEC reading, dZ*/dE tensor, Raman activity, chemistry
  metadata, CSV format, plotting) are identical to RASCBEC_phonopy.py.

E-field magnitude is read automatically from ./1/OUTCAR (EFIELD_PEAD tag).
Override with --E only if you want to force a specific value.

Required files (in working directory):
POSCAR             -- structure file
./1/OUTCAR         -- BEC, E-field along unrotated +
./m1/OUTCAR        -- BEC, E-field along unrotated -
./x/OUTCAR         -- BEC, E-field along +x (rotated along x)
./mx/OUTCAR        -- BEC, E-field along -x
./y/OUTCAR         -- BEC, E-field along +y (rotated along y)
./my/OUTCAR        -- BEC, E-field along -y
./z/OUTCAR         -- BEC, E-field along +z (rotated along z)
./mz/OUTCAR        -- BEC, E-field along -z
freqs_vasp.dat     -- phonon frequencies, shape (N_modes,), unit cm-1
eigvecs_vasp.dat   -- mass-normalised eigenvectors, shape (3N x N_modes)

Output naming (chemistry-based, derived from POSCAR; no directory tags):
<formula>[_<dopantpart>... ]_E<E>.csv / .png

Examples:
undoped Na3PS4, E=0.02  -> Na3PS4_E0.02.csv
Ca+Cl co-doped Na3PS4   -> Na3PS4_Ca01-Cl06_E0.02.csv

Dopant roles (--bas / --dops)
-----------------------------
Explicit --bas / --dops values take precedence over automatic inference
(dopants = species below 5 % of the supercell atoms, DOPANT_THRESHOLD).
When --bas is given, it defines the host-formula normalization AND the
denominator for the dopant x-values (x = n_dopant / n_bas), e.g.
Li5La9Ti16O48 --bas Ti -> Li0.31La0.56TiO3. When omitted, the
least-abundant host species is used as the legacy reference.

--bas may also be a `+` joined group of species (e.g. --bas Ti+Ga). The
group's member counts are summed into a single reference, so the group is
normalized to one and may include dopant species. For doped LLTO this
normalizes the formula so Ti + Ga = 1 (e.g. Li60La90Ti150O360 with
--bas Ti+Ga -> Li0.38La0.56Ti0.94O2.25, and dopant x-values reported
per 160 Ti+Ga sites).

Base (undoped) formula
-----------------------
The stored/# Formula value is the BASE host composition, not the raw
doped host-only count. The supercell contains only the host atoms that were
NOT displaced, so the base is recovered by adding back the atoms each dopant
took (or added) at every host element (see DOPANT_SUBSTITUTIONS), then
normalizing to the basis reference. E.g. a Ca-doped Na3PS4 supercell whose
doped host counts are Na20P8S30 (Ca=1, Cl=2) is reported as the base
Na3PS4, not Na2.5P1S3.75.

CSV metadata header (parsed by compare_raman.py & plot_raman.py):
# Formula: Na3PS4
# Dopants: Ca=0.125 Cl=0.0625   <- omitted if undoped
# File_Label: Ca01-Cl06         <- omitted if undoped
# E_field: 0.02
# Mode,Freq_cm-1,Activity
0001,<freq>,<activity>
...

Dependencies: argparse, pathlib, re, numpy, pymatgen, plot_raman

Usage:
python RASCBEC_VASP.py                          # all defaults
python RASCBEC_VASP.py --E 0.02 --gamma 0.2
python RASCBEC_VASP.py --no-plot
python RASCBEC_VASP.py --freq-min 50 --freq-max 600
python RASCBEC_VASP.py --bas Ti+Ga --dops Y Ga  # doped LLTO, Ti+Ga = 1
python RASCBEC_VASP.py --jobs 8                 # parallel BEC file reads
python RASCBEC_VASP.py --help
(existing output files are always overwritten -- no flag required)
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pymatgen.core import Element

from plot_raman import (
    plot_raman_spectrum,
    build_raman_title,
    DOPANT_SUBSTITUTIONS,
    DEFAULT_TEMP,
    DEFAULT_LASER_WL,
)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# VASP frequencies are in cm-1 — no THz conversion needed.
# cm-1 -> meV: ratio of CODATA THz conversion factors.
CM1MEV = 4.13566553853599 / 33.3564095198  # 0.12398419843 meV per cm-1
# 1 THz in cm-1, used only to express --gamma in THz like RASCBEC_phonopy.py.
THZCM1 = 33.3564095198

EPS0   = 55.2635e-4        # epsilon_0 in e^2 / (eV*Ang)
KB_T   = 8.617333e-2 * 298 # k_B T at 298 K, meV

# Species whose supercell fractional count falls below this threshold
# are classified as dopants (e.g. 1 Ca in 108 atoms ~ 0.009 < 0.05)
DOPANT_THRESHOLD = 0.05

# Subdirectory layout expected for the 8 finite-difference OUTCAR files.
# Each subdirectory contains a VASP LEPSILON run with the electric field
# applied along the indicated Cartesian direction.
OUTCAR_MAP = {
    '1' : './1/OUTCAR',
    'm1': './m1/OUTCAR',
    'x' : './x/OUTCAR',
    'mx': './mx/OUTCAR',
    'y' : './y/OUTCAR',
    'my': './my/OUTCAR',
    'z' : './z/OUTCAR',
    'mz': './mz/OUTCAR',
}


# ===========================================================================
# I.  Argument parser
# ===========================================================================

def parse_args():
    """Return parsed CLI arguments."""
    p = argparse.ArgumentParser(
        description='RASCBEC Raman activity calculator (VASP DFPT eigvecs).')
    p.add_argument('--E', type=float, default=None,
                   help='EFIELD_PEAD magnitude in eV/Å. If omitted, auto-read from '
                        './1/OUTCAR (the unscaled reference calculation).')
    p.add_argument('--gamma', type=float, default=0.25,
                   help='Lorentzian FWHM in THz for spectral broadening (default: 0.25);'
                        'converted to cm-1 internally via THZCM1.')
    p.add_argument('--temp', type=float, default=DEFAULT_TEMP,
                   help='Sample temperature in K for Bose-Einstein factor (default: 300.0 K)')
    p.add_argument('--laser-wl', type=float, default=DEFAULT_LASER_WL,
                   help='Excitation laser wavelength in nm (default: 632.8 nm)')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit of plot in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit of plot in cm-1 (default: auto)')
    p.add_argument('--bas', default=None,
                   help='Explicit basis species (or + joined group) used as the '
                        'concentration reference. It is normalized to '
                        'coefficient one in the host formula and defines the '
                        'denominator for dopant x-values (x = n_dop / n_bas). '
                        'A group sums its member counts, e.g. --bas Ti or '
                        '--bas Ti+Ga. If omitted, the least-abundant host '
                        'species is used.')
    p.add_argument('--dops', nargs='+', default=None,
                   help='Explicit dopant species list, e.g. --dops Ca Cl. '
                        'Overrides automatic dopant inference '
                        '(species below 5 %% of the supercell atoms).')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip plot generation; write CSV only')
    p.add_argument('--no-sticks', action='store_true',
                   help='Omit stick spectrum from plot')
    p.add_argument('--n-labels', type=int, default=10,
                   help='Number of peak labels to annotate on plot (default: 10)')
    p.add_argument('--out-csv', default=None,
                   help='Output CSV path (default: <chem>_E<E>.csv)')
    p.add_argument('--out-png', default=None,
                   help='Output PNG path (default: <chem>_E<E>.png)')
    # Outputs always overwrite existing files (no --overwrite / --no-overwrite flag).
    p.add_argument('--jobs', type=int, default=None,
                   help='Number of worker threads for the eight BEC file reads '
                        '(I/O bound). Default: 1 (sequential); raise to read '
                        'the OUTCARs in parallel (e.g. --jobs 8).')
    return p.parse_args()


# ===========================================================================
# II.  Structure reader
# ===========================================================================

def get_mass(symbol):
    """Return the standard atomic mass (amu) for *symbol* via pymatgen."""
    return float(Element(symbol).atomic_mass)


def structure_info(poscar='POSCAR'):
    """
    Parse a VASP POSCAR and return structural metadata.

    The POSCAR is read line-by-line following the standard format:
      line 1   : comment
      line 2   : universal scale factor
      lines 3-5: lattice vectors (Ang)
      line 6   : element symbols
      line 7   : atom counts per element
      line 8+  : coordinate type and positions

    Atomic masses are looked up automatically from pymatgen's Element
    database, avoiding hard-coded mass tables.  In the VASP variant,
    masses are used only for display/metadata — eigenvectors are already
    mass-normalised by VASP and no manual weighting is applied.

    Parameters
    ----------
    poscar : str — path to POSCAR (default 'POSCAR')

    Returns
    -------
    vol           : float     — unit-cell volume (Ang^3)
    species_n     : ndarray   — atom counts per species, shape (ntype,)
    atomic_mass   : list      — atomic masses (amu) per species
    ntype         : int       — number of distinct species
    nat           : int       — total number of atoms
    modes         : int       — 3*nat (theoretical maximum mode count)
    species_names : list[str] — element symbol strings
    """
    with open(poscar, 'r') as f:
        lines = f.readlines()
    box           = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)])
    vol           = np.linalg.det(box)
    species_names = lines[5].split()
    species_n     = np.array([int(x) for x in lines[6].split()])
    atomic_mass   = [get_mass(s) for s in species_names]
    ntype         = len(species_n)
    nat           = int(sum(species_n))
    modes         = nat * 3
    print(f"Species : {species_names}")
    print(f"Counts  : {species_n.tolist()}")
    print(f"Masses  : {[round(m, 4) for m in atomic_mass]} amu  (via pymatgen)")
    print(f"Atoms   : {nat}  |  Max modes: {modes}")
    return vol, species_n, atomic_mass, ntype, nat, modes, species_names


# ===========================================================================
# III.  Born Effective Charge readers
# ===========================================================================

def get_charges_from_OUTCAR(outcar_path, nat):
    """
    Extract the Born Effective Charge (BEC) tensor from a VASP OUTCAR.

    Scans for the section header
    "BORN EFFECTIVE CHARGES (including local field effects)"
    and reads the 3x3 tensor for each atom. Only the last occurrence
    is returned (VASP may write multiple BEC blocks during SCF).

    The file is read once into memory and reverse-scanned so that only
    the final BEC block is parsed. This avoids re-reading gigabyte-scale
    OUTCAR files for a block that always appears near the end.

    BEC block layout in OUTCAR (starting from the header line):
    line +0 : "BORN EFFECTIVE CHARGES (including local field effects)"
    line +1 : " -------..." (separator)
    line +2 : " ion    1"
    line +3 : "   1  Zxx  Zxy  Zxz"
    line +4 : "   2  Zyx  Zyy  Zyz"
    line +5 : "   3  Zzx  Zzy  Zzz"
    line +6 : " ion    2"
    ...  (each ion occupies 4 lines: 1 header + 3 tensor rows)

    Parameters
    ----------
    outcar_path : str -- path to the OUTCAR file
    nat         : int -- number of atoms in the cell

    Returns
    -------
    charges : list[list[list[float]]] -- shape (nat, 3, 3)
    charges[i][alpha][beta] = Z*_{i,alpha beta}

    Raises
    ------
    ValueError -- if no BEC block is found in the file
    """
    header = "BORN EFFECTIVE CHARGES (including local field effects)"
    with open(outcar_path, 'r') as fh:
        lines = fh.readlines()
    for idx in range(len(lines) - 1, -1, -1):
        if header in lines[idx]:
            base    = idx + 2   # first "ion N" header line (skip separator at idx+1)
            charges = []
            for i in range(nat):
                ion_base = base + i * 4
                charges.append([
                    list(map(float, lines[ion_base + row + 1].split()[1:4]))
                    for row in range(3)
                ])
            return charges
    raise ValueError(f"No BEC block found in {outcar_path}.")


def _resolve_worker_count(requested, default):
    """Resolve a --jobs value to a positive worker count (1 = sequential)."""
    if requested is None:
        return max(1, default)
    if requested < 1:
        raise ValueError('--jobs must be >= 1 (use --jobs 1 to force sequential processing).')
    return requested

def load_bec_charges(outcar_map, nat, jobs=None):
    """
    Read BEC tensors from all 8 finite-difference OUTCAR subdirectories.

    Each subdirectory corresponds to one Cartesian E-field direction.
    The resulting dict is keyed by the same labels used in OUTCAR_MAP
    and later consumed by build_charge_derivatives().

    The reads are I/O bound, so they are executed on a thread pool when
    ``jobs`` > 1. The returned dict preserves the insertion order of
    ``outcar_map`` regardless of worker completion order.

    Parameters
    ----------
    outcar_map : dict -- {label: path} as in OUTCAR_MAP
    nat        : int  -- number of atoms
    jobs       : int or None -- worker threads (None -> 1, i.e. sequential)

    Returns
    -------
    charges : dict[str, list[nat x 3 x 3]] -- BEC tensors keyed by direction label
    """
    def _read(key_path):
        key, path = key_path
        print(f"  Reading BEC: {path}")
        return key, get_charges_from_OUTCAR(path, nat)

    items = list(outcar_map.items())
    if jobs is None or jobs <= 1 or len(items) <= 1:
        return dict(_read(item) for item in items)
    workers = min(_resolve_worker_count(jobs, 1), len(items))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_read, items))


def read_efield_pead(outcar_path='./1/OUTCAR'):
    """
    Parse the EFIELD_PEAD magnitude from a VASP OUTCAR.

    VASP writes EFIELD_PEAD twice in the OUTCAR:
        1st (INCAR echo):   EFIELD_PEAD = 0.04 0.04 0.04  (Finite Electric Field...)
        2nd (VASP-parsed):  EFIELD_PEAD=    0.0400    0.0400    0.0400

    The second occurrence is VASP's own reformatted version — no trailing
    text, consistent spacing — and is what VASP actually uses internally.
    We skip the first match and read the second.

    The largest absolute component is returned as E, robust against
    near-zero values in the inactive Cartesian directions.

    Only ./1/OUTCAR (the unscaled reference) should be passed here.
    The rotated-field OUTCARs carry a sqrt(2)-scaled field.

    Parameters
    ----------
    outcar_path : str — path to the unscaled OUTCAR (default './1/OUTCAR')

    Returns
    -------
    E : float — electric field magnitude in eV/Ang

    Raises
    ------
    ValueError — if two EFIELD_PEAD occurrences are not found in the file
    """
    pattern = re.compile(
        r'EFIELD_PEAD\s*=\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', re.IGNORECASE)
    matches_found = 0
    with open(outcar_path, 'r') as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                matches_found += 1
                if matches_found == 2:
                    components = [float(m.group(i)) for i in range(1, 4)]
                    return max(abs(c) for c in components)
    raise ValueError(
        f"Expected 2 EFIELD_PEAD entries in {outcar_path}, found {matches_found}. "
        "Confirm the calculation used LCALCEPS = .TRUE. or LPEAD = .TRUE.")


# ===========================================================================
# IV.  BEC derivative tensor
# ===========================================================================

def charge_derivative(charge1, chargem1, chargex, chargey, chargez,
                      chargemx, chargemy, chargemz, E):
    """
    Compute the 3rd-rank BEC derivative tensor dZ*/dE for one atom.

    The tensor is assembled by finite differences from the 8 perturbed BEC
    tensors.  Diagonal components use a simple central difference along each
    Cartesian axis; off-diagonal components are recovered from the
    rotated-field perturbations following the RASCBEC scheme
    (see supplementary material of Zhang et al. 2025).

    Index convention:
        dq[i, j, k] = dZ*_{ij} / dE_k

    Parameters
    ----------
    charge{1,m1,x,y,z,mx,my,mz} : list[3x3]
        BEC tensor for the atom under each E-field direction.
        '1'/'m1'  = unrotated field +-;
        'x'/'mx', 'y'/'my', 'z'/'mz' = 45-degree-rotated field +-.
    E : float — electric field magnitude (eV/Ang)

    Returns
    -------
    dq : ndarray, shape (3, 3, 3)
    """
    c1, cm1, cx, cy, cz, cmx, cmy, cmz = (
        np.array(ch).T for ch in
        (charge1, chargem1, chargex, chargey, chargez, chargemx, chargemy, chargemz)
    )

    dq = np.zeros((3, 3, 3))

    # --- diagonal: central finite difference dZ*_ij / dE_j ---
    for i in range(3):
        for j in range(3):
            dq[i, j, j] = (c1[i, j] - cm1[i, j]) / E

    # --- off-diagonal components via rotated-field perturbations ---

    # x-rotated perturbation
    dq[2,0,1]=dq[2,1,0]=0.5*(np.sqrt(2)*(cx[2,0]-cmx[2,0])-(c1[2,0]-cm1[2,0])-(c1[2,1]-cm1[2,1]))/E
    dq[0,0,1]=dq[0,1,0]=0.5*(cx[0,0]-cmx[0,0]-cx[1,0]+cmx[1,0]-c1[0,0]+cm1[0,0]-c1[0,1]+cm1[0,1])/E
    dq[1,1,0]=dq[1,0,1]=0.5*(cx[0,0]-cmx[0,0]+cx[1,0]-cmx[1,0]-c1[1,0]+cm1[1,0]-c1[1,1]+cm1[1,1])/E

    # y-rotated perturbation
    dq[0,1,2]=dq[0,2,1]=0.5*(np.sqrt(2)*(cy[0,1]-cmy[0,1])-(c1[0,1]-cm1[0,1])-(c1[0,2]-cm1[0,2]))/E
    dq[1,1,2]=dq[1,2,1]=0.5*(cy[1,1]-cmy[1,1]-cy[2,1]+cmy[2,1]-c1[1,1]+cm1[1,1]-c1[1,2]+cm1[1,2])/E
    dq[2,2,1]=dq[2,1,2]=0.5*(cy[1,1]-cmy[1,1]+cy[2,1]-cmy[2,1]-c1[2,1]+cm1[2,1]-c1[2,2]+cm1[2,2])/E

    # z-rotated perturbation
    dq[1,2,0]=dq[1,0,2]=0.5*(np.sqrt(2)*(cz[1,2]-cmz[1,2])-(c1[1,0]-cm1[1,0])-(c1[1,2]-cm1[1,2]))/E
    dq[2,2,0]=dq[2,0,2]=0.5*(cz[2,2]-cmz[2,2]-cz[0,2]+cmz[0,2]-c1[2,2]+cm1[2,2]-c1[2,0]+cm1[2,0])/E
    dq[0,0,2]=dq[0,2,0]=0.5*(cz[2,2]-cmz[2,2]+cz[0,2]-cmz[0,2]-c1[0,2]+cm1[0,2]-c1[0,0]+cm1[0,0])/E

    return dq


def build_charge_derivatives(charges, nat, E):
    """
    Assemble the per-atom BEC derivative tensor for every atom in the cell.

    Wraps charge_derivative() in a list comprehension over all nat atoms,
    reading the appropriate tensors from the loaded charges dict.

    Parameters
    ----------
    charges : dict  — output of load_bec_charges()
    nat     : int   — number of atoms
    E       : float — electric field magnitude (eV/Ang)

    Returns
    -------
    dq : list[ndarray(3,3,3)] — dq[t] is the derivative tensor for atom t
    """
    return [charge_derivative(
                charges['1'][t],  charges['m1'][t],
                charges['x'][t],  charges['y'][t],
                charges['z'][t],  charges['mx'][t],
                charges['my'][t], charges['mz'][t],
                E)
            for t in range(nat)]


# ===========================================================================
# V.  VASP phonon data loader
# ===========================================================================

def load_vasp_phonon_data(freqs_file='freqs_vasp.dat',
                          eigvecs_file='eigvecs_vasp.dat',
                          modes_max=None):
    """
    Load VASP DFPT phonon frequencies and eigenvectors.

    VASP-specific notes vs. the phonopy variant:
      - Frequencies are already in cm-1; no THz -> cm-1 conversion is applied.
      - Eigenvectors are already mass-normalised by VASP; compute_raman_activities()
        uses them directly without any /sqrt(mass) step.
      - All 3N modes are typically present (acoustic/imaginary filtering is
        done separately before export if desired).

    Parameters
    ----------
    freqs_file   : str — frequency file path (cm-1, one value per line)
    eigvecs_file : str — eigenvector matrix file path (3N x N_modes)
    modes_max    : int — 3*nat, used only for the retained-modes log line

    Returns
    -------
    eigvals : ndarray (N_modes,) — frequencies in cm-1 (unchanged from file)
    hws     : ndarray (N_modes,) — frequencies in meV
    eigvecs : ndarray (3N, N_modes)
    n_modes : int
    """
    freqs   = np.loadtxt(freqs_file)    # cm-1, already converted by VASP
    eigvecs = np.loadtxt(eigvecs_file)  # mass-normalised displacement
    n_modes = eigvecs.shape[1]

    assert len(freqs) == n_modes, (
        f"freqs has {len(freqs)} entries but eigvecs has {n_modes} columns — "
        "check that both files were exported from the same VASP run.")

    if modes_max is not None:
        print(f"Retained modes : {n_modes} / {modes_max}  "
              f"(removed {modes_max - n_modes} imaginary / acoustic)")

    eigvals = freqs           # cm-1 — no conversion
    hws     = freqs * CM1MEV  # cm-1 -> meV
    return eigvals, hws, eigvecs, n_modes


# ===========================================================================
# VI.  Raman activity calculator  (VASP: eigenvectors already mass-normalised)
# ===========================================================================

def compute_raman_activities(dq, eigvecs, nat, n_modes):
    """
    Compute the Raman scattering activity for each phonon mode.

    VASP eigenvectors are already mass-normalised — no /sqrt(mass) weighting
    is applied here (contrast with RASCBEC_phonopy.py where raw phonopy
    eigenvectors require explicit mass normalisation).

    Implements Eqs. 7-8 of Zhang et al. (2025).

    Eq. 7 — polarizability tensor contribution from atom t, mode s:
        R^{(t)}_{ij} = sum_k  (dZ*_{ik}/dE_j)_t * e_{tk}
    where e_{tk} is the VASP mass-normalised eigenvector component.

    The full polarizability derivative tensor for mode s is:
        R_{ij} = sum_t  R^{(t)}_{ij}  /  (4*pi*eps0)

    Eq. 8 — Raman scattering activity:
        A_s = 45*alpha^2 + 7*beta^2
    with
        alpha  = (R_xx + R_yy + R_zz) / 3
        beta^2 = [(R_xx-R_yy)^2 + (R_xx-R_zz)^2 + (R_yy-R_zz)^2
                   + 6(R_xy^2 + R_xz^2 + R_yz^2)] / 2

    The nested Python loops (modes x atoms x 3 x 3 x 3) from the original
    implementation are replaced by a single numpy einsum:

        ra_all[s, j, k] = sum_{t,i} dq[t, i, k, j] * e_mw[s, t, i]

    where e_mw[s, t, i] is the (already mass-normalised) eigenvector tensor
    of shape (n_modes, nat, 3).  This collapses the O(n_modes * nat * 27)
    pure-Python loop into a BLAS-backed tensor contraction.

    Parameters
    ----------
    dq      : list[ndarray(3,3,3)] — per-atom BEC derivative tensors,
              dq[t][i,k,j] = dZ*_{ij}/dE_k for atom t
    eigvecs : ndarray (3N, N_modes) — VASP mass-normalised eigenvectors
    nat     : int
    n_modes : int

    Returns
    -------
    activity : list[float] — Raman scattering activity per mode
    """
    dq_arr = np.array(dq)   # (nat, 3, 3, 3), indices: t, i, k, j

    # eigvecs: (3*nat, n_modes) -> (n_modes, nat, 3); already mass-normalised
    e_mw = eigvecs.T.reshape(n_modes, nat, 3)

    # ra_all[s, j, k] = sum_{t,i} dq[t,i,k,j] * e_mw[s,t,i]
    ra_all = np.einsum('tikj,sti->sjk', dq_arr, e_mw) / (4.0 * np.pi * EPS0)

    alpha = (ra_all[:, 0, 0] + ra_all[:, 1, 1] + ra_all[:, 2, 2]) / 3.0
    beta2 = ((ra_all[:, 0, 0] - ra_all[:, 1, 1])**2
           + (ra_all[:, 0, 0] - ra_all[:, 2, 2])**2
           + (ra_all[:, 1, 1] - ra_all[:, 2, 2])**2
           + 6.0 * (ra_all[:, 0, 1]**2
                  + ra_all[:, 0, 2]**2
                  + ra_all[:, 1, 2]**2)) / 2.0

    return list(45.0 * alpha**2 + 7.0 * beta2)


# ===========================================================================
# VII.  Output writer
# ===========================================================================

def write_csv(out_csv, eigvals, activity, formula='', dopants='', file_label='', E=None):
    """
    Write Raman frequencies and activities to a CSV with a metadata header.

    The header lines are parsed by compare_raman.py and plot_raman.py for
    automatic labelling and plot titling.  The formula and dopants lines
    fully describe the chemistry; no directory-tag Composition field is
    needed or written.

    File format
    -----------
    # Formula: <e.g. Na3PS4>
    # Dopants: <e.g. Ca=0.125 Cl=0.0625>      <- omitted if undoped
    # File_Label: <e.g. Ca01-Cl06>            <- omitted if undoped
    # E_field: <e.g. 0.02>
    # Mode,Freq_cm-1,Activity
    0001,<freq>,<activity>
    ...

    Parameters
    ----------
    out_csv    : str   -- output file path
    eigvals    : array -- frequencies in cm-1, length n_modes
    activity   : list  -- Raman activities, length n_modes
    formula    : str   -- base host formula, e.g. 'Na3PS4'
    dopants    : str   -- dopant descriptor, e.g. 'Ca=0.125 Cl=0.0625'
                          ('' or 'Undoped' when no dopants)
    file_label : str   -- filesystem-safe dopant label, e.g. 'Ca01-Cl06'
                          ('' when undoped)
    E          : float -- electric field magnitude used (written to header)
    """
    with open(out_csv, 'w') as fh:
        fh.write(f"# Formula: {formula}\n")
        if dopants and dopants != "Undoped":
            fh.write(f"# Dopants: {dopants}\n")
        if file_label:
            fh.write(f"# File_Label: {file_label}\n")
        if E is not None:
            fh.write(f"# E_field: {E:g}\n")
        fh.write("# Mode,Freq_cm-1,Activity\n")
        for i, (freq, act) in enumerate(zip(eigvals, activity), 1):
            fh.write(f"{i:04d},{freq:.6f},{act:.6f}\n")
    print(f"Written : {out_csv} ({len(eigvals)} modes)")


# ===========================================================================
# VIII. Chemistry metadata and naming
# (mirrors RASCBEC_phonopy.py: safe_name, split_dops, host_formula,
#  split_basis, validate_chemistry_roles, build_chemistry_metadata,
#  raman_filename)
# ===========================================================================

def safe_name(text):
    """Return a filename-safe label by replacing spaces & removing selected punctuation."""
    safe = text.replace(" ", "_").replace("/", "_")
    for char in "()=:":
        safe = safe.replace(char, "")
    return safe


def split_dops(species_names, species_n, threshold=DOPANT_THRESHOLD):
    """
    Infer dopant species automatically from low atomic count fraction.

    Any species whose atom count is below ``threshold`` of the total atom
    count is classified as a dopant; everything else is classified as host.
    This heuristic works well for dilute-substitution systems
    (e.g. Na3PS4:Ca/Cl) but should be bypassed via ``--dops`` for systems
    where dopant concentration is not dilute or role assignment is
    otherwise ambiguous (e.g. some LLTO compositions).

    Returns
    -------
    tuple
        (host_names, host_counts, dop_names, dop_counts)
    """
    total = float(sum(species_n))
    host_names, host_n, dop_names, dop_n = [], [], [], []
    for name, count in zip(species_names, species_n):
        if total > 0 and count / total < threshold:
            dop_names.append(name)
            dop_n.append(int(count))
        else:
            host_names.append(name)
            host_n.append(int(count))
    return host_names, host_n, dop_names, dop_n


def host_formula(species_names, species_n, ref=None):
    """
    Build a compact host-composition label with a consistent normalization.

    All displayed coefficients are divided by the reference count ``ref`` so
    that the reference is normalized to one. ``ref`` is the summed count of
    the resolved basis group (see :func:`build_chemistry_metadata`), so a group
    such as ``Ti+Ga`` normalizes to one even though the dopant member is not
    itself shown in the host formula. For example, a host cell
    ``Li60 La90 Ti150 O360`` with ``ref=150`` (basis Ti) renders as
    ``Li0.4La0.6TiO2.4``, and the same cell with ``ref=160`` (basis Ti+Ga)
    renders as ``Li0.38La0.56Ti0.94O2.25`` -- the sub-unity Ti coefficient
    correctly reflects that part of the Ti site is occupied by Ga.

    When ``ref`` is ``None`` the legacy behavior is retained: normalize to the
    least-abundant shown species and round to nearby integers.
    This produces a readable analysis label rather than a general-purpose
    crystallographic formula reduction.
    """
    if not species_names or not species_n:
        raise ValueError("Composition species could not be determined.")
    count_map = {name: int(count) for name, count in zip(species_names, species_n)}

    if ref is None:
        # Legacy: normalize to the least-abundant shown species and round.
        base_ref = float(min(species_n))
        coefficients = [float(max(1, round(count / base_ref))) for count in species_n]
    else:
        base_ref = max(float(ref), 1.0)
        coefficients = [count_map[name] / base_ref for name in species_names]

    parts = []
    for name, coefficient in zip(species_names, coefficients):
        if np.isclose(coefficient, round(coefficient), atol=1e-8):
            integer = int(round(coefficient))
            parts.append(name if integer == 1 else f"{name}{integer}")
        else:
            parts.append(f"{name}{coefficient:.2f}".rstrip("0").rstrip("."))
    return "".join(parts)


def split_basis(bas_species):
    """
    Split a ``--bas`` value into an ordered list of basis species.

    A ``+`` joined group (e.g. ``Ti+Ga``) becomes ``['Ti', 'Ga']``; a bare
    species (e.g. ``Ti``) becomes ``['Ti']``. Returns ``None`` when no basis
    was supplied, so the caller can fall back to the legacy reference.

    Parameters
    ----------
    bas_species : str or None -- raw CLI value (e.g. 'Ti' or 'Ti+Ga')

    Returns
    -------
    list of str or None -- ordered basis species, or None when unspecified.
    """
    if bas_species is None:
        return None
    members = [sp.strip() for sp in bas_species.split('+')]
    return members


def validate_chemistry_roles(species_names, bas_species, dop_species):
    """
    Validate chemistry-role overrides before they are consumed downstream.

    This is the single validation guard through which every explicit CLI
    role override (--bas / --dops) must pass. It exists to catch
    misconfiguration in non-sulfide chemistries (e.g. LLTO) where role
    assignment is not automatic, before any downstream computation,
    filename, or plot silently uses a wrong or inconsistent assignment.

    The basis (``--bas``) may be a single species or a ``+`` joined group
    (e.g. ``Ti+Ga``). Every basis member -- host or dopant alike -- is
    allowed, since a doped formula can legitimately be normalized to a
    combined site population (e.g. ``Ti + Ga = 1``).

    Checks performed
    -----------------
    - Every ``--bas`` member must be a non-empty species present in
      ``species_names`` (when provided).
    - ``--bas`` must not contain duplicate members.
    - Every ``--dops`` entry must be a non-empty string with no duplicates.

    Raises
    ------
    ValueError
        If any of the checks above fail, with a message identifying which
        role(s) are inconsistent.
    """
    species_set = set(species_names)
    dop_species = list(dop_species) if dop_species else []

    if bas_species is not None:
        if not bas_species.strip():
            raise ValueError("--bas cannot be an empty string.")
        members = [sp.strip() for sp in bas_species.split('+')]
        if any(not sp for sp in members):
            raise ValueError(
                "--bas contains an empty species (check for a stray '+').")
        missing = sorted(sp for sp in set(members) if sp not in species_set)
        if missing:
            raise ValueError(
                f"Basis species {missing} not present in the structure "
                f"(available species: {sorted(species_set)}).")
        duplicates = sorted({sp for sp in set(members) if members.count(sp) > 1})
        if duplicates:
            raise ValueError(f"--bas contains duplicate species: {duplicates}.")

    if dop_species:
        if any(not sp.strip() for sp in dop_species):
            raise ValueError("--dops entries cannot be empty strings.")
        duplicates = sorted({sp for sp in dop_species if dop_species.count(sp) > 1})
        if duplicates:
            raise ValueError(f"--dops contains duplicate species: {sorted(duplicates)}.")


def build_chemistry_metadata(species_names, species_n,
                             bas_species=None, dop_species=None):
    """
    Resolve host, dopant & basis metadata for a structure.

    This function converts ordered species labels & counts into the
    normalized chemistry-role dictionary used for naming, file labels,
    plot titles, & the metadata stored in CSV headers.

    Role resolution follows a strict precedence order. Explicit CLI
    overrides are validated first and, when present, take priority. If
    dopants are not supplied explicitly, they are inferred with
    ``split_dops`` from low atomic fraction. When ``bas_species`` is
    supplied, it defines both the dopant x-value denominator and a
    host-formula coefficient of one. When it is omitted, the
    least-abundant host species supplies the legacy reference.

    Parameters
    ----------
    species_names : list of str
        Ordered species labels & per-species atom counts parsed from POSCAR.
    species_n : ndarray
        Atom count per species.
    bas_species : str or None, optional
        Species (or ``+`` joined group) used as the concentration reference
        for dopant x-values. A group's member counts are summed into a single
        reference, e.g. ``Ti+Ga`` normalizes ``Ti + Ga`` to one.
    dop_species : list of str or None, optional
        Explicit dopant list. When provided, it fully overrides automatic
        dopant inference.

    Returns
    -------
    dict with keys
        formula      : str -- e.g. 'Na3PS4'
        dopants      : str -- e.g. 'Ca=0.125 Cl=0.0625' ('' if none)
        file_label   : str -- dopant filename part, e.g. 'Ca01-Cl06'
                              (empty string if undoped)
        host_species : list of str
        dop_species  : list of str
        bas_species  : str or None -- resolved basis species

    Raises
    ------
    ValueError
        If any explicit role override is invalid or internally inconsistent.
    """
    validate_chemistry_roles(species_names, bas_species, dop_species)

    species_count_map = {name: int(count) for name, count in zip(species_names, species_n)}
    explicit_dops = list(dop_species) if dop_species else None

    if explicit_dops is not None:
        dop_names = list(explicit_dops)
        dop_counts = [int(species_count_map.get(sp, 0)) for sp in dop_names]
        host_names = [sp for sp in species_names if sp not in set(dop_names)]
        host_counts = [species_count_map[sp] for sp in host_names]
    else:
        host_names, host_counts, dop_names, dop_counts = split_dops(
            species_names, species_n)

    # --- Resolve the basis reference (count + label) ---
    # ``--bas`` may be a single species or a ``+`` joined group (e.g. Ti+Ga).
    # The reference count is the SUM of the group's member counts, so the
    # group is normalized to one and may include dopant species (doped LLTO:
    # normalize so Ti + Ga = 1). When omitted, fall back to the legacy
    # least-abundant host species.
    bas_members = split_basis(bas_species)
    if bas_members is not None:
        bas_count = sum(int(species_count_map.get(sp, 0)) for sp in bas_members)
        bas_count = max(bas_count, 1)
        resolved_bas_species = "+".join(bas_members)
    elif host_names:
        resolved_bas_species = host_names[int(np.argmin(host_counts))]
        bas_count = max(int(species_count_map[resolved_bas_species]), 1)
    elif len(species_names):
        resolved_bas_species = species_names[int(np.argmin(species_n))]
        bas_count = max(int(species_count_map[resolved_bas_species]), 1)
    else:
        resolved_bas_species = None
        bas_count = 1

    # Preserve the exact CLI order: --dops Ca Cl
    label_dops = list(explicit_dops) if explicit_dops is not None else list(dop_names)

    # --- Build the formula label ---
    # The stored formula is the BASE (undoped) host composition, not the raw
    # doped host-only count.  The supercell only contains the host atoms that
    # were NOT displaced, so the base is recovered by adding back the atoms
    # each dopant took (or added) at every host element:
    #     base[el] = doped_host[el] + Σ_sp DOPANT_SUBSTITUTIONS[sp][el] · n_sp
    # (e.g. a Ca that displaces 2 Na contributes +2 Na back).  The base counts
    # are then normalized to the SAME basis used for the dopant x-values, so a
    # single reference applies everywhere. For a group basis (e.g. Ti+Ga) that
    # means every host coefficient is divided by (n_Ti + n_Ga).
    # The dopant species themselves stay out of the host formula and are shown
    # via the dopant metadata / plot title instead.
    if host_names:
        base_counts = list(host_counts)
        host_el_index = {el: i for i, el in enumerate(host_names)}
        for sp in label_dops:
            n_sp = int(species_count_map.get(sp, 0))
            if n_sp == 0:
                continue
            for host_el, coeff in DOPANT_SUBSTITUTIONS.get(sp, {}).items():
                j = host_el_index.get(host_el)
                if j is not None:
                    base_counts[j] += int(coeff * n_sp)
        formula = host_formula(host_names, base_counts, ref=bas_count)
    else:
        formula = "Unknown"

    fractions = {
        name: float(species_count_map.get(name, 0)) / float(bas_count)
        for name in label_dops
    }

    if label_dops:
        dopants = (
            "Undoped"
            if not fractions or all(abs(x) < 1e-12 for x in fractions.values())
            else " ".join(f"{name}={x:.4f}" for name, x in fractions.items())
        )
        # Always include every declared dopant, including zeros.
        file_label = "-".join(
            f"{name}{int(round(x * 100.0)):02d}"
            for name, x in fractions.items()
        )
    else:
        dopants = ""
        file_label = ""

    return {
        "formula": formula,
        "dopants": dopants,
        "file_label": safe_name(file_label),
        "host_species": host_names,
        "dop_species": dop_names,
        "bas_species": resolved_bas_species,
    }


def raman_filename(chem, E):
    """
    Build the chemistry-based Raman output stem (no extension).

    Uses the host formula & the dopant-only filename part from
    :func:`build_chemistry_metadata` (e.g. ``Ca01-Cl06``).

    Parameters
    ----------
    chem : dict  -- output of :func:`build_chemistry_metadata`
    E    : float -- electric field magnitude (eV/Å)

    Returns
    -------
    str -- e.g. 'Na3PS4_Ca01-Cl06_E0.02' or 'Na3PS4_E0.02'
    """
    stem = f"{chem['formula']}"
    if chem["file_label"]:
        stem += f"_{chem['file_label']}"
    return f"{stem}_E{E:g}"


# ===========================================================================
# IX.  Main orchestrator
# ===========================================================================

def main():
    args = parse_args()

    # --- 1. Structure ---
    print("\n" + "=" * 70)
    vol, species_n, atomic_mass, ntype, nat, modes, species_names = \
        structure_info('POSCAR')

    # --- 2. Resolve electric field magnitude ---
    if args.E is not None:
        E = args.E
        print(f"\nE-field : {E} eV/Å (user-supplied)")
    else:
        E = read_efield_pead(OUTCAR_MAP['1'])
        print(f"\nE-field : {E} eV/Å (auto-read from {OUTCAR_MAP['1']})")

    # --- 3. Derive naming and metadata from POSCAR composition ---
    chem = build_chemistry_metadata(
        species_names, species_n,
        bas_species=args.bas, dop_species=args.dops)
    formula   = chem["formula"]
    dopants   = chem["dopants"]
    base_name = raman_filename(chem, E)
    out_csv = args.out_csv or f"{base_name}.csv"
    out_png = args.out_png or f"{base_name}.png"
    print(f"Formula : {formula}")
    if dopants:
        print(f"Dopants : {dopants}")
    print(f"Output  : {out_csv}")
    print("=" * 70)

    # --- 4. Born Effective Charges ---
    print("\nLoading BEC tensors ...")
    charges = load_bec_charges(OUTCAR_MAP, nat, jobs=args.jobs)

    # --- 5. BEC derivatives ---
    dq = build_charge_derivatives(charges, nat, E)

    # --- 6. Phonon data from VASP DFPT output ---
    print("\nLoading VASP phonon data ...")
    eigvals, hws, eigvecs, n_modes = load_vasp_phonon_data(
        'freqs_vasp.dat', 'eigvecs_vasp.dat', modes_max=modes)

    if eigvecs.shape[0] != 3 * nat:
        raise ValueError(
            f"Eigenvector rows ({eigvecs.shape[0]}) != 3*nat ({3*nat}). "
            "Check that the .dat files and POSCAR describe the same structure.")

    # --- 7. Raman activities ---
    print("\nComputing Raman activities ...")
    activity = compute_raman_activities(dq, eigvecs, nat, n_modes)

    # --- 8. Write CSV ---
    write_csv(out_csv, eigvals, activity,
              formula=formula, dopants=dopants,
              file_label=chem["file_label"], E=E)

    # --- 9. Plot ---
    if not args.no_plot:
        title = build_raman_title(chem, E, args.gamma * THZCM1, args.temp)
        plot_raman_spectrum(
            dat_file = out_csv,
            out_png  = out_png,
            gamma    = args.gamma * THZCM1,
            freq_min = args.freq_min,
            freq_max = args.freq_max,
            sticks   = not args.no_sticks,
            n_labels = args.n_labels,
            title    = title,
            temp     = args.temp,
            laser_wl = args.laser_wl,
        )
        print(f"Written : {out_png}")

    print("\nDone.")


if __name__ == '__main__':
    main()
