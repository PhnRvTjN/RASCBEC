#!/usr/bin/env python3

"""
Title: Calculation of Raman Activities from VASP DFPT Output Files

Authors: Rui Zhang (original);
         Phani Ravi Teja Nunna (refactor and enhancements)

Patches: auto atomic-mass lookup via pymatgen,
         variable mode count (filtered imaginary modes),
         subdirectory-based OUTCAR paths (./1/OUTCAR, ./m1/OUTCAR, ...),
         chemistry-based output filenames derived from POSCAR composition,
         CSV metadata header (formula, dopants, E-field),
         auto-read EFIELD_PEAD from OUTCAR,
         plot output option,
         auto-generated informative plot title.

Date: 2025-04-28 (original); 2026-02-23 (patched); 2026-03-04 (chemistry naming)

License: GNU GENERAL PUBLIC LICENSE

Cite: Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation
      via born effective charge." Computer Physics Communications 307 (2025): 109425

Description:
    Reads VASP DFPT output to calculate Raman activities per phonon mode
    using the RASCBEC method.

    Key distinctions from RASCBEC_phonopy.py:
      - VASP outputs frequencies already in cm-1  (no THz conversion needed)
      - VASP eigenvectors are already mass-normalised  (no /sqrt(mass) step)

    E-field magnitude is read automatically from ./1/OUTCAR (EFIELD_PEAD tag).
    Override with --E only if you want to force a specific value.

Required files (in working directory):
    POSCAR              — structure file
    ./1/OUTCAR          — BEC, E-field along unrotated +
    ./m1/OUTCAR         — BEC, E-field along unrotated -
    ./x/OUTCAR          — BEC, E-field along +x (rotated along x)
    ./mx/OUTCAR         — BEC, E-field along -x
    ./y/OUTCAR          — BEC, E-field along +y (rotated along y)
    ./my/OUTCAR         — BEC, E-field along -y
    ./z/OUTCAR          — BEC, E-field along +z (rotated along z)
    ./mz/OUTCAR         — BEC, E-field along -z
    freqs_vasp.dat      — phonon frequencies, shape (3N,), unit cm-1
    eigvecs_vasp.dat    — mass-normalised eigenvectors, shape (3N x 3N)

Output naming (chemistry-based, derived from POSCAR; no directory tags):
    raman_<formula>_<dopants>_E<field>.csv / .png

    Examples:
      undoped Na3PS4, E=0.02   ->  raman_Na3PS4_E0.02.csv
      Ca+Cl co-doped Na3PS4    ->  raman_Na3PS4_Ca0.125_Cl0.5_E0.02.csv

    Dopant fractions are computed from actual supercell counts as
    x = n_dopant / n_ref, where n_ref is the count of the least-abundant
    host species (stoichiometric coefficient = 1 in the ideal formula).

CSV metadata header (parsed by compare_raman.py):
    # Formula: Na3PS4
    # Dopants: Ca(x=0.125)/Cl(x=0.5)   <- omitted if undoped
    # E_field: 0.02
    # Mode,Freq_cm-1,Activity

Dependencies: argparse, pathlib, re, numpy, pymatgen, plot_raman

Usage:
    python RASCBEC_VASP.py                      # all defaults
    python RASCBEC_VASP.py --E 0.02 --gamma 15
    python RASCBEC_VASP.py --no-plot
    python RASCBEC_VASP.py --freq-min 50 --freq-max 600
    python RASCBEC_VASP.py --help
"""

import argparse
import re
from pathlib import Path

import numpy as np
from pymatgen.core import Element

from plot_raman import plot_raman_spectrum


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# VASP frequencies are in cm-1 — no THz conversion needed.
# cm-1 -> meV: ratio of CODATA THz conversion factors.
CM1MEV = 4.13566553853599 / 33.3564095198  # 0.12398419843 meV per cm-1

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
                   help='EFIELD_PEAD magnitude in eV/Ang.  If omitted, '
                        'auto-read from ./1/OUTCAR (the unscaled reference calculation).')
    p.add_argument('--gamma', type=float, default=10.0,
                   help='Lorentzian FWHM in cm-1 for broadening (default: 10.0)')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Plot lower x-axis limit in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Plot upper x-axis limit in cm-1 (default: auto)')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip plot generation; write CSV only')
    p.add_argument('--no-sticks', action='store_true',
                   help='Omit stick spectrum from plot')
    p.add_argument('--n-labels', type=int, default=20,
                   help='Number of peak labels to annotate on plot (default: 20)')
    p.add_argument('--out-csv', default=None,
                   help='Output CSV path (default: raman_<chem>_E<field>.csv)')
    p.add_argument('--out-png', default=None,
                   help='Output PNG path (default: raman_<chem>_E<field>.png)')
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

def get_charges_from_OUTCAR(outcar, nat):
    """
    Extract the Born Effective Charge (BEC) tensor from one open OUTCAR.

    Scans the file for the section header
    "BORN EFFECTIVE CHARGES (including local field effects)"
    and reads the 3x3 tensor for each atom.  Only the last occurrence
    is returned (VASP may write multiple BEC blocks during SCF).

    Parameters
    ----------
    outcar : file object — open OUTCAR, iterable line-by-line
    nat    : int         — number of atoms in the cell

    Returns
    -------
    charges : list[list[list[float]]] — shape (nat, 3, 3)
              charges[i][alpha][beta] = Z*_{i,alpha*beta}
    """
    charges = [[0.0] * 3 for _ in range(nat)]
    for line in outcar:
        if "BORN EFFECTIVE CHARGES (including local field effects)" in line:
            outcar.readline()  # blank separator line
            for i in range(nat):
                outcar.readline()  # "ion N" header line
                charges[i] = [list(map(float, outcar.readline().split()[1:4]))
                               for _ in range(3)]
    return charges


def load_bec_charges(outcar_map, nat):
    """
    Read BEC tensors from all 8 finite-difference OUTCAR subdirectories.

    Each subdirectory corresponds to one Cartesian E-field direction.
    The resulting dict is keyed by the same labels used in OUTCAR_MAP
    and later consumed by build_charge_derivatives().

    Parameters
    ----------
    outcar_map : dict — {label: path} as in OUTCAR_MAP
    nat        : int  — number of atoms

    Returns
    -------
    charges : dict[str, list[nat x 3 x 3]] — BEC tensors keyed by direction label
    """
    charges = {}
    for key, path in outcar_map.items():
        print(f"  Reading BEC: {path}")
        with open(path, 'r') as fh:
            charges[key] = get_charges_from_OUTCAR(fh, nat)
    return charges


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

    Parameters
    ----------
    dq      : list[ndarray(3,3,3)] — per-atom BEC derivative tensors
    eigvecs : ndarray (3N, N_modes) — VASP mass-normalised eigenvectors
    nat     : int
    n_modes : int

    Returns
    -------
    activity : list[float] — Raman scattering activity per mode
    """
    activity = [0.0] * n_modes

    for s in range(n_modes):
        eigvec = eigvecs[:, s].reshape((nat, 3))  # already mass-normalised

        ra_tot = np.zeros((3, 3))
        for t in range(nat):
            dqt     = dq[t]
            eigvect = eigvec[t, :]
            act     = np.zeros((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        act[i, j, k] = dqt[i, k, j] * eigvect[i]
            ra_tot += act[0] + act[1] + act[2]

        ra    = ra_tot / (4.0 * np.pi * EPS0)
        alpha = (ra[0, 0] + ra[1, 1] + ra[2, 2]) / 3.0
        beta2 = ((ra[0, 0] - ra[1, 1])**2
               + (ra[0, 0] - ra[2, 2])**2
               + (ra[1, 1] - ra[2, 2])**2
               + 6.0 * (ra[0, 1]**2 + ra[0, 2]**2 + ra[1, 2]**2)) / 2.0
        activity[s] = 45.0 * alpha**2 + 7.0 * beta2

    return activity


# ===========================================================================
# VII.  Output writer
# ===========================================================================

def write_csv(out_csv, eigvals, activity, formula='', dopants='', E=None):
    """
    Write Raman frequencies and activities to a CSV with a metadata header.

    The header lines are parsed by compare_raman.py for automatic labelling
    and plot titling.  The formula and dopants lines fully describe the
    chemistry; no directory-tag Composition field is needed or written.

    File format
    -----------
    # Formula: <e.g. Na3PS4>
    # Dopants: <e.g. Ca(x=0.125)/Cl(x=0.5)>   <- omitted if undoped
    # E_field: <e.g. 0.02>
    # Mode,Freq_cm-1,Activity
    0001,<freq>,<activity>
    ...

    Parameters
    ----------
    out_csv  : str   — output file path
    eigvals  : array — frequencies in cm-1, length n_modes
    activity : list  — Raman activities, length n_modes
    formula  : str   — ideal host formula, e.g. 'Na3PS4'
    dopants  : str   — dopant descriptor, e.g. 'Ca(x=0.125)/Cl(x=0.5)'
    E        : float — electric field magnitude used (written to header)
    """
    with open(out_csv, 'w') as fh:
        fh.write(f"# Formula: {formula}\n")
        if dopants:
            fh.write(f"# Dopants: {dopants}\n")
        if E is not None:
            fh.write(f"# E_field: {E:g}\n")
        fh.write("# Mode,Freq_cm-1,Activity\n")
        for i, (freq, act) in enumerate(zip(eigvals, activity), 1):
            fh.write(f"{i:04d},{freq:.6f},{act:.6f}\n")
    print(f"Written : {out_csv}  ({len(eigvals)} modes)")


# ===========================================================================
# VIII.  Chemistry metadata and naming
# ===========================================================================

def identify_dopants(species_names, species_n, threshold=DOPANT_THRESHOLD):
    """
    Classify each species as host or dopant based on its supercell abundance.

    Species whose fraction of total atoms falls below *threshold* are treated
    as dopants.  The default (0.05) correctly identifies substituents present
    at a few percent or less while retaining all framework species.

    Example: in Na23P8S28Ca1Cl4 (64 atoms total):
        Ca -> 1/64 = 0.016  < 0.05  -> dopant  ok
        Cl -> 4/64 = 0.063  > 0.05  -> host    (increase threshold if needed)

    Parameters
    ----------
    species_names : list[str]
    species_n     : ndarray of ints
    threshold     : float — fractional abundance cutoff

    Returns
    -------
    host_names   : list[str]
    host_n       : list[int]
    dopant_names : list[str]
    dopant_n     : list[int]
    """
    total = float(sum(species_n))
    host_names, host_n, dopant_names, dopant_n = [], [], [], []
    for name, n in zip(species_names, species_n):
        if n / total < threshold:
            dopant_names.append(name)
            dopant_n.append(int(n))
        else:
            host_names.append(name)
            host_n.append(int(n))
    return host_names, host_n, dopant_names, dopant_n


def approximate_host_formula(host_names, host_n):
    """
    Recover the ideal stoichiometric formula from doping-perturbed host counts.

    Strategy: divide all host counts by the smallest host count and round to
    the nearest integer.  This anchors the normalisation to the species with
    stoichiometric coefficient 1 in the ideal formula (e.g. P in Na3PS4),
    which is almost always the least substituted and thus the most reliable
    reference.

    Examples
    --------
    Na23 P8 S28  (one Ca + four Cl in Na3PS4 supercell) -> Na3PS4  ok
    Na24 P8 S32  (undoped Na3PS4 x8 supercell)          -> Na3PS4  ok
    Ba8  Ti8 O24 (undoped BaTiO3 x8 supercell)           -> BaTiO3  ok

    Limitation: breaks down when the reference species itself is heavily
    substituted (> ~25%).  Inspect the printed formula and override with
    --out-csv if needed.

    Parameters
    ----------
    host_names : list[str]
    host_n     : list[int]

    Returns
    -------
    formula : str — e.g. 'Na3PS4'
    """
    min_n  = min(host_n)
    ratios = [max(1, round(n / min_n)) for n in host_n]
    return ''.join(f"{name}{r if r > 1 else ''}" for name, r in zip(host_names, ratios))


def build_chemistry_metadata(species_names, species_n):
    """
    Derive all chemistry-based labels from the supercell composition alone.

    Dopant fractions are computed as:
        x_dopant = n_dopant / n_ref
    where n_ref = count of the least-abundant host species — the one with
    stoichiometric coefficient 1 in the ideal formula.  For Na3PS4 this is P,
    so x_Ca = n_Ca / n_P gives substitutions per formula unit.

    Three strings are returned to serve different purposes:
      formula     — human-readable ideal formula for CSV headers and plot titles
      dopants_str — parenthetical dopant descriptor for CSV headers and titles
      file_label  — filesystem-safe label used to build output filenames

    Parameters
    ----------
    species_names : list[str]
    species_n     : ndarray

    Returns
    -------
    formula     : str — e.g. 'Na3PS4'
    dopants_str : str — e.g. 'Ca(x=0.125)/Cl(x=0.5)'  ('' if undoped)
    file_label  : str — e.g. 'Na3PS4_Ca0.125_Cl0.5'   (no parentheses)
    """
    host_names, host_n, dopant_names, dopant_n = identify_dopants(species_names, species_n)
    formula   = approximate_host_formula(host_names, host_n)
    ref_count = min(host_n)

    if dopant_names:
        fracs       = {name: n / ref_count for name, n in zip(dopant_names, dopant_n)}
        dopants_str = '/'.join(f"{name}(x={x:.3g})" for name, x in fracs.items())
        file_label  = '_'.join([formula] + [f"{name}{x:.3g}" for name, x in fracs.items()])
    else:
        dopants_str = ''
        file_label  = formula

    return formula, dopants_str, file_label


def build_plot_title(formula, dopants_str, E, gamma):
    """
    Build an informative plot title from chemistry strings and run parameters.

    Parameters
    ----------
    formula     : str   — host formula
    dopants_str : str   — dopant descriptor (empty string if undoped)
    E           : float — electric field magnitude (eV/Ang)
    gamma       : float — Lorentzian FWHM (cm-1)

    Returns
    -------
    title : str
    """
    chem_part = f"{formula} [{dopants_str} doped]" if dopants_str else formula
    return f"{chem_part} | E = {E} eV/Ang | FWHM = {gamma} cm$^{{-1}}$"


# ===========================================================================
# IX.  Main orchestrator
# ===========================================================================

def main():
    args = parse_args()

    # --- 1. Structure ---
    print("\n" + "=" * 70)
    vol, species_n, atomic_mass, ntype, nat, modes, species_names = structure_info('POSCAR')

    # --- 2. Resolve electric field magnitude ---
    if args.E is not None:
        E = args.E
        print(f"\nE-field  : {E} eV/Ang  (user-supplied)")
    else:
        E = read_efield_pead(OUTCAR_MAP['1'])
        print(f"\nE-field  : {E} eV/Ang  (auto-read from {OUTCAR_MAP['1']})")

    # --- 3. Derive all naming and metadata from POSCAR composition ---
    formula, dopants_str, file_label = build_chemistry_metadata(species_names, species_n)
    out_csv = args.out_csv or f"raman_{file_label}_E{E:g}.csv"
    out_png = args.out_png or f"raman_{file_label}_E{E:g}.png"
    print(f"Formula  : {formula}")
    if dopants_str:
        print(f"Dopants  : {dopants_str}")
    print(f"Output   : {out_csv}")
    print("=" * 70)

    # --- 4. Born Effective Charges ---
    print("\nLoading BEC tensors ...")
    charges = load_bec_charges(OUTCAR_MAP, nat)

    # --- 5. BEC derivatives ---
    dq = build_charge_derivatives(charges, nat, E)

    # --- 6. Phonon data ---
    print("\nLoading VASP phonon data ...")
    eigvals, hws, eigvecs, n_modes = load_vasp_phonon_data(
        'freqs_vasp.dat', 'eigvecs_vasp.dat', modes_max=modes)

    # --- 7. Raman activities ---
    print("\nComputing Raman activities ...")
    activity = compute_raman_activities(dq, eigvecs, nat, n_modes)

    # --- 8. Write CSV ---
    write_csv(out_csv, eigvals, activity,
              formula=formula, dopants=dopants_str, E=E)

    # --- 9. Plot ---
    if not args.no_plot:
        title = build_plot_title(formula, dopants_str, E, args.gamma)
        plot_raman_spectrum(
            dat_file = out_csv,
            out_png  = out_png,
            gamma    = args.gamma,
            freq_min = args.freq_min,
            freq_max = args.freq_max,
            sticks   = not args.no_sticks,
            n_labels = args.n_labels,
            title    = title,
        )
        print(f"Written  : {out_png}")

    print("\nDone.")


if __name__ == '__main__':
    main()
