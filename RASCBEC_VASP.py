#!/usr/bin/env python3

"""
Title: Calculation of Raman Activities from VASP DFPT Output Files

Authors: Rui Zhang (original);
         Phani Ravi Teja Nunna (refactor and enhancements)

Patches: auto atomic-mass lookup,
         variable mode count (filtered imaginary modes),
         subdirectory-based OUTCAR paths (./1/OUTCAR, ./m1/OUTCAR, ...),
         CSV output named raman_<parent>-<E>.csv, modular structure,
         CSV metadata header.
         added plot outut option,
         and auto-generated informative plot title.

Date: 2025-04-28 (original);
      2026-02-23 (patched version)

License: GNU GENERAL PUBLIC LICENSE

Cite: Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation
      via born effective charge." Computer Physics Communications 307 (2025): 109425

Description:
    Reads VASP DFPT output to calculate Raman activities per phonon mode
    using the RASCBEC method.

    Key distinction from RASCBEC_phonopy.py:
      - VASP outputs frequencies already in cm-1  (no THz conversion)
      - VASP eigenvectors are already mass-normalised (no /sqrt(mass) step)

    Required files (in working directory):
      POSCAR              — structure
      ./1/OUTCAR          — BEC, E-field along + (unrotated)
      ./m1/OUTCAR         — BEC, E-field along - (unrotated)
      ./x/OUTCAR          — BEC, E-field along +x (rotated along x-axis)
      ./mx/OUTCAR         — BEC, E-field along -x (rotated along x-axis)
      ./y/OUTCAR          — BEC, E-field along +y (rotated along y-axis)
      ./my/OUTCAR         — BEC, E-field along -y (rotated along y-axis)
      ./z/OUTCAR          — BEC, E-field along +z (rotated along z-axis)
      ./mz/OUTCAR         — BEC, E-field along -z (rotated along z-axis)
      freqs_vasp.dat      — phonon frequencies, shape (3N,), unit cm-1
      eigvecs_vasp.dat    — mass-normalised eigenvectors, shape (3N × 3N)

Output naming:
    raman_<parent>_<E>.csv / raman_<parent>_<E>.png where <parent> is the directory
    one level above cwd and <E> is the EFIELD_PEAD value.
    Example: running from ~/calcs/06-12/9-RASCBEC with E=0.02
             → raman_06-12_0.02.csv / raman_06-12_0.02.png
CSV output (raman_<parent>_<E>.csv) includes metadata comment lines:
    # Formula: Na3PS4
    # Dopants: Ca(x=0.06)/Cl(x=0.12)
    # E_field: 0.02
    # Composition: 06-12

Title convention for <parent> = "06-12":
    All-digit tokens split by '-' or '_' are interpreted as doping levels × 100.
    So "06-12" → dopant 1 at x = 0.06, dopant 2 at x = 0.12.
    Dopants are auto-identified as species whose fractional count in the
    supercell is below DOPANT_THRESHOLD.
    The host formula is recovered by normalising counts to the smallest
    host species (e.g. Na23P8S28 → Na3PS4).

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
# cm-1 → meV: use high-precision ratio of CODATA THz conversion factors.
CM1MEV  = 4.13566553853599 / 33.3564095198  # 0.12398419843 meV per cm-1

EPS0    = 55.2635e-4        # ε₀ in e² / (eV·Å)
KB_T    = 8.617333e-2 * 298 # k_B T at 298 K, meV

# Species whose supercell fractional count falls below this threshold
# are classified as dopants (e.g. 1 Ca in 108 atoms ≈ 0.009 < 0.05)
DOPANT_THRESHOLD = 0.05

# Subdirectory layout for the 8 finite-difference OUTCAR files
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
    p.add_argument('--E', type=float, default=0.02,
                   help='EFIELD_PEAD strength in eV/Ang (default: 0.02)')
    p.add_argument('--gamma', type=float, default=10.0,
                   help='Lorentzian FWHM in cm-1 for broadening (default: 10.0)')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Plot lower x-axis limit in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Plot upper x-axis limit in cm-1 (default: auto)')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip plot generation')
    p.add_argument('--no-sticks', action='store_true',
                   help='Hide stick spectrum in plot')
    p.add_argument('--n-labels', type=int, default=20,
                   help='Number of peak labels to annotate (default: 20)')
    p.add_argument('--out-csv', default=None,
                   help='Output CSV filename (default: raman_<parent>.csv)')
    p.add_argument('--out-png', default=None,
                   help='Output PNG filename (default: raman_<parent>.png)')
    return p.parse_args()


# ===========================================================================
# II.  Structure reader
# ===========================================================================

def get_mass(symbol):
    """Return the standard atomic mass (amu) for a given element symbol via pymatgen."""
    return float(Element(symbol).atomic_mass)


def structure_info(poscar='POSCAR'):
    """
    Parse a POSCAR and return structural metadata.

    Returns
    -------
    vol          : float   — unit-cell volume (Å³)
    species_n    : ndarray — atom counts per species, shape (ntype,)
    atomic_mass  : list    — atomic masses (amu) per species (for reference/title)
    ntype        : int     — number of species
    nat          : int     — total number of atoms
    modes        : int     — 3*nat (total phonon mode count)
    species_names: list    — element symbol strings
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
    print(f"Masses  : {[round(m, 4) for m in atomic_mass]} (amu, via pymatgen)")
    print(f"Atoms   : {nat} | Modes: {modes}")
    return vol, species_n, atomic_mass, ntype, nat, modes, species_names


# ===========================================================================
# III.  Born Effective Charge readers
# ===========================================================================

def get_charges_from_OUTCAR(outcar, nat):
    """
    Extract the Born Effective Charge tensor from a single open OUTCAR file.

    Parameters
    ----------
    outcar : file object  — open OUTCAR (text mode, iterable by line)
    nat    : int          — number of atoms in the cell

    Returns
    -------
    charges : list[list[list[float]]]  — shape (nat, 3, 3)
              charges[i][α][β] = Z*_{i,αβ}
    """
    charges = [[0.0] * 3 for _ in range(nat)]
    for line in outcar:
        if "BORN EFFECTIVE CHARGES (including local field effects)" in line:
            outcar.readline()                       # skip separator line
            for i in range(nat):
                outcar.readline()                   # skip "ion N" header
                charges[i] = [list(map(float, outcar.readline().split()[1:4]))
                               for _ in range(3)]
    return charges


def load_bec_charges(outcar_map, nat):
    """
    Read BEC tensors from all 8 finite-difference OUTCAR subdirectories.

    Parameters
    ----------
    outcar_map : dict  — {key: path} mapping (see OUTCAR_MAP)
    nat        : int   — number of atoms

    Returns
    -------
    charges : dict  — {key: list[nat × 3 × 3 BEC tensor]}
    """
    charges = {}
    for key, path in outcar_map.items():
        print(f"  Reading BEC: {path}")
        with open(path, 'r') as fh:
            charges[key] = get_charges_from_OUTCAR(fh, nat)
    return charges


# ===========================================================================
# IV.  BEC derivative tensor (single atom)
# ===========================================================================

def charge_derivative(charge1, chargem1, chargex, chargey, chargez,
                      chargemx, chargemy, chargemz, E):
    """
    Compute the 3rd-rank BEC derivative tensor dZ*/dE for one atom using
    finite differences of the 8 perturbed BEC tensors.

    Parameters
    ----------
    charge{1,m1,x,y,z,mx,my,mz} : list[3×3]
        BEC tensor for the atom under each E-field perturbation.
    E : float  — electric field magnitude (eV/Å)

    Returns
    -------
    dq : ndarray, shape (3, 3, 3)
         dq[i, j, k] = ∂Z*_{ij} / ∂E_k
    """
    c1, cm1, cx, cy, cz, cmx, cmy, cmz = (
        np.array(ch).T for ch in
        (charge1, chargem1, chargex, chargey, chargez, chargemx, chargemy, chargemz)
    )

    dq = np.zeros((3, 3, 3))

    # Diagonal components: central finite difference along each Cartesian direction
    for i in range(3):
        for j in range(3):
            dq[i, j, j] = (c1[i, j] - cm1[i, j]) / E

    # Off-diagonal — x perturbation
    dq[2,0,1]=dq[2,1,0]=0.5*(np.sqrt(2)*(cx[2,0]-cmx[2,0])-(c1[2,0]-cm1[2,0])-(c1[2,1]-cm1[2,1]))/E
    dq[0,0,1]=dq[0,1,0]=0.5*(cx[0,0]-cmx[0,0]-cx[1,0]+cmx[1,0]-c1[0,0]+cm1[0,0]-c1[0,1]+cm1[0,1])/E
    dq[1,1,0]=dq[1,0,1]=0.5*(cx[0,0]-cmx[0,0]+cx[1,0]-cmx[1,0]-c1[1,0]+cm1[1,0]-c1[1,1]+cm1[1,1])/E

    # Off-diagonal — y perturbation
    dq[0,1,2]=dq[0,2,1]=0.5*(np.sqrt(2)*(cy[0,1]-cmy[0,1])-(c1[0,1]-cm1[0,1])-(c1[0,2]-cm1[0,2]))/E
    dq[1,1,2]=dq[1,2,1]=0.5*(cy[1,1]-cmy[1,1]-cy[2,1]+cmy[2,1]-c1[1,1]+cm1[1,1]-c1[1,2]+cm1[1,2])/E
    dq[2,2,1]=dq[2,1,2]=0.5*(cy[1,1]-cmy[1,1]+cy[2,1]-cmy[2,1]-c1[2,1]+cm1[2,1]-c1[2,2]+cm1[2,2])/E

    # Off-diagonal — z perturbation
    dq[1,2,0]=dq[1,0,2]=0.5*(np.sqrt(2)*(cz[1,2]-cmz[1,2])-(c1[1,0]-cm1[1,0])-(c1[1,2]-cm1[1,2]))/E
    dq[2,2,0]=dq[2,0,2]=0.5*(cz[2,2]-cmz[2,2]-cz[0,2]+cmz[0,2]-c1[2,2]+cm1[2,2]-c1[2,0]+cm1[2,0])/E
    dq[0,0,2]=dq[0,2,0]=0.5*(cz[2,2]-cmz[2,2]+cz[0,2]-cmz[0,2]-c1[0,2]+cm1[0,2]-c1[0,0]+cm1[0,0])/E

    return dq


def build_charge_derivatives(charges, nat, E):
    """
    Build the per-atom BEC derivative tensor list for all atoms.

    Parameters
    ----------
    charges : dict   — output of load_bec_charges()
    nat     : int    — number of atoms
    E       : float  — electric field magnitude (eV/Å)

    Returns
    -------
    dq : list[ndarray(3,3,3)]  — length nat; dq[t] = derivative tensor for atom t
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

    VASP-specific notes:
      - Frequencies are already in cm-1 (no THz conversion applied).
      - Eigenvectors are already mass-normalised by VASP; do NOT divide
        by sqrt(mass) again in the Raman loop.
      - All 3N modes are typically present (no acoustic filtering).

    Parameters
    ----------
    freqs_file   : str — path to frequency file, shape (3N,), unit cm-1
    eigvecs_file : str — path to eigenvector file, shape (3N × 3N)
    modes_max    : int — 3*nat, used only for the log message

    Returns
    -------
    eigvals : ndarray (N_modes,)   — frequencies in cm-1  (unchanged)
    hws     : ndarray (N_modes,)   — frequencies in meV
    eigvecs : ndarray (3N, N_modes)
    n_modes : int
    """
    freqs   = np.loadtxt(freqs_file)    # cm-1  (already converted by VASP)
    eigvecs = np.loadtxt(eigvecs_file)  # mass-normalised displacement
    n_modes = eigvecs.shape[1]

    assert len(freqs) == n_modes, (
        f"freqs has {len(freqs)} entries but eigvecs has {n_modes} columns!")

    if modes_max is not None:
        print(f"Modes loaded: {n_modes} / {modes_max}")

    eigvals = freqs             # cm-1 — no conversion needed
    hws     = freqs * CM1MEV   # cm-1 → meV
    return eigvals, hws, eigvecs, n_modes


# ===========================================================================
# VI.  Raman activity calculator  (VASP: eigenvectors already mass-normalised)
# ===========================================================================

def compute_raman_activities(dq, eigvecs, nat, n_modes):
    """
    Compute the Raman scattering activity for each phonon mode.

    VASP eigenvectors are already mass-normalised — no /sqrt(mass) is applied.

    Implements Eqs. 7–8 of Zhang et al. (2025):
      Eq. 7  — Raman tensor contribution per atom:
               act[i,j,k] = dq_t[i,k,j] * e_t[i]
               where e_t = eigenvector component for atom t (mass-normalised by VASP)
      Eq. 8  — Raman scattering activity:
               A_s = 45·ᾱ² + 7·β²
               ᾱ  = (R_xx + R_yy + R_zz) / 3          (mean polarizability derivative)
               β² = [(R_xx-R_yy)² + (R_xx-R_zz)² + (R_yy-R_zz)²
                     + 6(R_xy² + R_xz² + R_yz²)] / 2  (anisotropy)

    Parameters
    ----------
    dq      : list[ndarray(3,3,3)]  — per-atom BEC derivative tensors
    eigvecs : ndarray (3N, N_modes) — VASP mass-normalised eigenvectors
    nat     : int
    n_modes : int

    Returns
    -------
    activity : list[float]  — Raman activity per mode, length n_modes
    """
    activity = [0.0] * n_modes

    for s in range(n_modes):
        # VASP eigenvectors are already mass-normalised; use directly
        eigvec = eigvecs[:, s].reshape((nat, 3))

        # Accumulate polarizability tensor over all atoms
        ra_tot = np.zeros((3, 3))
        for t in range(nat):
            dqt     = dq[t]
            eigvect = eigvec[t, :]
            # Eq. 7
            act = np.zeros((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        act[i, j, k] = dqt[i, k, j] * eigvect[i]
            ra_tot += act[0] + act[1] + act[2]

        # Scale by global constant 1 / (4πε₀)
        ra = ra_tot / (4.0 * np.pi * EPS0)

        # Eq. 8 — mean polarizability derivative
        alpha = (ra[0, 0] + ra[1, 1] + ra[2, 2]) / 3.0

        # Eq. 8 — anisotropy of the polarizability tensor derivative
        beta2 = ((ra[0, 0] - ra[1, 1])**2
                 + (ra[0, 0] - ra[2, 2])**2
                 + (ra[1, 1] - ra[2, 2])**2
                 + 6.0 * (ra[0, 1]**2 + ra[0, 2]**2 + ra[1, 2]**2)) / 2.0

        # Eq. 8 — Raman scattering activity
        activity[s] = 45.0 * alpha**2 + 7.0 * beta2

    return activity


# ===========================================================================
# VII.  Output writer  (with metadata header)
# ===========================================================================

def write_csv(out_csv, eigvals, activity, formula='', dopants='',
              composition='', E=None):
    """
    Write mode index, frequency (cm-1), and Raman activity to a CSV file with metadata comment header.

    Parameters
    ----------
    out_csv  : str             — output file path
    eigvals  : ndarray (N,)   — frequencies in cm-1
    activity : list[float]    — Raman activities, length N

    Header lines (parsed by compare_raman.py):
      # Formula:      e.g. Na3PS4
      # Dopants:      e.g. Ca(x=0.06)/Cl(x=0.12)   (omitted if undoped)
      # E_field:      e.g. 0.02
      # Composition:  e.g. 06-12
      # Mode,Freq_cm-1,Activity
    """
    with open(out_csv, 'w') as fh:
        fh.write(f"# Formula: {formula}\n")
        if dopants:
            fh.write(f"# Dopants: {dopants}\n")
        if E is not None:
            fh.write(f"# E_field: {E:g}\n")
        if composition:
            fh.write(f"# Composition: {composition}\n")
        fh.write("# Mode,Freq_cm-1,Activity\n")
        for i, (freq, act) in enumerate(zip(eigvals, activity), 1):
            fh.write(f"{i:04d},{freq:.6f},{act:.6f}\n")
    print(f"Written: {out_csv}  ({len(eigvals)} modes)")


# ===========================================================================
# VIII.  Title / metadata builder
# ===========================================================================

def parse_doping_fractions(parent_name):
    """
    Extract doping level fractions from the parent directory name.

    Convention: all-digit tokens (separated by '-' or '_') are interpreted
    as doping percentages × 100.
      '06-12'      → [0.06, 0.12]
      'run_06_12'  → [0.06, 0.12]  (non-digit tokens are ignored)
      '10'         → [0.10]        (single dopant)
    """
    tokens = re.split(r'[-_]', parent_name)
    return [int(t) / 100.0 for t in tokens if re.fullmatch(r'\d+', t)]


def identify_dopants(species_names, species_n, threshold=DOPANT_THRESHOLD):
    """
    Classify species as host or dopant by their fractional count in the supercell.

    Species whose fraction of total atoms < threshold are dopants.
    Default threshold = DOPANT_THRESHOLD (0.05), i.e. < 5% of atoms.
    """
    total = float(sum(species_n))
    host_names, host_n, dopant_names, dopant_n = [], [], [], []
    for name, n in zip(species_names, species_n):
        if n / total < threshold:
            dopant_names.append(name); dopant_n.append(int(n))
        else:
            host_names.append(name);   host_n.append(int(n))
    return host_names, host_n, dopant_names, dopant_n


def approximate_host_formula(host_names, host_n):
    """
    Recover the ideal host formula despite doping-shifted stoichiometry.

    Strategy: divide all host counts by the smallest host count and round
    to the nearest integer.  Works because the least-abundant host species
    (e.g. P in Na3PS4) typically has stoichiometric coefficient 1.

    Examples
    --------
    Na23 P8 S28  (Na3PS4, 1 Ca + 4 Cl dopants) → Na3PS4  ✓
    Na24 P8 S32  (undoped Na3PS4 supercell)     → Na3PS4  ✓
    Ba8  Ti8 O24 (undoped BaTiO3 supercell)     → BaTiO3  ✓
    """
    min_n  = min(host_n)
    ratios = [max(1, round(n / min_n)) for n in host_n]
    return ''.join(f"{name}{r if r > 1 else ''}" for name, r in zip(host_names, ratios))


def build_csv_metadata(species_names, species_n, parent_name, E):
    fracs = parse_doping_fractions(parent_name)
    host_names, host_n, dopant_names, _ = identify_dopants(species_names, species_n)
    formula = approximate_host_formula(host_names, host_n)
    if dopant_names:
        dop_parts = [f"{name}(x={fracs[i]:.2f})" if i < len(fracs) else name
                     for i, name in enumerate(dopant_names)]
        dopants_str = '/'.join(dop_parts)
    else:
        dopants_str = ''
    return formula, dopants_str


def build_plot_title(formula, dopants_str, E, gamma):
    """
    Construct an informative, fully dynamic plot title.

    Title format:
      <host_formula> [<dopant>(x=<frac>) / ...  doped]  |  E = <E> eV/Å  |  FWHM = <gamma> cm⁻¹
    """
    chem_part = f"{formula} [{dopants_str} doped]" if dopants_str else formula
    return f"{chem_part}  |  E = {E} eV/Å  |  FWHM = {gamma} cm$^{{-1}}$"


# ===========================================================================
# IX.  Main orchestrator
# ===========================================================================

def main():
    args        = parse_args()
    parent_name = Path.cwd().parent.name

    out_csv = args.out_csv or f"raman_{parent_name}_{args.E:g}.csv"
    out_png = args.out_png or f"raman_{parent_name}_{args.E:g}.png"

    print(f"\n=== RASCBEC (VASP)  |  tag: {parent_name}  |  E = {args.E:g} eV/Å ===\n")

    # 1. Structure
    vol, species_n, atomic_mass, ntype, nat, modes, species_names = structure_info('POSCAR')

    # 2. Born Effective Charges
    print("\nLoading BEC tensors ...")
    charges = load_bec_charges(OUTCAR_MAP, nat)

    # 3. BEC derivatives
    dq = build_charge_derivatives(charges, nat, args.E)

    # 4. Phonon data
    print("\nLoading VASP phonon data ...")
    eigvals, hws, eigvecs, n_modes = load_vasp_phonon_data(
        'freqs_vasp.dat', 'eigvecs_vasp.dat', modes_max=modes)

    # 5. Raman activities
    print("\nComputing Raman activities ...")
    activity = compute_raman_activities(dq, eigvecs, nat, n_modes)

    # 6. Build metadata and write CSV
    formula, dopants_str = build_csv_metadata(species_names, species_n, parent_name, args.E)
    write_csv(out_csv, eigvals, activity,
              formula=formula, dopants=dopants_str,
              composition=parent_name, E=args.E)

    # 7. Plot
    if not args.no_plot:
        title = build_plot_title(formula, dopants_str, args.E, args.gamma)
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
        print(f"Written: {out_png}")

    print("\nDone.")


if __name__ == '__main__':
    main()
