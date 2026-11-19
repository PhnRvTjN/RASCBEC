#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Title: Calculation of Raman Activities from VASP and Phonopy Output Files

Reads VASP/phonopy output and computes the Raman (Placzek) activity of each
Gamma-point phonon mode via the RASCBEC method -- the born-effective-charge
(BEC) field-derivative response convolved with the phonon eigenvectors.
Two input modes are supported, selected automatically:

  * Single mode -- a phonopy YAML at Gamma (qpoints.yaml, or mesh.yaml +
    irreps.yaml) together with the eight finite-field OUTCAR BEC files.
  * NAC multi-direction mode -- when nac_directions.json or dirXX/
    direction.conf folders are present, the script instead performs a
    Lebedev-Laikov quadrature (order 3/5/7/9, via scipy.integrate.
    lebedev_rule) over the directional eigensystems and writes a weighted
    spherical average. NAC outputs carry the automatic _NAC suffix.
    No installer is required.

Authors:
  Rui Zhang               -- original RASCBEC method and script
  Phani Ravi Teja Nunna   -- refactor and enhancements

Capabilities (beyond the original single-file tool):
  * Auto atomic-mass lookup via pymatgen (no manual mass table)
  * Variable mode count -- imaginary and acoustic modes filtered
  * Subdirectory OUTCAR layout (./1/OUTCAR, ./m1/OUTCAR, ...)
  * Chemistry-based output filenames derived from the POSCAR composition
  * CSV metadata header (formula, dopants, file label, E-field)
  * Plot output with an auto-generated informative title
  * Direct phonopy YAML input (qpoints.yaml, or mesh.yaml + irreps.yaml)
  * Phonon-mode symmetry labels (irreps) in the CSV and plot
  * Temperature / laser-wavelength Stokes-intensity options (--temp, --laser-wl)
  * Basis-group and dopant roles (--bas, --dops) with base-formula reconstruction
  * Vectorised Raman-activity loop (numpy einsum)
  * Reverse-scan OUTCAR BEC reader
  * NAC multi-direction mode with Lebedev-Laikov quadrature and weighted
    spherical averaging, C-accelerated YAML parsing (CSafeLoader, ~6x faster),
    and a process-parallel direction loop (--jobs) with parallel BEC reads
  * --lebedev-order: run a lower-order rule on the subset of the computed
    directions (e.g. order 5 = 14 points from 26-point order-7 data); a
    higher order than the data errors out (no silent fallback)
  * Existing output files are always overwritten (no flag required)

Date:
  2025-04-28  original (Rui Zhang)
  2026-02-23  refactor: pymatgen atomic masses, variable mode count,
              subdirectory OUTCAR paths, chemistry-based output filenames,
              CSV metadata header, plot option, informative plot title
  2026-03-04  chemistry-based naming (dopant file labels)
  2026-03-05  phonopy YAML input, irrep labels, vectorised activity loop,
              reverse-scan OUTCAR BEC reader
  2026-11-16  --temp/--laser-wl Stokes intensity, --bas/--dops roles,
              NAC multi-direction mode
  2026-11-18  C-accelerated YAML parsing, process-parallel --jobs loop with
              parallel BEC reads, scipy-backed Lebedev-Laikov rule (order
              3/5/7/9), --lebedev-order subset selection, always-overwrite
              outputs

License: GNU GENERAL PUBLIC LICENSE

Cite: Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation
via born effective charge." Computer Physics Communications 307 (2025): 109425

Description:
Computes the Raman activity per phonon mode from VASP and phonopy output
using the RASCBEC method.

Required files (in the working directory):
  POSCAR             -- structure file
  ./1/OUTCAR         -- BEC, E-field along unrotated +
  ./m1/OUTCAR        -- BEC, E-field along unrotated -
  ./x/OUTCAR         -- BEC, E-field along +x (rotated along x)
  ./mx/OUTCAR        -- BEC, E-field along -x
  ./y/OUTCAR         -- BEC, E-field along +y (rotated along y)
  ./my/OUTCAR        -- BEC, E-field along -y
  ./z/OUTCAR         -- BEC, E-field along +z (rotated along z)
  ./mz/OUTCAR        -- BEC, E-field along -z
  qpoints.yaml       -- phonopy phonon data at Gamma (or mesh.yaml, see --phonon-yaml)
  irreps.yaml        -- phonopy irreducible representations at Gamma

E-field magnitude is read automatically from ./1/OUTCAR (EFIELD_PEAD tag);
override with --E only to force a specific value.

Output naming (chemistry-based, derived from POSCAR; no directory tags):
  <formula>[_<dopantpart>...]_E<E>.csv / .png
  undoped Na3PS4, E=0.02  -> Na3PS4_E0.02.csv
  Ca+Cl co-doped Na3PS4   -> Na3PS4_Ca01-Cl06_E0.02.csv

Dopant roles (--bas / --dops)
-----------------------------
Explicit --bas / --dops values take precedence over automatic inference
(dopants = species below 5 % of the supercell atoms, DOPANT_THRESHOLD).
When --bas is given, it defines the host-formula normalization AND the
denominator for the dopant x-values (x = n_dopant / n_bas), e.g.
Li5La9Ti16O48 --bas Ti -> Li0.31La0.56TiO3. When omitted, the
least-abundant host species is used as the reference.

--bas may also be a `+`-joined group of species (e.g. --bas Ti+Ga). The
group's member counts are summed into a single reference, so the group is
normalized to one and may include dopant species. For doped LLTO this
normalizes the formula so Ti + Ga = 1 (e.g. Li60La90Ti150O360 with
--bas Ti+Ga -> Li0.38La0.56Ti0.94O2.25, dopant x-values reported
per 160 Ti+Ga sites).

Base (undoped) formula
-----------------------
The stored / `# Formula` value is the BASE host composition, not the raw
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
  # Mode,Freq_cm-1,Activity,Irrep
  0001,<freq>,<activity>,A1
  ...

Dependencies:
  argparse, pathlib, numpy, yaml, pymatgen, plot_raman
  (scipy for NAC multi-direction mode: scipy.integrate.lebedev_rule)

Usage:
  python RASCBEC_phonopy.py                                # all defaults
  python RASCBEC_phonopy.py --E 0.02 --gamma 0.2
  python RASCBEC_phonopy.py --phonon-yaml mesh.yaml
  python RASCBEC_phonopy.py --irreps irreps.yaml
  python RASCBEC_phonopy.py --no-plot
  python RASCBEC_phonopy.py --freq-min 50 --freq-max 600
  python RASCBEC_phonopy.py --bas Ti+Ga --dops Y Ga        # doped LLTO, Ti+Ga = 1
  python RASCBEC_phonopy.py --jobs 8                       # cap NAC parallelism at 8 workers
  python RASCBEC_phonopy.py --lebedev-order 5              # NAC: order-5 rule (14 pts); from
                                                           #   order-7 data this uses the 14
                                                           #   order-5 directions and reweights
  python RASCBEC_phonopy.py --help
  (existing output files are always overwritten -- no flag required)
"""

import argparse
import os
import numpy as np
import re
import yaml
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from pymatgen.core import Element

from plot_raman import (
    plot_raman_spectrum,
    build_raman_title,
    DOPANT_SUBSTITUTIONS,
    THZCM1,
    KB_CM1,
    DEFAULT_TEMP,
    DEFAULT_LASER_WL,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

THZMEV  = 4.13566553853599   # THz -> meV
EPS0    = 55.2635e-4         # e0 in e^2 / (eV*Å)
KB_T    = 8.617333e-2 * 298  # k_B T at 298 K, meV

# Species whose fractional count in the supercell falls below this threshold
# are classified as dopants. Example: 1 Ca in 64 atoms = 0.016 < 0.05.
DOPANT_THRESHOLD = 0.05

# Subdirectory layout expected for the 8 finite-difference OUTCAR files.
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
# 0. Fast YAML loading
# ===========================================================================

# yaml.safe_load() uses PyYAML's *pure-Python* SafeLoader, which is slow on
# large files (tens of seconds per phonopy qpoints.yaml). CSafeLoader is the
# C-accelerated equivalent (~6x faster; identical safe semantics -- no
# arbitrary Python object construction). Note: even the C loader holds the GIL
# for the whole parse, so parallelism across the NAC directions comes from
# a *process* pool (see _nac_direction_worker), not threads.
_YAML_LOADER = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)

def fast_yaml_load(stream):
    """Parse a YAML file/stream using the C-accelerated loader when available.

    Drop-in for ``yaml.safe_load`` but ~6x faster. Falls back to ``safe_load``
    if the C extension is not built into this PyYAML installation.
    """
    if _YAML_LOADER is yaml.SafeLoader:
        return yaml.safe_load(stream)
    return yaml.load(stream, Loader=_YAML_LOADER)

# ===========================================================================
# I. Argument parser
# ===========================================================================

def parse_args():
    """Return parsed CLI arguments."""
    p = argparse.ArgumentParser(
        description='RASCBEC Raman activity calculator (phonopy YAML input).')
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
    p.add_argument('--phonon-yaml', default='qpoints.yaml',
                   help='Phonopy YAML file containing Gamma-point phonon data. '
                        'Accepts qpoints.yaml or mesh.yaml (default: qpoints.yaml)')
    p.add_argument('--irreps', default='irreps.yaml',
                   help='Phonopy irreps.yaml file with symmetry labels '
                        '(default: irreps.yaml)')
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
    p.add_argument('--input-mode', choices=['auto', 'single', 'multi'], default='auto',
                   help='Default auto: NAC manifest/dir folders first, otherwise original single YAML. Explicit --phonon-yaml selects single mode.')
    p.add_argument('--directions-json', '--direction-json', dest='directions_json', default=None,
                   help='Optional override; nac_directions.json is discovered automatically.')
    p.add_argument('--nac-min-thz', type=float, default=0.1,
                   help='Multi-mode minimum retained frequency in THz (default: 0.1).')
    p.add_argument('--translation-overlap', type=float, default=0.95,
                   help='Exclude rigid translations by mass-weighted overlap in multi mode (default: 0.95).')
    p.add_argument('--lebedev-order', dest='lebedev_order', type=int, default=None,
                   help='NAC Lebedev-Laikov quadrature order (3, 5, 7, or 9 = 6/14/26/38 '
                        'points, via scipy.integrate.lebedev_rule). Default auto: inferred '
                        'from the discovered directions (6->3, 14->5, 26->7, 38->9). '
                        'Setting a LOWER order than the data selects that rule\'s subset '
                        'of the available directions and reweights to that order; setting a '
                        'HIGHER order than the data errors out (those directions were not '
                        'computed).')
    # Outputs always overwrite existing files (no --overwrite / --no-overwrite flag).
    p.add_argument('--jobs', type=int, default=None,
                   help='Number of workers for the NAC direction loop (worker '
                        'processes -- YAML parsing is CPU/GIL-bound) and BEC file '
                        'reads. Default: auto (min(#directions, CPU count)); use '
                        '--jobs 1 to force sequential processing.')
    return p.parse_args()

# ===========================================================================
# II. Structure reader
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
    lines 3-5: lattice vectors (Å)
    line 6   : element symbols
    line 7   : atom counts per element
    line 8+  : coordinate type and positions

    Atomic masses are looked up automatically from pymatgen's Element
    database, avoiding hard-coded mass tables.

    Parameters
    ----------
    poscar : str -- path to POSCAR (default 'POSCAR')

    Returns
    -------
    vol          : float      -- unit-cell volume (Å^3)
    species_n    : ndarray    -- atom counts per species, shape (ntype,)
    atomic_mass  : list       -- atomic masses (amu) per species
    ntype        : int        -- number of distinct species
    nat          : int        -- total number of atoms
    modes        : int        -- 3*nat (theoretical maximum mode count)
    species_names: list[str]  -- element symbol strings
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
    print(f"Masses  : {[round(m, 4) for m in atomic_mass]} amu (via pymatgen)")
    print(f"Atoms   : {nat} | Max modes: {modes}")
    return vol, species_n, atomic_mass, ntype, nat, modes, species_names

# ===========================================================================
# III. Born Effective Charge readers
# ===========================================================================

def read_efield_pead(outcar_path='./1/OUTCAR'):
    """
    Parse the EFIELD_PEAD magnitude from a VASP OUTCAR.

    VASP writes EFIELD_PEAD twice in the OUTCAR:
    1st (INCAR echo): EFIELD_PEAD = 0.04 0.04 0.04 (Finite Electric Field...)
    2nd (VASP-parsed): EFIELD_PEAD= 0.0400 0.0400 0.0400

    The second occurrence is VASP's own reformatted version -- no trailing
    text, consistent spacing -- and is what VASP actually uses internally.
    We skip the first match and read the second.

    The largest absolute component is returned as E, robust against
    near-zero values in the inactive Cartesian directions.

    Only ./1/OUTCAR (the unscaled reference) should be passed here.
    The rotated-field OUTCARs carry a sqrt(2)-scaled field.

    Parameters
    ----------
    outcar_path : str -- path to the unscaled OUTCAR (default './1/OUTCAR')

    Returns
    -------
    E : float -- electric field magnitude in eV/Å

    Raises
    ------
    ValueError -- if two EFIELD_PEAD occurrences are not found in the file
    """
    pattern = re.compile(
        r'EFIELD_PEAD\s*=\s*([-.\dE+]+)\s+([-.\dE+]+)\s+([-.\dE+]+)',
        re.IGNORECASE)
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

# ===========================================================================
# IV. BEC derivative tensor
# ===========================================================================

def charge_derivative(charge1, chargem1, chargex, chargey, chargez,
                      chargemx, chargemy, chargemz, E):
    """
    Compute the 3rd-rank BEC derivative tensor dZ*/dE for one atom.

    The tensor is assembled by finite differences from the 8 perturbed BEC
    tensors. Diagonal components use a simple central difference along each
    Cartesian axis; off-diagonal components are recovered from the
    rotated-field perturbations following the RASCBEC scheme
    (see supplementary material of Zhang et al. 2025).

    Index convention:
    dq[i, j, k] = dZ*_{ij} / dE_k

    Parameters
    ----------
    charge{1,m1,x,y,z,mx,my,mz} : list[3x3]
        BEC tensor for the atom under each E-field direction.
        '1'/'m1'             = unrotated field +/-;
        'x'/'mx', 'y'/'my', 'z'/'mz' = 45-degree-rotated field +/-.
    E : float -- electric field magnitude (eV/Å)

    Returns
    -------
    dq : ndarray, shape (3, 3, 3)
    """
    c1, cm1, cx, cy, cz, cmx, cmy, cmz = (
        np.array(ch).T for ch in
        (charge1, chargem1, chargex, chargey, chargez, chargemx, chargemy, chargemz))

    dq = np.zeros((3, 3, 3))

    for i in range(3):
        for j in range(3):
            dq[i, j, j] = (c1[i, j] - cm1[i, j]) / E

    dq[2,0,1]=dq[2,1,0]=0.5*(np.sqrt(2)*(cx[2,0]-cmx[2,0])-(c1[2,0]-cm1[2,0])-(c1[2,1]-cm1[2,1]))/E
    dq[0,0,1]=dq[0,1,0]=0.5*(cx[0,0]-cmx[0,0]-cx[1,0]+cmx[1,0]-c1[0,0]+cm1[0,0]-c1[0,1]+cm1[0,1])/E
    dq[1,1,0]=dq[1,0,1]=0.5*(cx[0,0]-cmx[0,0]+cx[1,0]-cmx[1,0]-c1[1,0]+cm1[1,0]-c1[1,1]+cm1[1,1])/E

    dq[0,1,2]=dq[0,2,1]=0.5*(np.sqrt(2)*(cy[0,1]-cmy[0,1])-(c1[0,1]-cm1[0,1])-(c1[0,2]-cm1[0,2]))/E
    dq[1,1,2]=dq[1,2,1]=0.5*(cy[1,1]-cmy[1,1]-cy[2,1]+cmy[2,1]-c1[1,1]+cm1[1,1]-c1[1,2]+cm1[1,2])/E
    dq[2,2,1]=dq[2,1,2]=0.5*(cy[1,1]-cmy[1,1]+cy[2,1]-cmy[2,1]-c1[2,1]+cm1[2,1]-c1[2,2]+cm1[2,2])/E

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
    charges : dict -- output of load_bec_charges()
    nat     : int  -- number of atoms
    E       : float -- electric field magnitude (eV/Å)

    Returns
    -------
    dq : list[ndarray(3,3,3)] -- dq[t] is the derivative tensor for atom t
    """
    return [charge_derivative(
        charges['1'][t], charges['m1'][t],
        charges['x'][t], charges['y'][t],
        charges['z'][t], charges['mx'][t],
        charges['my'][t], charges['mz'][t], E)
        for t in range(nat)]

# ===========================================================================
# V. Phonon data loader  (reads phonopy YAML directly)
# ===========================================================================

def load_phonon_data(yaml_file='qpoints.yaml', irreps_file='irreps.yaml',
                     modes_max=None):
    """
    Load phonon frequencies, eigenvectors, and symmetry labels from phonopy
    YAML output files.

    Replaces the old freqs_phonopy.dat / eigvecs_phonopy.dat workflow.
    Reads the Gamma-point band data from *yaml_file* (qpoints.yaml or
    mesh.yaml) and the irreducible representation labels from *irreps_file*
    (irreps.yaml).

    Phonopy eigenvector YAML layout:
    eigvec[nat][3][2] -- outer list iterates over atoms, middle list over
    Cartesian components (x, y, z), innermost list is the [real, imag] pair.
    At Gamma the imaginary part is zero; only the real part is retained.
    Some phonopy versions insert null YAML entries (Python None) as
    atom-index comment markers; these are filtered before indexing.

    Two independent filters are applied and a mode is dropped if either
    condition is true:
    (1) freq < 0.1   -- imaginary modes.  A soft mode at e.g. -5.3 THz may
                      carry a valid irrep label in irreps.yaml yet must
                      still be excluded from the Raman calculation.
    (2) ir_label is None -- acoustic / unassigned modes as classified by
                      phonopy (typically the three Gamma-point translations).

    The returned eigvecs matrix has shape (3*nat, N_modes): rows are
    Cartesian displacement components ordered atom_0_x, atom_0_y, atom_0_z,
    atom_1_x, ...; columns are retained modes.

    Parameters
    ----------
    yaml_file   : str -- path to qpoints.yaml or mesh.yaml
    irreps_file : str -- path to irreps.yaml (pass None to skip labels)
    modes_max   : int -- 3*nat, used only for the retained-modes log line

    Returns
    -------
    eigvals      : ndarray (N_modes,)     -- frequencies in cm-1
    hws          : ndarray (N_modes,)     -- frequencies in meV
    eigvecs      : ndarray (3N, N_modes)  -- real eigenvector matrix
    n_modes      : int
    irrep_labels : list[str]              -- symmetry label per retained mode
    """
    with open(yaml_file, 'r') as fh:
        ydata = fast_yaml_load(fh)

    band_to_irrep = {}
    if irreps_file and Path(irreps_file).exists():
        with open(irreps_file, 'r') as fh:
            idata = fast_yaml_load(fh)
        for nm in idata.get('normal_modes', []):
            label = nm.get('ir_label', None)
            for idx in nm.get('band_indices', []):
                band_to_irrep[idx] = label
    else:
        print(f"  Warning: {irreps_file} not found; irrep labels will be empty.")

    bands = ydata['phonon'][0]['band']
    freqs_list, vecs_list, irrep_labels = [], [], []

    for band_idx_0, band in enumerate(bands):
        band_idx_1 = band_idx_0 + 1
        freq       = band['frequency']   # THz
        label      = band_to_irrep.get(band_idx_1, None)

        # Filter (1): imaginary -- independent of irrep label
        if freq < 0.1:
            continue
        # Filter (2): acoustic / unassigned
        if label is None:
            continue

        # Eigenvector layout: eigvec[nat][3][2] where [2] = [real, imag].
        # None entries are phonopy atom-index comment markers; skip them.
        raw   = band['eigenvector']
        atoms = [a for a in raw if a is not None]          # length = nat
        vec   = np.array([[comp[0] for comp in atom]
                          for atom in atoms]).ravel()      # shape (3*nat,)

        freqs_list.append(freq)
        vecs_list.append(vec)
        irrep_labels.append(str(label))

    freqs   = np.array(freqs_list)
    eigvecs = np.column_stack(vecs_list)   # shape (3*nat, n_modes)
    n_modes = len(freqs)

    if modes_max is not None:
        print(f"Retained modes : {n_modes} / {modes_max} "
              f"(removed {modes_max - n_modes} imaginary / acoustic)")

    eigvals = freqs * THZCM1
    hws     = freqs * THZMEV
    return eigvals, hws, eigvecs, n_modes, irrep_labels

# ===========================================================================
# VI. Raman activity calculator
# ===========================================================================

def compute_raman_activities(dq, eigvecs, nat, n_modes, atomic_mass, species_n):
    """
    Compute the Raman scattering activity for each phonon mode.

    Implements Eqs. 7-8 of Zhang et al. (2025).

    Phonopy eigenvectors are column-normalized but NOT mass-weighted.
    Mass weighting (division by sqrt(m_t)) is applied here before the
    polarizability tensor is accumulated.

    Eq. 7 -- polarizability tensor contribution from atom t, mode s:
    R^{(t)}_{ij} = sum_k (dZ*_{ik}/dE_j)_t * e~_{tk}
    where e~_{tk} = e_{tk} / sqrt(m_t)  (mass-normalized displacement).

    The full polarizability derivative tensor for mode s is:
    R_{ij} = sum_t R^{(t)}_{ij} / (4*pi*e0)

    Eq. 8 -- Raman scattering activity:
    A_s = 45*alpha^2 + 7*beta^2
    with
    alpha  = (R_xx + R_yy + R_zz) / 3
    beta^2 = [(R_xx-R_yy)^2 + (R_xx-R_zz)^2 + (R_yy-R_zz)^2
              + 6*(R_xy^2 + R_xz^2 + R_yz^2)] / 2

    The four nested Python loops (modes x atoms x 3 x 3 x 3) from the
    original implementation are replaced by a single numpy einsum:

    ra_all[s, j, k] = sum_{t,i} dq[t, i, k, j] * e_mw[s, t, i]

    where e_mw[s, t, i] is the mass-weighted eigenvector tensor of shape
    (n_modes, nat, 3).  This collapses the O(n_modes * nat * 27) pure-Python
    loop into a BLAS-backed tensor contraction.

    Parameters
    ----------
    dq          : list[ndarray(3,3,3)]  -- per-atom BEC derivative tensors,
                  dq[t][i,k,j] = dZ*_{ij}/dE_k for atom t
    eigvecs     : ndarray (3N, N_modes) -- raw (non-mass-weighted) eigenvectors
    nat         : int
    n_modes     : int
    atomic_mass : list[float]           -- mass per species (amu)
    species_n   : ndarray               -- atom count per species

    Returns
    -------
    activity : list[float] -- Raman scattering activity per mode
    """
    mass_list = np.repeat(atomic_mass, species_n.astype(int))   # (nat,)
    mass_T    = np.tile(mass_list, (3, 1)).T                    # (nat, 3)

    # asarray (not array) so a precomputed ndarray is used in place rather
    # than re-copied on every per-direction call in the multi-direction loop.
    dq_arr = np.asarray(dq)   # (nat, 3, 3, 3), indices: t, i, k, j

    # eigvecs: (3*nat, n_modes) -> (n_modes, nat, 3), then mass-weighted
    eigvecs_mw = (eigvecs.T.reshape(n_modes, nat, 3)
                  / np.sqrt(mass_T[np.newaxis, :, :]))

    # ra_all[s, j, k] = sum_{t,i} dq[t,i,k,j] * e_mw[s,t,i]
    ra_all = (np.einsum('tikj,sti->sjk', dq_arr, eigvecs_mw)
              / (4.0 * np.pi * EPS0))

    alpha = (ra_all[:, 0, 0] + ra_all[:, 1, 1] + ra_all[:, 2, 2]) / 3.0
    beta2 = ((ra_all[:, 0, 0] - ra_all[:, 1, 1])**2
           + (ra_all[:, 0, 0] - ra_all[:, 2, 2])**2
           + (ra_all[:, 1, 1] - ra_all[:, 2, 2])**2
           + 6.0 * (ra_all[:, 0, 1]**2
                  + ra_all[:, 0, 2]**2
                  + ra_all[:, 1, 2]**2)) / 2.0

    return list(45.0 * alpha**2 + 7.0 * beta2)

# ===========================================================================
# VII. Output writer
# ===========================================================================

def write_csv(out_csv, eigvals, activity, irrep_labels,
              formula='', dopants='', file_label='', E=None):
    """
    Write Raman frequencies, activities, and irrep labels to a CSV with a
    metadata header.

    The header lines are parsed by compare_raman.py and plot_raman.py for
    automatic labelling and plot titling.

    File format
    -----------
    # Formula: <formula>
    # Dopants: <dopants>    <- omitted if undoped, e.g. 'Ca=0.125 Cl=0.0625'
    # File_Label: <file_label>   <- omitted if undoped, e.g. 'Ca01-Cl06'
    # E_field: <E>
    # Mode,Freq_cm-1,Activity,Irrep
    0001,<freq>,<activity>,<label>
    ...

    Parameters
    ----------
    out_csv      : str        -- output file path
    eigvals      : array      -- frequencies in cm-1, length n_modes
    activity     : list       -- Raman activities, length n_modes
    irrep_labels : list[str]  -- symmetry label per mode, length n_modes
    formula      : str        -- ideal host formula, e.g. 'Na3PS4'
    dopants      : str        -- dopant descriptor, e.g. 'Ca=0.125 Cl=0.0625'
                                 ('' or 'Undoped' when no dopants)
    file_label   : str        -- filesystem-safe dopant label, e.g. 'Ca01-Cl06'
                                 ('' when undoped)
    E            : float      -- electric field magnitude used (written to header)
    """
    with open(out_csv, 'w') as fh:
        fh.write(f"# Formula: {formula}\n")
        if dopants and dopants != "Undoped":
            fh.write(f"# Dopants: {dopants}\n")
        if file_label:
            fh.write(f"# File_Label: {file_label}\n")
        if E is not None:
            fh.write(f"# E_field: {E:g}\n")
        fh.write("# Mode,Freq_cm-1,Activity,Irrep\n")
        for i, (freq, act, irrep) in enumerate(
                zip(eigvals, activity, irrep_labels), 1):
            fh.write(f"{i:04d},{freq:.6f},{act:.6f},{irrep}\n")
    print(f"Written : {out_csv} ({len(eigvals)} modes)")

# ===========================================================================
# VIII. Chemistry metadata and naming
# (ported from phonon_mode_pred.py: safe_name, host_formula,
#  validate_chemistry_roles, build_chemistry_metadata)
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
    This heuristic works well for dilute-substitution sulfide systems
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
        dopants      : str -- e.g. 'Ca=0.125 Cl=0.0625' ('Undoped' if none)
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
# IX. Main orchestrator
# ===========================================================================

def main_single(args):

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

    # --- 6. Phonon data from YAML ---
    print(f"\nLoading phonon data from {args.phonon_yaml} + {args.irreps} ...")
    eigvals, hws, eigvecs, n_modes, irrep_labels = load_phonon_data(
        args.phonon_yaml, args.irreps, modes_max=modes)

    if eigvecs.shape[0] != 3 * nat:
        raise ValueError(
            f"Eigenvector rows ({eigvecs.shape[0]}) != 3*nat ({3*nat}). "
            "Check that the YAML and POSCAR describe the same structure.")

    # --- 7. Raman activities ---
    print("\nComputing Raman activities ...")
    activity = compute_raman_activities(
        dq, eigvecs, nat, n_modes, atomic_mass, species_n)

    # --- 8. Write CSV ---
    write_csv(out_csv, eigvals, activity, irrep_labels,
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


# ---------------------------------------------------------------------------
# Automatic NAC-direction support. Scientific BEC/Raman routines above are
# unchanged. Multi-mode activities are weighted once, before CSV output.
# ---------------------------------------------------------------------------
def _nac_poscar():
    from pymatgen.core import Structure
    return Structure.from_file('POSCAR')


def _nac_config_direction(directory):
    conf = directory / 'direction.conf'
    if not conf.is_file():
        raise ValueError(f'{directory}: no direction metadata. Keep nac_directions.json or direction.conf.')
    text = conf.read_text()
    if not re.search(r'^\s*NAC\s*=\s*\.TRUE\.', text, re.M | re.I):
        raise ValueError(f'{conf}: NAC = .TRUE. not found.')
    match = re.search(r'^\s*Q_DIRECTION\s*=\s*([^#\n]+)', text, re.M | re.I)
    if not match:
        raise ValueError(f'{conf}: Q_DIRECTION missing.')
    q = np.array([float(v) for v in match.group(1).split()])
    if q.shape != (3,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError(f'{conf}: invalid Q_DIRECTION.')
    return q


def _lebedev_points(order):
    """Return (points, weights) for a Lebedev-Laikov quadrature of *order*.

    points  : ndarray (3, m) -- unit direction vectors, all octants signed
    weights : ndarray (m,)   -- normalized to sum to 1 (for the spherical avg)

    Delegates to ``scipy.integrate.lebedev_rule`` (Parrish getLebedevSphere),
    so the node set is no longer hand-coded. Supported orders: 3, 5, 7, 9,
    11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71,
    77, 83, 89, 95, 101, 107, 113, 119, 125, 131 (point counts 6, 14, 26,
    38, 50, ...). Order 7 is the classic 26-point rule.
    """
    try:
        from scipy.integrate import lebedev_rule
    except ImportError as exc:
        raise ImportError(
            'scipy is required for NAC multi-direction mode '
            '(scipy.integrate.lebedev_rule). Install scipy, or pin the rule '
            'you already have via nac_directions.json.') from exc
    try:
        x, w = lebedev_rule(int(order))
    except NotImplementedError as exc:
        raise ValueError(
            f'Lebedev order {order!r} not available via scipy; supported orders '
            'are 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, '
            '41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131.') from exc
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    x = x / np.linalg.norm(x, axis=0, keepdims=True)
    w = w / w.sum()
    return x, w


def _lebedev_subset(lo, hi):
    """True if every point of the order-`lo` rule lies in the order-`hi` rule.

    Lower-order Lebedev-Laikov rules are generally geometric subsets of higher
    ones, but NOT contiguously: e.g. order 7 (26 pts) is NOT a subset of order
    9 (38 pts), while orders 3 and 5 are. Used to validate a --lebedev-order
    that is lower than the data's own order.
    """
    lx, _ = _lebedev_points(lo)
    hx, _ = _lebedev_points(hi)
    return all(any(np.allclose(lx[:, i], hx[:, j], atol=1e-7) for j in range(hx.shape[1]))
               for i in range(lx.shape[1]))

def _nac_discover(args):
    import json, sys
    explicit_single = any(x == '--phonon-yaml' or x.startswith('--phonon-yaml=') for x in sys.argv[1:])
    if args.input_mode == 'single' or (explicit_single and args.input_mode == 'auto' and not args.directions_json):
        return None
    manifest = Path(args.directions_json or 'nac_directions.json')
    dirs = sorted(p for p in Path.cwd().iterdir()
                  if p.is_dir() and re.fullmatch(r'dir\d+', p.name))
    if not manifest.is_file() and not dirs:
        if args.input_mode == 'multi' or args.directions_json:
            raise FileNotFoundError('No NAC manifest/direction folders found.')
        return None
    structure = _nac_poscar()
    lattice = np.asarray(structure.lattice.matrix)
    data = None
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        rows = data.get('directions', [])
        if 'nat' in data and int(data['nat']) != len(structure):
            raise ValueError('Manifest atom count differs from POSCAR.')
        if 'lattice' in data and not np.allclose(data['lattice'], lattice, atol=1e-7, rtol=0):
            raise ValueError('Manifest lattice differs from POSCAR.')
        if 'symbols' in data and data['symbols'] != [str(site.specie) for site in structure]:
            raise ValueError('Manifest species/order differs from POSCAR.')
        if 'scaled_positions' in data:
            delta = (np.array(data['scaled_positions']) - structure.frac_coords + .5) % 1 - .5
            if delta.shape != structure.frac_coords.shape or not np.allclose(delta, 0, atol=1e-7):
                raise ValueError('Manifest atomic positions/order differs from POSCAR.')
        root = manifest.resolve().parent
    else:
        rows = []
        root = Path.cwd()
        for directory in dirs:
            q = _nac_config_direction(directory)
            rows.append({'directory': directory.name, 'yaml': f'{directory.name}/qpoints.yaml',
                         'q_direction': q.tolist(), 'weight': None})
    # Determine the order of the data on disk (from its direction count) and
    # the order to actually use. Auto = the disk order. An explicit
    # --lebedev-order LOWER than the disk order selects that rule's subset of
    # the available directions (lower-order Lebedev rules are geometric
    # subsets of higher-order ones, verified against scipy's node sets).
    # Requesting a HIGHER order than the disk order errors out, because those
    # extra directions were never computed (no silent fallback to single mode).
    point_count_to_order = {6: 3, 14: 5, 26: 7, 38: 9}
    order_to_point_count = {3: 6, 5: 14, 7: 26, 9: 38}
    data_order = point_count_to_order.get(len(rows))
    if data_order is None:
        raise ValueError(
            f'Found {len(rows)} NAC directions; expected a complete Lebedev rule '
            '(6/14/26/38 directions -> order 3/5/7/9). Use --lebedev-order to '
            'specify other point counts explicitly.')
    order = args.lebedev_order if args.lebedev_order is not None else data_order
    if order not in order_to_point_count:
        raise ValueError(
            f'--lebedev-order {order} is not supported; supported orders are '
            '3, 5, 7, 9 (point counts 6, 14, 26, 38).')
    if order > data_order:
        raise ValueError(
            f'--lebedev-order {order} needs {order_to_point_count[order]} '
            f'directions, but only {len(rows)} (order {data_order}) are present. '
            f'Run phonopy for all {order_to_point_count[order]} directions first '
            '(no silent fallback to single mode).')
    if order < data_order:
        # Only valid when the lower-order rule is a true geometric subset of the
        # data order. Containment is not contiguous (order 7 is NOT a subset of
        # order 9), so check explicitly before selecting.
        if not _lebedev_subset(order, data_order):
            valid = [lo for lo in (3, 5, 7, 9) if lo < data_order and _lebedev_subset(lo, data_order)]
            raise ValueError(
                f'--lebedev-order {order} is not a subset of the order-{data_order} '
                f'data ({len(rows)} dirs). Lower orders selectable from order '
                f'{data_order} data: {", ".join(map(str, valid)) if valid else "none"}.')
        print(f'Lebedev order: data is order {data_order} ({len(rows)} dirs); '
              f'using the order-{order} subset '
              f'({order_to_point_count[order]} of {len(rows)} dirs).')
    # Validate every on-disk direction against the disk-order rule. Weights are
    # re-assigned from the *target* order below when a subset is selected.
    expected_x, expected_w = _lebedev_points(data_order)   # (3, m) points, (m,) weights
    used = set(); paths = set(); validated = []
    for row in rows:
        path = (root / row.get('yaml', str(Path(row['directory'])/'qpoints.yaml'))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Missing NAC eigensystem: {path}')
        if path in paths:
            raise ValueError(f'Duplicate YAML: {path}')
        paths.add(path)
        if 'q_direction' in row:
            q = np.asarray(row['q_direction'], dtype=float)
        else:
            q = _nac_config_direction(path.parent)
        if q.shape != (3,) or not np.isfinite(q).all() or np.linalg.norm(q)==0:
            raise ValueError(f'Invalid direction: {path}')
        cart = q @ np.linalg.inv(lattice).T; cart /= np.linalg.norm(cart)
        matches = [i for i in range(len(expected_w)) if np.allclose(expected_x[:, i], cart, atol=1e-7)]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError(f'Duplicate/non-Lebedev physical direction: {path}')
        idx = matches[0]
        used.add(idx)
        # Weight must match the scipy rule for this node (manifest may carry it;
        # config-folder discovery resolves it here from the expected set).
        w = row['weight']
        required = float(expected_w[idx])
        if w is None:
            w = required
        if not np.isfinite(w) or not np.isclose(w, required, atol=1e-12, rtol=0):
            raise ValueError(f'Incorrect Lebedev (order {order}) weight: {path}')
        conf = path.parent/'direction.conf'
        if conf.exists():
            qc = _nac_config_direction(path.parent)
            if not np.allclose(qc/np.linalg.norm(qc),q/np.linalg.norm(q),atol=1e-7):
                raise ValueError(f'Manifest/config direction disagreement: {path}')
        if 'cartesian' in row and not np.allclose(row['cartesian'],cart,atol=1e-7):
            raise ValueError(f'Manifest Cartesian direction mismatch: {path}')
        validated.append({'directory':row.get('directory',path.parent.name),'path':path,
                          'weight':w,'cartesian':cart.tolist(),'q_direction':q.tolist()})
    if order < data_order:
        # Select the directions that form the lower-order rule and reweight them
        # to that order's weights (lower-order Lebedev rules are geometric
        # subsets of the data order; weights are NOT simply scaled down).
        target_x, target_w = _lebedev_points(order)
        selected = []
        for r in validated:
            cart = np.asarray(r['cartesian'], dtype=float)
            idx = [i for i in range(len(target_w)) if np.allclose(target_x[:, i], cart, atol=1e-7)]
            if len(idx) != 1:
                continue                       # not a member of the target rule
            r['weight'] = float(target_w[idx[0]])
            selected.append(r)
        if len(selected) != len(target_w):
            raise ValueError(
                f'Order {order} needs {len(target_w)} directions but only '
                f'{len(selected)} of the order-{data_order} data match it.')
        validated = selected
    if not np.isclose(sum(r['weight'] for r in validated),1,atol=1e-12,rtol=0):
        raise ValueError('Weights do not sum to one.')
    return validated, order


def _nac_load_direction(row, structure, masses, args):
    import hashlib, json, warnings
    path = row['path']
    marker = path.parent/'complete.json'
    if marker.exists():
        recorded = json.loads(marker.read_text()).get('yaml_sha256')
        if recorded and recorded != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'YAML checksum mismatch: {path}')
    with path.open() as f:
        data = fast_yaml_load(f)
    points = data.get('phonon',[])
    if len(points)!=1 or not np.allclose(points[0].get('q-position',[99]*3),0,atol=1e-10):
        raise ValueError(f'Expected one Gamma point: {path}')
    nat = len(structure)
    if 'lattice' in data and not np.allclose(data['lattice'], structure.lattice.matrix, atol=1e-6,rtol=0):
        raise ValueError(f'YAML lattice differs from POSCAR: {path}')
    atoms = data.get('points')
    if atoms is not None:
        if len(atoms)!=nat: raise ValueError(f'YAML atom count mismatch: {path}')
        for site,atom in zip(structure,atoms):
            if atom.get('symbol',str(site.specie)) != str(site.specie):
                raise ValueError(f'YAML atom order differs: {path}')
            if 'coordinates' in atom:
                delta = (np.array(atom['coordinates'])-site.frac_coords+.5)%1-.5
                if not np.allclose(delta,0,atol=1e-6): raise ValueError(f'YAML positions differ: {path}')
    bands = points[0]['band']
    if len(bands)!=3*nat: raise ValueError(f'Expected {3*nat} modes: {path}')
    frequencies=[];vectors=[];indices=[];skipped=[]
    trans=np.zeros((3*nat,3))
    for axis in range(3): trans[axis::3,axis]=np.sqrt(masses)/np.sqrt(sum(masses))
    for i, band in enumerate(bands, 1):
        freq = float(band['frequency'])
        if not np.isfinite(freq):
            raise ValueError(f'Invalid frequency: {path}')
        # Exclude these modes before phase/realness checks.
        if freq < args.nac_min_thz:
            skipped.append({
                'source_mode': i,
                'frequency_thz': freq,
                'translation_overlap': None,
                'reason': 'below_minimum_frequency',
            })
            continue
        # Existing eigenvector reading and validation follows.
        raw=[atom for atom in band['eigenvector'] if atom is not None]
        arr=np.asarray(raw,dtype=float)
        if arr.shape!=(nat,3,2) or not np.isfinite(arr).all(): raise ValueError(f'Invalid eigenvector: {path}')
        vec=(arr[:,:,0]+1j*arr[:,:,1]).ravel()
        if not np.isclose(np.linalg.norm(vec),1,atol=1e-5): raise ValueError(f'Nonunit eigenvector: {path}')
        vec*=np.exp(-1j*np.angle(vec[np.argmax(abs(vec))]))
        if np.max(abs(vec.imag))>1e-7: raise ValueError(f'Complex eigenvector needs complex Raman support: {path}')
        vec=vec.real
        translation=float(np.sum(np.abs(trans.T@vec)**2))
        reason = ('below_minimum_frequency' if freq<args.nac_min_thz else
                  'rigid_translation' if translation>args.translation_overlap else '')
        if reason:
            skipped.append({'source_mode':i,'frequency_thz':freq,'translation_overlap':translation,'reason':reason})
            continue
        frequencies.append(freq*THZCM1);vectors.append(vec);indices.append(i)
    if not vectors: raise ValueError(f'No retained modes: {path}')
    if any(float(b['frequency']) < -.1 for b in bands):
        warnings.warn(f'{path}: frequencies below -0.1 THz exist; examine structural stability.')
    return np.array(frequencies),np.column_stack(vectors),indices,skipped


def _nac_direction_worker(payload):
    """Process one NAC direction: load its YAML, compute activities, emit records.

    Module-level (not a closure) so it is picklable and can run in a
    ``ProcessPoolExecutor`` -- required because YAML parsing is GIL-bound and
    cannot be parallelized with threads. All inputs and outputs are plain
    picklable objects (dicts, ndarrays, ``argparse.Namespace``, a pymatgen
    ``Structure``), so the per-direction state needs no cross-process
    coordination.

    Parameters
    ----------
    payload : tuple
        (row, structure, masses, args, dq, atomic_mass, species_n, nat)

    Returns
    -------
    tuple (records, rejected) -- both JSON/pickle-serializable.
    """
    row, structure, masses, args, dq, atomic_mass, species_n, nat = payload
    freqs, vecs, indices, rejected = _nac_load_direction(row, structure, masses, args)
    raw = np.asarray(compute_raman_activities(dq, vecs, nat, len(freqs), atomic_mass, species_n))
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Invalid Raman activity.')
    records = [{'direction': row['directory'], 'source_mode': index,
                'frequency_cm1': float(freq), 'weight': row['weight'],
                'raw_activity': float(activity),
                'weighted_activity': float(activity * row['weight'])}
               for freq, activity, index in zip(freqs, raw, indices)]
    return records, rejected


def _nac_write_outputs(path, chem, E, records, directions, skipped, args, order):
    import json, os, tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    audit=path.with_suffix('.directions.json')
    lines=[f"# Formula: {chem['formula']}"]
    if chem['dopants'] and chem['dopants']!='Undoped': lines.append(f"# Dopants: {chem['dopants']}")
    if chem['file_label']: lines.append(f"# File_Label: {chem['file_label']}")
    lines += [f'# E_field: {E:g}', f'# Phonon_treatment: NAC_Lebedev_order{order}',
              '# Activity_semantics: quadrature_weight_times_raw_Placzek_activity',
              '# Row_semantics: directional_sticks_not_unique_modes',
              '# Mode,Freq_cm-1,Activity,Irrep']
    for i,r in enumerate(records,1):
        r['output_row']=i
        lines.append(f"{i:06d},{r['frequency_cm1']:.12g},{r['weighted_activity']:.12g},")
    details={'treatment':'NAC modes with fixed-field RASCBEC response; no extra electro-optic correction',
             'lebedev_order':order,'n_directions':len(directions),
             'minimum_frequency_thz':args.nac_min_thz,'translation_overlap_threshold':args.translation_overlap,
             'directions':[{**r,'path':str(r['path'])} for r in directions],
             'skipped_modes':skipped,'sticks':records}
    for destination,content in [(path,'\n'.join(lines)+'\n'),(audit,json.dumps(details,indent=2)+'\n')]:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=destination.parent,delete=False) as f:
            temporary=Path(f.name);f.write(content)
        os.replace(temporary,destination)
    print(f'Written : {path} ({len(records)} weighted directional sticks)')
    print(f'Written : {audit}')


def main():
    args=parse_args()
    discovered=_nac_discover(args)
    if discovered is None:
        return main_single(args)
    directions, order=discovered
    if args.nac_min_thz<0 or not np.isfinite(args.nac_min_thz):
        raise ValueError('--nac-min-thz must be finite and nonnegative.')
    if not 0<args.translation_overlap<=1:
        raise ValueError('--translation-overlap must lie in (0,1].')
    structure=_nac_poscar()
    _,species_n,atomic_mass,_,nat,_,species_names=structure_info('POSCAR')
    E=args.E if args.E is not None else read_efield_pead(OUTCAR_MAP['1'])
    if not np.isfinite(E) or E<=0: raise ValueError('Finite positive field magnitude required.')
    chem=build_chemistry_metadata(species_names,species_n,bas_species=args.bas,dop_species=args.dops)
    base=raman_filename(chem,E)+'_NAC'
    out_csv=Path(args.out_csv or f'{base}.csv')
    out_png=Path(args.out_png or out_csv.with_suffix('.png'))
    # Outputs are always overwritten; existing files are replaced in place.
    print(f'Input   : automatic NAC multi-direction mode ({len(directions)} directions, Lebedev order {order})')
    print(f'Output  : {out_csv}')
    print('Loading the eight BEC files once ...')
    charges=load_bec_charges(OUTCAR_MAP,nat,jobs=args.jobs)
    # Materialize the per-atom derivative tensor once; every direction reuses
    # the same array (compute_raman_activities no longer copies it).
    dq=np.asarray(build_charge_derivatives(charges,nat,E))
    masses=np.repeat(atomic_mass,species_n.astype(int))

    # Each direction is an independent (YAML load + einsum) unit of work.
    # YAML parsing is GIL-bound (even the C loader holds the GIL), so a thread
    # pool gives NO speedup -- we use a *process* pool for true parallelism.
    # Every worker input is picklable; _nac_direction_worker is a module-level
    # function so it pickles cleanly for the process pool.
    payloads=[(row,structure,masses,args,dq,atomic_mass,species_n,nat) for row in directions]
    # auto = min(#directions, CPU count): never spawn more workers than there
    # are CPUs, and never more than there are directions. (Use --jobs to
    # override, e.g. --jobs <physical-cores> for the CPU-bound YAML-parse
    # optimum on hyperthreaded machines.)
    workers=_resolve_worker_count(args.jobs,min(len(directions),os.cpu_count() or 1))
    if workers>1 and len(directions)>1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # Python 3.14 defaults to 'forkserver', which re-imports this whole
        # module (numpy + pymatgen) into every worker and swamps the work.
        # 'fork' copies the already-loaded module in place (no re-import) and
        # is safe here: all BEC threads have finished before this point and no
        # other threads are running at fork time.
        try:
            mp_context = mp.get_context('fork')
        except ValueError:
            mp_context = None
        print(f'Processing {len(directions)} NAC directions on {workers} worker processes ...')
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
            results=list(pool.map(_nac_direction_worker,payloads))
    else:
        print(f'Processing {len(directions)} NAC direction(s) sequentially ...')
        results=[_nac_direction_worker(p) for p in payloads]

    records=[];skipped={}
    for row,(rec,rejected) in zip(directions,results):
        records.extend(rec)
        skipped[row['directory']]=rejected
        print(f"  {row['directory']}: {len(rec)} retained, {len(rejected)} excluded")
    records.sort(key=lambda r:r['frequency_cm1'])
    _nac_write_outputs(out_csv,chem,E,records,directions,skipped,args,order)
    if not args.no_plot:
        out_png.parent.mkdir(parents=True,exist_ok=True)
        title=build_raman_title(chem,E,args.gamma*THZCM1,args.temp,suffix='NAC, spherical average')
        plot_raman_spectrum(dat_file=out_csv,out_png=out_png,gamma=args.gamma*THZCM1,
            freq_min=args.freq_min,freq_max=args.freq_max,sticks=not args.no_sticks,
            n_labels=args.n_labels,title=title,temp=args.temp,laser_wl=args.laser_wl)
    print('Done. Weights are already included; downstream Stokes conversion/broadening remain unchanged.')

if __name__ == '__main__':
    main()
