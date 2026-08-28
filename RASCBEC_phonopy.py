#!/usr/bin/env python3

"""
Title: Calculation of Raman Activities from VASP and Phonopy Output Files

Authors: Rui Zhang (original);
Phani Ravi Teja Nunna (refactor and enhancements)

Patches: auto atomic-mass lookup via pymatgen,
variable mode count (filtered imaginary and acoustic modes),
subdirectory-based OUTCAR paths (./1/OUTCAR, ./m1/OUTCAR, ...),
chemistry-based output filenames derived from POSCAR composition,
CSV metadata header (formula, dopants, file label, E-field),
plot output option,
auto-generated informative plot title,
direct phonopy YAML input (qpoints.yaml or mesh.yaml + irreps.yaml),
mode symmetry labels (irreps) in CSV and plot,
vectorised Raman activity loop via numpy einsum,
reverse-scan OUTCAR BEC reader.

Date: 2025-04-28 (original); 2026-02-23 (patched); 2026-03-04 (chemistry naming);
      2026-03-05 (YAML phonon input, irrep labels, vectorised activity loop,
                  reverse-scan OUTCAR BEC reader)

License: GNU GENERAL PUBLIC LICENSE

Cite: Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation
via born effective charge." Computer Physics Communications 307 (2025): 109425

Description:
Reads VASP/phonopy output to calculate Raman activities per phonon mode
using the RASCBEC method.

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
qpoints.yaml       -- phonopy phonon data at Gamma (or mesh.yaml, see --phonon-yaml)
irreps.yaml        -- phonopy irreducible representations at Gamma

E-field magnitude is read automatically from ./1/OUTCAR (EFIELD_PEAD tag).
Override with --E only if you want to force a specific value.

Output naming (chemistry-based, derived from POSCAR; no directory tags):
<formula>[_<dopantpart>... ]_E<E>.csv / .png

Examples:
undoped Na3PS4, E=0.02  -> Na3PS4_E0.02.csv
Ca+Cl co-doped Na3PS4   -> Na3PS4_Ca01-Cl06_E0.02.csv

Dopant roles (--bas / --dops)
-----------------------------
Explicit --bas / --dops values take precedence over automatic inference
(dopants = species below 5 % of the supercell atoms, DOPANT_THRESHOLD).
When --bas is given, it defines the host-formula coefficient of one AND the
denominator for the dopant x-values (x = n_dopant / n_bas), e.g.
Li5La9Ti16O48 --bas Ti -> Li0.31La0.56TiO3. When omitted, the
least-abundant host species is used as the legacy reference.

CSV metadata header (parsed by compare_raman.py & plot_raman.py):
# Formula: Na3PS4
# Dopants: Ca=0.125 Cl=0.0625   <- omitted if undoped
# File_Label: Ca01-Cl06         <- omitted if undoped
# E_field: 0.02
# Mode,Freq_cm-1,Activity,Irrep
0001,<freq>,<activity>,A1
...

Dependencies: argparse, pathlib, numpy, yaml, pymatgen, plot_raman

Usage:
python RASCBEC_phonopy.py                          # all defaults
python RASCBEC_phonopy.py --E 0.02 --gamma 0.2
python RASCBEC_phonopy.py --phonon-yaml mesh.yaml
python RASCBEC_phonopy.py --irreps irreps.yaml
python RASCBEC_phonopy.py --no-plot
python RASCBEC_phonopy.py --freq-min 50 --freq-max 600
python RASCBEC_phonopy.py --help
"""

import argparse
import numpy as np
import re
import yaml

from pathlib import Path
from pymatgen.core import Element

from plot_raman import plot_raman_spectrum

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

THZCM1 = 33.3564095198       # THz -> cm-1
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
                   help='Explicit host basis species. It is normalized to '
                        'coefficient one in the host formula and defines the '
                        'denominator for dopant x-values, e.g. Ti. If omitted, '
                        'the least-abundant host species is used.')
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

def load_bec_charges(outcar_map, nat):
    """
    Read BEC tensors from all 8 finite-difference OUTCAR subdirectories.

    Each subdirectory corresponds to one Cartesian E-field direction.
    The resulting dict is keyed by the same labels used in OUTCAR_MAP
    and later consumed by build_charge_derivatives().

    Parameters
    ----------
    outcar_map : dict -- {label: path} as in OUTCAR_MAP
    nat        : int  -- number of atoms

    Returns
    -------
    charges : dict[str, list[nat x 3 x 3]] -- BEC tensors keyed by direction label
    """
    charges = {}
    for key, path in outcar_map.items():
        print(f"  Reading BEC: {path}")
        charges[key] = get_charges_from_OUTCAR(path, nat)
    return charges

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
        ydata = yaml.safe_load(fh)

    band_to_irrep = {}
    if irreps_file and Path(irreps_file).exists():
        with open(irreps_file, 'r') as fh:
            idata = yaml.safe_load(fh)
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

    Phonopy eigenvectors are column-normalised but NOT mass-weighted.
    Mass weighting (division by sqrt(m_t)) is applied here before the
    polarizability tensor is accumulated.

    Eq. 7 -- polarizability tensor contribution from atom t, mode s:
    R^{(t)}_{ij} = sum_k (dZ*_{ik}/dE_j)_t * e~_{tk}
    where e~_{tk} = e_{tk} / sqrt(m_t)  (mass-normalised displacement).

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

    dq_arr = np.array(dq)   # (nat, 3, 3, 3), indices: t, i, k, j

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

def host_formula(host_names, host_n, basis_species=None):
    """
    Build a compact host-composition label with a consistent normalization.

    When ``basis_species`` is supplied, all host stoichiometric coefficients
    are divided by that species' count, so its displayed coefficient is one.
    The same count is used by :func:`build_chemistry_metadata` to report
    dopant x-values. For example, ``Li5 La9 Ti16 O48`` with ``--bas Ti`` is
    rendered as ``Li0.31La0.56TiO3``. If no basis is supplied, the legacy
    behavior is retained: normalize to the least-abundant host species and
    round to nearby integers.

    This produces a readable analysis label rather than a general-purpose
    crystallographic formula reduction.
    """
    if not host_names or not host_n:
        raise ValueError("Host species could not be determined from the composition.")
    count_map = {name: int(count) for name, count in zip(host_names, host_n)}
    if basis_species is None:
        ref = float(min(host_n))
        coefficients = [float(max(1, round(count / ref))) for count in host_n]
    else:
        if basis_species not in count_map:
            raise ValueError(
                f"Basis species '{basis_species}' is not a host species; "
                "it cannot define the host formula.")
        ref = float(count_map[basis_species])
        coefficients = [count_map[name] / ref for name in host_names]

    parts = []
    for name, coefficient in zip(host_names, coefficients):
        if np.isclose(coefficient, round(coefficient), atol=1e-8):
            integer = int(round(coefficient))
            parts.append(name if integer == 1 else f"{name}{integer}")
        else:
            parts.append(f"{name}{coefficient:.2f}".rstrip("0").rstrip("."))
    return "".join(parts)

def validate_chemistry_roles(species_names, bas_species, dop_species):
    """
    Validate chemistry-role overrides before they are consumed downstream.

    This is the single validation guard through which every explicit CLI
    role override (--bas / --dops) must pass. It exists to catch
    misconfiguration in non-sulfide chemistries (e.g. LLTO) where role
    assignment is not automatic, before any downstream computation,
    filename, or plot silently uses a wrong or inconsistent assignment.

    Checks performed
    -----------------
    - ``bas_species`` and every entry of ``dop_species`` must be a
      non-empty string present in ``species_names`` (when provided).
    - ``dop_species`` must not contain duplicate entries.
    - ``bas_species`` cannot also appear in ``dop_species`` (the
      concentration reference should be a host species, not a dopant).

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
        if bas_species not in species_set:
            raise ValueError(
                f"Basis species '{bas_species}' is not present in the structure "
                f"(available species: {sorted(species_set)}).")

    if dop_species:
        if any(not sp.strip() for sp in dop_species):
            raise ValueError("--dops entries cannot be empty strings.")
        duplicates = sorted({sp for sp in dop_species if dop_species.count(sp) > 1})
        if duplicates:
            raise ValueError(f"--dops contains duplicate species: {sorted(duplicates)}.")

    if bas_species is not None and bas_species in dop_species:
        raise ValueError(
            f"'{bas_species}' cannot be both the basis species & a dopant species.")

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
        Species used as the concentration reference for dopant x-values.
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

    if bas_species is not None:
        resolved_bas_species = bas_species
    elif host_names:
        resolved_bas_species = host_names[int(np.argmin(host_counts))]
    elif len(species_names):
        resolved_bas_species = species_names[int(np.argmin(species_n))]
    else:
        resolved_bas_species = None

    bas_count = (species_count_map[resolved_bas_species]
                 if resolved_bas_species is not None else 1)
    bas_count = max(int(bas_count), 1)

    formula = (
        host_formula(host_names, host_counts, basis_species=bas_species)
        if host_names
        else "Unknown"
    )

    # Preserve the exact CLI order: --dops Ca Cl
    label_dops = list(explicit_dops) if explicit_dops is not None else list(dop_names)

    fractions = {
        name: float(species_count_map.get(name, 0)) / float(bas_count)
        for name in label_dops
    }

    if label_dops:
        dopants = (
            "Undoped"
            if not fractions or all(abs(x) < 1e-12 for x in fractions.values())
            else " ".join(f"{name}={x:.3g}" for name, x in fractions.items())
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

def build_plot_title(chem, E, gamma):
    """
    Build an informative plot title from chemistry metadata and run parameters.

    The chemistry fragment uses the phonon_mode_pred convention:
    ``<formula> | <dopants>`` when doped, else ``<formula>``.

    Parameters
    ----------
    chem    : dict  -- output of :func:`build_chemistry_metadata`
    E       : float -- electric field magnitude (eV/Å)
    gamma   : float -- Lorentzian FWHM (cm-1)

    Returns
    -------
    title : str
    """
    dopants = chem.get("dopants") or ""
    chem_part = (
        f"{chem['formula']} | {dopants}"
        if dopants and dopants != "Undoped"
        else chem["formula"]
    )
    return f"{chem_part}\nE = {E} eV/Å | FWHM = {gamma:.4f} cm$^{{-1}}$"

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
    charges = load_bec_charges(OUTCAR_MAP, nat)

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
        title = build_plot_title(chem, E, args.gamma * THZCM1)
        plot_raman_spectrum(
            dat_file = out_csv,
            out_png  = out_png,
            gamma    = args.gamma * THZCM1,
            freq_min = args.freq_min,
            freq_max = args.freq_max,
            sticks   = not args.no_sticks,
            n_labels = args.n_labels,
            title    = title,
        )
        print(f"Written : {out_png}")

    print("\nDone.")

if __name__ == '__main__':
    main()
