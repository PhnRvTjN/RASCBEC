#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Raman spectrum plotter for RASCBEC output.

Also importable:
    from plot_raman import plot_raman_spectrum

Physical Formulation and Methodological Background:
---------------------------------------------------
The RASCBEC method evaluates Raman activities :math:`A_s` from first principles
by computing finite-difference electric-field derivatives of the Born effective
charge (BEC) tensors, contracted with mass-weighted Gamma-point phonon eigenvectors:

.. math::
    R_{s,ij} = \\frac{1}{4\\pi \\epsilon_0} \\sum_{t} \\sum_{k} \\frac{\\partial Z^*_{t,ik}}{\\partial \\mathcal{E}_j} \\frac{e_{t,k}^{(s)}}{\\sqrt{M_t}}

The isotropic powder-averaged Raman activity is given by Placzek's invariants:

.. math::
    A_s = 45 \\bar{\\alpha}'^2 + 7 {\\gamma'}^2

In Raman spectroscopy, the detector records the Stokes scattered photon flux
:math:`I_s(\\nu, T)`, not the raw quantum activity :math:`A_s`. Transition
matrix elements introduce the harmonic zero-point coordinate amplitude (:math:`1/\\nu_s`),
the thermal phonon occupation (Bose-Einstein factor :math:`n(\\nu_s, T) + 1`),
and dipole radiation scattering (:math:`(\\nu_L - \\nu_s)^4`):

.. math::
    I_s(\\nu, T) \\propto \\frac{(\\nu_L - \\nu_s)^4}{\\nu_s}
    \\left[ \\frac{1}{1 - \\exp\\left(-\\frac{h c \\nu_s}{k_B T}\\right)} \\right] A_s

Key Physical Relevance (e.g., LLTO vs. Na3PS4):
-----------------------------------------------
1. In molecular-like framework compounds such as Na3PS4, the spectrum is dominated
   by high-frequency P-S stretching/breathing modes (~420 cm^-1). Low-frequency
   cation modes (< 200 cm^-1) have negligible intrinsic polarizability derivatives,
   so raw activity S(nu) plots appear qualitatively similar to experiment.
2. In polar perovskite solid-state electrolytes such as Lithium Lanthanum Titanate
   (LLTO: Li_{3x}La_{(2/3)-x}TiO3), low-frequency modes (< 300 cm^-1) involve large
   displacements of polarizable La/Li cations coupled to collective TiO6 tilts.
   Omitting the :math:`[n(\\nu_s, T) + 1] / \\nu_s` prefactor severely suppresses
   low-frequency intensity by 4x to 7x relative to high-frequency Ti-O stretches
   (500-600 cm^-1).
3. By default, this updated module evaluates the full temperature-dependent Stokes
   scattered intensity :math:`I(\\nu, T)` at T = 300 K using a 632.8 nm laser line,
   accurately reproducing laboratory intensity distributions. The unweighted raw
   Raman activity :math:`S(\\nu)` can still be plotted using ``--raw-activity``.

Usage:
    python plot_raman.py                          # auto-uses the only CSV in cwd (T=350K, 632.8nm)
    python plot_raman.py --dat LLTO_E0.02.csv
    python plot_raman.py --dat LLTO_E0.02.csv --temperature 280 --laser-wl 532
    python plot_raman.py --dat LLTO_E0.02.csv --raw-activity     # legacy S(nu) mode
    python plot_raman.py --dat LLTO_E0.02.csv --gamma 0.25
    python plot_raman.py --dat LLTO_E0.02.csv --freq-min 50 --freq-max 900
    python plot_raman.py --dat LLTO_E0.02.csv --no-sticks --n-labels 10
    python plot_raman.py --help
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Physical Constants and Plot Defaults
# ---------------------------------------------------------------------------

STICK_A = 0.6                 # stick alpha (pre-broadened stick intensity)
STICK_LW = 0.8                # stick linewidth
THZCM1 = 33.356409519815204   # 1 THz in cm^-1 (c = 2.99792458e10 cm/s)
KB_CM1 = 0.69503476           # Boltzmann constant kB in cm^-1 / K
DEFAULT_TEMP = 350.0          # Default sample temperature in Kelvin
DEFAULT_LASER_WL = 632.8      # Default excitation laser wavelength in nm
ASPECT_RATIO = 1.618          # Golden ratio for figure width/height when auto-scaling
UWM_GOLD = "#ffbd00"        # Underwater Materials gold for titles / labels
UWM_GREY = "#666666"        # Underwater Materials grey for titles / labels


# ---------------------------------------------------------------------------
# Style (publication-ready rcParams; mirrors plot_raman_waterfall.py)
# ---------------------------------------------------------------------------

def set_style() -> None:
    """Publication-ready matplotlib rcParams (same as plot_raman_waterfall)."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.6,
        "legend.facecolor": "#cccccc",
        "legend.edgecolor": "#cccccc",
        "grid.alpha": 0.8,
        "grid.linestyle": "--",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "mathtext.default": "regular",
    })


def resolve_default_csv() -> Path:
    """
    Auto-resolve the input CSV from the current working directory.

    Globs ``*.csv`` in the cwd and requires exactly one match. Exits with a
    helpful message when zero or multiple files match, telling the user to
    pass ``--dat`` explicitly.

    Returns
    -------
    Path
        Resolved path to the single CSV file in the working directory.
    """
    files = sorted(Path.cwd().glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No CSV files found in the working directory ({Path.cwd()}).\n"
            "Pass the input file explicitly with --dat <file>."
        )
    if len(files) > 1:
        listing = "\n  ".join(p.name for p in files)
        raise SystemExit(
            f"Multiple CSV files found in {Path.cwd()}:\n  {listing}\n"
            "Cannot guess which one to use - pass the input explicitly "
            "with --dat <file>."
        )
    return files[0]


def lorentzian(x: np.ndarray, x0: float, A: float, gamma: float) -> np.ndarray:
    """
    Area-preserving Lorentzian broadening of a delta-function stick.

    The integrated area under the curve equals A (the mode intensity or activity),
    independent of gamma. This is the standard lineshape used by phonopy-spectroscopy
    and other solid-state spectroscopy packages:

    .. math::
        L(x) = \\frac{1}{\\pi} \\frac{A (\\gamma / 2)}{(x - x_0)^2 + (\\gamma / 2)^2}

    Parameters
    ----------
    x : np.ndarray
        Frequency evaluation grid in cm^-1.
    x0 : float
        Peak center frequency in cm^-1.
    A : float
        Mode intensity / activity stick value (becomes integrated area).
    gamma : float
        Full width at half maximum (FWHM) in cm^-1.

    Returns
    -------
    np.ndarray
        Broadened Lorentzian intensity on the x grid.
    """
    half_gamma = gamma / 2.0
    return (A * half_gamma) / (np.pi * ((x - x0)**2 + half_gamma**2))


def activity_to_stokes_intensity(
    freqs_cm1: np.ndarray,
    activities: np.ndarray,
    temperature: float = DEFAULT_TEMP,
    laser_wl_nm: float = DEFAULT_LASER_WL,
    freq_cutoff_cm1: float = 1.0,
) -> np.ndarray:
    """
    Convert raw quantum Raman activities to experimental Stokes scattered intensities.

    Accounts for the harmonic oscillator coordinate amplitude scaling (1 / nu),
    the thermal Bose-Einstein population factor [n(nu, T) + 1], and the dipole
    radiation factor (nu_L - nu)^4:

    .. math::
        I(\\nu_k, T) = A_k \\times \\left( \\frac{\\nu_L - \\nu_k}{\\nu_L} \\right)^4
                       \\times \\frac{1}{\\nu_k}
                       \\times \\frac{1}{1 - \\exp\\left(-\\frac{h c \\nu_k}{k_B T}\\right)}

    Modes with frequency below ``freq_cutoff_cm1`` (such as acoustic or residual
    near-zero modes) are assigned zero intensity to prevent 1/nu or Bose divergence.

    Parameters
    ----------
    freqs_cm1 : np.ndarray
        Phonon frequencies in cm^-1.
    activities : np.ndarray
        Calculated RASCBEC Raman activities (isotropic powder averaged: 45*a^2 + 7*g^2).
    temperature : float, optional
        Sample temperature in Kelvin (default: 280.0 K).
    laser_wl_nm : float, optional
        Excitation laser wavelength in nm (default: 632.8 nm).
    freq_cutoff_cm1 : float, optional
        Minimum frequency threshold to avoid division by zero (default: 1.0 cm^-1).

    Returns
    -------
    np.ndarray
        Converted Stokes intensities ready for broadening and plotting.
    """
    omega_L = 1.0e7 / laser_wl_nm  # Laser wavenumber in cm^-1
    intensities = np.zeros_like(activities, dtype=float)

    for i, (f, act) in enumerate(zip(freqs_cm1, activities)):
        if act <= 0.0 or f <= freq_cutoff_cm1:
            continue

        # Bose-Einstein occupation factor for Stokes scattering: n(nu, T) + 1
        x = f / (KB_CM1 * temperature)
        if x > 700.0:
            bose = 1.0
        else:
            bose = 1.0 / (1.0 - np.exp(-x))

        # Dipole scattering and classical coordinate amplitude penalty
        laser_term = ((omega_L - f) / omega_L) ** 4
        harmonic_term = 1.0 / f

        intensities[i] = act * laser_term * harmonic_term * bose

    return intensities


def _load_raman_csv(dat_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parse a RASCBEC CSV file.

    Handles both the old 3-column format (Mode, Freq_cm-1, Activity) and the
    new 4-column format (Mode, Freq_cm-1, Activity, Irrep). Comment lines
    beginning with '#' are skipped.

    Parameters
    ----------
    dat_file : str or Path
        Path to the CSV file.

    Returns
    -------
    freqs_cm1 : np.ndarray
        Mode frequencies in cm^-1.
    activities : np.ndarray
        Mode Raman activities from RASCBEC.
    irrep_labels : List[str]
        Mode irreducible representations (empty strings if not present).
    """
    freqs, acts, irreps = [], [], []
    with open(dat_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('#') or not s:
                continue
            parts = s.split(',')
            freqs.append(float(parts[1]))
            acts.append(float(parts[2]))
            irreps.append(parts[3].strip() if len(parts) > 3 else '')
    return np.array(freqs, dtype=float), np.array(acts, dtype=float), irreps


def _read_csv_metadata(dat_file: Union[str, Path]) -> Dict[str, Optional[Union[str, float]]]:
    """
    Read metadata from commented CSV header lines written by RASCBEC_phonopy.py.

    Recognized keys: Formula, Dopants, File_Label, E_field.

    Parameters
    ----------
    dat_file : str or Path
        Path to the CSV file.

    Returns
    -------
    dict
        Metadata dictionary with keys 'formula', 'dopants', 'file_label', 'E_field'.
    """
    meta: Dict[str, Optional[Union[str, float]]] = {
        'formula': None,
        'dopants': '',
        'file_label': '',
        'E_field': None,
    }

    with open(dat_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith('#'):
                break

            s = s[1:].strip()
            if s.startswith('Formula:'):
                meta['formula'] = s.split(':', 1)[1].strip()
            elif s.startswith('Dopants:'):
                meta['dopants'] = s.split(':', 1)[1].strip()
            elif s.startswith('File_Label:'):
                meta['file_label'] = s.split(':', 1)[1].strip()
            elif s.startswith('E_field:'):
                val = s.split(':', 1)[1].strip()
                try:
                    meta['E_field'] = float(val)
                except ValueError:
                    meta['E_field'] = val

    return meta


# ---------------------------------------------------------------------------
# Raman plot title builder (shared by RASCBEC_phonopy.py & standalone plotting)
# ---------------------------------------------------------------------------

# Substitution rules (per formula unit) for dopant species.
# Keys: dopant species. Values: dict of {displaced host species → coefficient}.
# A positive coefficient means the dopant displaces that many host atoms
# (e.g. Ca → 2× Na).  A negative coefficient means the dopant adds to the
# host count (rare; e.g. Ga → +1 Li).
DOPANT_SUBSTITUTIONS = {
    'Ca': {'Na': 2},
    'Cl': {'Na': 1, 'S': 1},
    'Ga': {'Ti': 1, 'Li': -1},
    'Y':  {'La': 1},
}


def _parse_coefficient(token: str):
    """Convert a formula coefficient token to ``int`` (preferred) or ``float``.

    An empty token means the element has no explicit coefficient (→ 1).
    Integer-valued strings stay ``int`` so rendering is unchanged
    (``'3'`` → ``3``, not ``3.0``); fractional values (e.g. ``'0.31'`` from a
    basis-normalized host formula) become ``float``.
    """
    if not token:
        return 1
    value = float(token)
    return int(value) if value.is_integer() else value


def _build_symbolic_formula_mathtext(
    host_formula: str,
    dopant_species: List[str],
    dopant_fractions: Dict[str, float],
) -> str:
    """
    Build a matplotlib mathtext formula string with bold element symbols and
    subscript letters for the dopant fractions.

    The host formula (e.g. ``'Na3PS4'``) is the *base* (undoped) host
    composition: RASCBEC reconstructs it by adding the dopant displacements
    back onto the doped host counts, so its coefficients are already the true
    base stoichiometry.  For each host element displaced by a dopant the
    symbolic ± terms are appended directly (e.g. ``Na_{3} → Na_{3-2x-y}``);
    host elements that are not displaced render their bare base coefficient.
    Each dopant is rendered immediately after the last host element it
    displaces, so a dopant sits next to the site it substitutes (e.g.
    ``Na_{3-2x-y}Ca_{x}``).

    Element symbols use ``\\mathbf`` to match the bold title weight; the
    ``axes.titleweight`` rcParam does not propagate into mathtext.

    Parameters
    ----------
    host_formula     : str  -- base host formula, e.g. 'Na3PS4'
    dopant_species   : list of str -- e.g. ['Ca', 'Cl']
    dopant_fractions : dict of str → float -- e.g. {'Ca': 0.125, 'Cl': 0.0625}

    Returns
    -------
    str
        e.g. ``\\mathbf{Na}_{3-2x-y}\\mathbf{Ca}_{x}\\mathbf{P}\\mathbf{S}_{4-y}\\mathbf{Cl}_{y}``
    """
    host_pairs = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', host_formula)
    host_pairs = [(el, _parse_coefficient(co)) for el, co in host_pairs]

    letters = "xyzwuv"
    dopant_letter = {sp: letters[i] for i, sp in enumerate(dopant_species)}

    # For each host element, collect which dopants displace it and by what coeff.
    displaced_by = {}  # host_el → list of (letter, coefficient)
    for sp in dopant_species:
        rules = DOPANT_SUBSTITUTIONS.get(sp, {})
        letter = dopant_letter[sp]
        for host_el, coeff in rules.items():
            displaced_by.setdefault(host_el, []).append((letter, coeff))

    # Each dopant is rendered after the LAST host element (in host order) it displaces.
    host_index = {el: i for i, (el, _) in enumerate(host_pairs)}
    placed = {}        # host index → list of dopant species
    trailing = []      # dopants that displace no known host element
    for sp in dopant_species:
        hosts = [h for h in DOPANT_SUBSTITUTIONS.get(sp, {}) if h in host_index]
        if not hosts:
            trailing.append(sp)
        else:
            placed.setdefault(max(host_index[h] for h in hosts), []).append(sp)

    def _signed(c: int, l: str) -> str:
        """Build a ±c·l term; omit coefficient 1; handle negative (addition)."""
        if c > 0:
            return f"-{c}{l}" if c != 1 else f"-{l}"
        c = -c
        return f"+{c}{l}" if c != 1 else f"+{l}"

    parts = []
    for i, (el, base) in enumerate(host_pairs):
        sub = displaced_by.get(el, [])
        if not sub:
            # Braces are required whenever the subscript is more than one
            # character (e.g. '0.31', '12'); a bare ``_`` binds only the next
            # single character, leaving the rest at baseline. Braces are a no-op
            # for single-digit integers, so this is safe to always apply.
            sub_str = f"_{{{base:g}}}" if base != 1 else ""
            parts.append(f"\\mathbf{{{el}}}{sub_str}")
        else:
            expr = f"{base:g}" + "".join(_signed(c, l) for l, c in sub)
            parts.append(f"\\mathbf{{{el}}}_{{{expr}}}")
        for sp in placed.get(i, []):
            parts.append(f"\\mathbf{{{sp}}}_{{{dopant_letter[sp]}}}")

    parts.extend(f"\\mathbf{{{sp}}}_{{{dopant_letter[sp]}}}" for sp in trailing)

    return "".join(parts)


def formula_to_mathtext(formula: str, dopants: str = "") -> str:
    """Render a single composition as a bold subscripted matplotlib mathtext string.

    Wraps the same builder used by :func:`build_raman_title` so that trace
    labels, legend entries, and the figure title all agree on how a formula
    is written (e.g. ``'Na3PS4'`` → ``$\\mathbf{Na}_{3}\\mathbf{P}\\mathbf{S}_{4}$``;
    ``'Na3PS4'`` with ``'Ca=0.125 Cl=0.0625'`` → the symbolic
    ``Na_{3-2x-y}Ca_{x}PS_{4-y}Cl_{y}``).

    Parameters
    ----------
    formula : str
        Base host formula, e.g. ``'Na3PS3.5O0.5'``.
    dopants : str, optional
        Dopant descriptor ``"Sp=val Sp=val ..."`` or ``""``/``"Undoped"``.

    Returns
    -------
    str
        A ``$...$``-delimited mathtext string. Empty string if ``formula`` is empty.
    """
    if not formula:
        return ""
    pairs = []
    for entry in re.split(r'\s+', (dopants or "").strip()):
        if '=' in entry:
            sp, val = entry.split('=', 1)
            try:
                pairs.append((sp, float(val)))
            except ValueError:
                pass
    return f"${_build_symbolic_formula_mathtext(formula, [sp for sp, _ in pairs], {sp: v for sp, v in pairs})}$"


def select_title_line(title: str, which: str = "both") -> str:
    """Return the full title, or only its first (line1) or second (line2) line.

    A two-line Raman title is ``line1\\nline2`` where line1 is the chemical
    formula line and line2 is the run-parameter line (E, FWHM, T / S(ν)).

    Parameters
    ----------
    title : str
        The full title string (may or may not contain a line break).
    which : str, optional
        One of ``'both'`` (default, unchanged), ``'line1'`` (formula line only),
        or ``'line2'`` (parameter line only).

    Returns
    -------
    str
        The selected line. Titles without a line break are returned unchanged
        regardless of ``which`` (so custom single-line titles are unaffected).
    """
    if which == "both" or "\n" not in title:
        return title
    lines = title.split("\n")
    if which == "line1":
        return lines[0]
    if which == "line2":
        return lines[-1] if len(lines) > 1 else title
    return title


def build_raman_title(
    chem: dict,
    E: Optional[float],
    gamma: float,
    temperature: float,
    raw_activity: bool = False,
    suffix: str = "Raman",
    show_legend: bool = True,
) -> str:
    """
    Build a two-line Raman plot title.

    Line 1: bold doped formula (mathtext) + ``suffix`` + dopant-fraction legend
            (e.g. ``Na₃₋₂ₓ₋ᵧCaₓPS₄₋ᵧClᵧ Raman | x=0.125 y=0.0625``).
    Line 2: run parameters — E, FWHM, T (or S(ν) when ``raw_activity=True``).

    Parameters
    ----------
    chem         : dict
        Metadata with keys ``'formula'`` (str) and ``'dopants'`` (str, e.g.
        ``"Ca=0.125 Cl=0.0625"`` or ``"Undoped"``).
    E            : float or None
        E-field magnitude in eV/Å.  Omitted from title when None.
    gamma        : float
        Lorentzian FWHM in cm^-1.
    temperature  : float
        Sample temperature in K.
    raw_activity : bool, optional
        If True, show ``S(ν)`` instead of temperature in line 2.
    suffix       : str, optional
        Text following the formula on line 1 (default ``"Raman"``; use
        ``"Raman Comparison"`` for multi-spectrum figures).
    show_legend  : bool, optional
        If True (default), append the ``x=…, y=…`` dopant-fraction legend to
        line 1. Set False when the per-trace labels already carry the
        dopant fractions (compare_raman.py waterfall).

    Returns
    -------
    str
        Two-line title string ready for matplotlib ``set_title``.
    """
    letters = "xyzwuv"

    # Parse dopant species and fractions from "Sp=val Sp=val ..." string.
    dopant_pairs = []
    for entry in re.split(r'\s+', (chem.get("dopants") or "").strip()):
        if '=' in entry:
            sp, val = entry.split('=', 1)
            try:
                dopant_pairs.append((sp, float(val)))
            except ValueError:
                pass

    # Line 1: formula + suffix (+ legend when enabled)
    if dopant_pairs:
        dop_species   = [sp for sp, _ in dopant_pairs]
        dop_fractions = {sp: val for sp, val in dopant_pairs}
        formula_str = _build_symbolic_formula_mathtext(
            chem['formula'], dop_species, dop_fractions)
        line1 = f"${formula_str}$  {suffix}"
        if show_legend:
            legend = " ".join(
                f"{letters[i]}={dop_fractions[sp]:.2f}"
                for i, sp in enumerate(dop_species))
            line1 += f"  |  {legend}"
    else:
        # Undoped: still render the host formula as subscripted mathtext
        # (e.g. Li3LaTiO3 -> Li_{3}LaTiO_{3}); previously fell back to
        # plain text and dropped the subscripts.
        formula = chem.get('formula') or ''
        if formula:
            line1 = (f"${_build_symbolic_formula_mathtext(formula, [], {})}$"
                     f"  {suffix}")
        else:
            line1 = f"  {suffix}"

    # Line 2: parameters
    line2_parts = []
    if E is not None:
        line2_parts.append(f"E = {E:g} eV/Å")
    line2_parts.append(f"FWHM = {gamma:.2f} cm$^{{-1}}$")
    if raw_activity:
        line2_parts.append("S(ν)")
    else:
        line2_parts.append(f"T = {temperature:g} K")
    line2 = "  |  ".join(line2_parts)

    return f"{line1}\n{line2}"


def _build_plot_title_from_csv(
    dat_file: Union[str, Path],
    gamma: float,
    raw_activity: bool = False,
    temperature: float = DEFAULT_TEMP,
) -> str:
    """
    Rebuild the informative plot title from CSV metadata and physical parameters.
    (Backward-compatible wrapper around :func:`build_raman_title`.)

    Parameters
    ----------
    dat_file : str or Path
        Path to the CSV file.
    gamma : float
        Lorentzian FWHM in cm^-1.
    raw_activity : bool, optional
        Whether raw activity S(nu) is being plotted.
    temperature : float, optional
        Sample temperature in Kelvin.

    Returns
    -------
    str
        Formatted plot title string.
    """
    meta = _read_csv_metadata(dat_file)
    formula = meta['formula']
    if not formula:
        return Path(dat_file).stem

    return build_raman_title(
        chem={"formula": formula, "dopants": meta['dopants']},
        E=meta['E_field'],
        gamma=gamma,
        temperature=temperature,
        raw_activity=raw_activity,
    )


def select_labeled_peaks(
    y: np.ndarray,
    freqs: np.ndarray,
    n_labels: int,
    min_label_sep: float,
    prominence_frac: float,
) -> List[int]:
    """
    Select up to ``n_labels`` peak indices for labelling, ranked by prominence.

    Peaks are found with ``scipy.signal.find_peaks`` using a *prominence*
    threshold (``prominence_frac`` of the trace's peak-to-baseline span) rather
    than a relative-height threshold. Prominence measures how far a peak rises
    above the local baseline it sits on, so a genuine peak riding the flank of
    a stronger one is still detected, while a tiny ripple is not.

    Candidates are then ranked by prominence and kept greedily so that no two
    kept peaks are within ``min_label_sep`` (cm^-1). A candidate that violates
    the separation rule is treated as a split Lorentzian shoulder and merged
    into the dominant nearby label (i.e. dropped), so a single broad peak never
    yields two neighbouring labels.

    Parameters
    ----------
    y : np.ndarray
        Broadened spectrum (1-D) over ``freqs``.
    freqs : np.ndarray
        Frequency grid in cm^-1 matching ``y``.
    n_labels : int
        Maximum number of peaks to label (0 selects none).
    min_label_sep : float
        Minimum cm^-1 separation between labelled peaks; smaller candidates are
        merged into the stronger nearby peak (0 disables merging).
    prominence_frac : float
        Prominence threshold as a fraction of the trace's peak-to-baseline span
        (e.g. 0.01 = 1%).

    Returns
    -------
    list of int
        Indices into ``y`` / ``freqs`` of the chosen peaks, strongest first.
    """
    if y.size == 0 or n_labels <= 0 or float(y.max()) <= 0.0:
        return []

    prominence = prominence_frac * (float(y.max()) - float(y.min()))

    # Distance in grid points (avoids find_peaks returning a peak and its
    # shoulder as two separate candidates); the label-separation rule below
    # does the final merging.
    grid_spacing = float(np.mean(np.diff(freqs))) if freqs.size > 1 else 1.0
    min_dist = (max(int(np.ceil(min_label_sep / grid_spacing)), 1)
                if min_label_sep > 0 else 1)

    peaks, props = find_peaks(y, prominence=prominence, distance=min_dist)
    if peaks.size == 0:
        return []

    prominences = props["prominences"]
    order = np.argsort(-prominences)
    chosen: List[int] = []
    for j in order:
        if len(chosen) >= n_labels:
            break
        pk = int(peaks[j])
        if all(abs(freqs[pk] - freqs[q]) >= min_label_sep for q in chosen):
            chosen.append(pk)
    return chosen


def plot_raman_spectrum(
    dat_file: Union[str, Path],
    out_png: Optional[Union[str, Path]] = None,
    gamma: float = 8.3391,
    freq_min: float = 0.0,
    freq_max: Optional[float] = None,
    sticks: bool = True,
    n_labels: int = 10,
    min_label_sep: float = 10.0,
    prominence: float = 0.01,
    title: Optional[str] = None,
    raw_activity: bool = False,
    temperature: float = DEFAULT_TEMP,
    laser_wl: float = DEFAULT_LASER_WL,
    grid: str = "both",
    x_lines: Optional[List[float]] = None,
    y_lines: Optional[List[float]] = None,
    title_line: str = "both",
) -> None:
    """
    Read a RASCBEC CSV output, apply thermal/Stokes conversion and Lorentzian broadening, and save a plot.

    By default, calculates the physically observed Stokes scattered Raman intensity
    :math:`I(\\nu, T) \\propto \\frac{(\\nu_L - \\nu)^4}{\\nu} [n(\\nu, T) + 1] S(\\nu)`.
    This correctly restores the intense low-frequency bands (< 300 cm^-1) observed in
    experimental measurements of perovskite solid electrolytes like LLTO. If
    ``raw_activity=True`` is passed, it reproduces the unweighted Raman activity
    :math:`S(\\nu)`.

    Parameters
    ----------
    dat_file : str or Path
        Input CSV file path (required).
    out_png : str or Path, optional
        Output PNG path (default: dat_file stem + .png).
    gamma : float, optional
        Lorentzian FWHM in cm^-1 (default: 8.3391 cm^-1, equivalent to 0.25 THz).
    freq_min : float, optional
        Lower x-axis limit in cm^-1 (default: 0.0).
    freq_max : float, optional
        Upper x-axis limit in cm^-1 (default: auto, max mode + 50 cm^-1).
    sticks : bool, optional
        Whether to overlay stick spectrum (default: True).
    n_labels : int, optional
        Number of peak labels to annotate (default: 10).
    min_label_sep : float, optional
        Minimum cm^-1 separation between labelled peaks; a candidate closer
        than this to a stronger labelled peak is merged into it (a split
        Lorentzian shoulder, default: 10; set 0 to disable).
    prominence : float, optional
        Prominence threshold as a fraction of the trace's peak-to-baseline span
        used to detect labellable peaks (default: 0.01 = 1%).
    title : str, optional
        Custom plot title (default: auto-generated from CSV metadata).
    raw_activity : bool, optional
        If True, plot raw Raman activity S(nu) instead of Stokes intensity I(nu, T).
    temperature : float, optional
        Sample temperature in Kelvin for Bose-Einstein factor (default: 350.0 K).
    laser_wl : float, optional
        Laser excitation wavelength in nm for (nu_L - nu)^4 factor (default: 532.0 nm).
    grid : str, optional
        Grid visibility: 'both' (default, x and y grids), 'x' (x grid only),
        'y' (y grid only), or 'none' (no grid).
    x_lines : list of float, optional
        If given, draw a vertical reference line at each of these x-axis
        (wavenumber) values in cm^-1 (e.g. ``[420.0]`` to mark a peak position).
    y_lines : list of float, optional
        If given, draw a horizontal reference line at each of these y-axis
        values (e.g. ``[1.0]`` to mark the normalised peak height).
    title_line : str, optional
        Which auto-generated title line to show: 'both' (default), 'line1', or
        'line2'. Custom ``title`` values are displayed as-is regardless.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive; got {gamma}")

    set_style()
    out_png = out_png or Path(dat_file).with_suffix('.png').name

    freqs_cm1, activities, irrep_labels = _load_raman_csv(dat_file)
    has_irreps = any(lbl != '' for lbl in irrep_labels)

    if freqs_cm1.size == 0:
        raise ValueError(f"No Raman modes found in {dat_file}")

    if freq_max is None:
        freq_max = float(freqs_cm1.max() + 50.0)
    if freq_max <= freq_min:
        raise ValueError(
            f"freq_max must exceed freq_min; got {freq_min} and {freq_max}"
        )

    # Convert quantum activities to detector intensities unless raw activity requested
    if raw_activity:
        mode_weights = np.copy(activities)
        spectrum_label = "broadened S(ν)"
    else:
        mode_weights = activity_to_stokes_intensity(
            freqs_cm1,
            activities,
            temperature=temperature,
            laser_wl_nm=laser_wl,
        )
        spectrum_label = "broadened I(ν)"

    x = np.linspace(freq_min, freq_max, 5000)
    spectrum = np.zeros_like(x)

    # Sum every mode on the evaluation grid. Lorentzian tails contribute even
    # if their centers fall outside the displayed frequency window.
    for f, w in zip(freqs_cm1, mode_weights):
        if w > 0:
            spectrum += lorentzian(x, f, w, gamma)

    if spectrum.max() > 0:
        spectrum /= spectrum.max()

    ymax = spectrum.max() if spectrum.size else 1.0
    top_pad = 0.14 if n_labels > 0 else 0.06
    y_top = max(1.0, ymax) * (1.0 + top_pad)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Broadened continuum
    ax.fill_between(x, spectrum, alpha=0.2, color='tab:blue', zorder=2)
    ax.plot(x, spectrum, color='tab:blue', lw=1.6, label=spectrum_label, zorder=4)

    # Calculated sticks
    if sticks and freqs_cm1.size:
        max_weight = mode_weights.max()
        stick_scale = 0.6 / max_weight if max_weight > 0 else 1.0
        for f, w in zip(freqs_cm1, mode_weights):
            if w > 0 and freq_min <= f <= freq_max:
                ax.vlines(
                    f, 0.0, w * stick_scale,
                    color='tab:red', lw=STICK_LW, alpha=STICK_A, zorder=3
                )
        ax.legend(handles=[
            Line2D([0], [0], color='tab:blue', lw=1.6, label=spectrum_label),
            Line2D([0], [0], color='tab:red', lw=STICK_LW, alpha=0.8,
                   label="calculated sticks"),
        ], loc='upper right')
    else:
        ax.legend(loc='upper right')

    # Label prominent peaks (prominence-ranked, split-Lorentzian shoulders
    # merged within min_label_sep -- see select_labeled_peaks).
    if n_labels > 0:
        for pk in select_labeled_peaks(
                spectrum, x,
                n_labels=n_labels,
                min_label_sep=min_label_sep,
                prominence_frac=prominence):
            peak_freq = float(x[pk])
            peak_y = float(spectrum[pk])

            if has_irreps and len(freqs_cm1) > 0:
                closest_idx = int(np.argmin(np.abs(freqs_cm1 - peak_freq)))
                irrep = irrep_labels[closest_idx]
                label_text = f"{irrep}\n{peak_freq:.1f}"
            else:
                label_text = f"{peak_freq:.1f}"

            label_y = min(peak_y + 0.035 * y_top, 0.93 * y_top)
            ax.text(
                peak_freq,
                label_y,
                label_text,
                color="tab:blue",
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="center",
                linespacing=1.0,
            )

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(0, y_top)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)")

    # Grid visibility (default 'both' preserves the original behaviour).
    if grid == "both":
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
    elif grid == "x":
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
    elif grid == "y":
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    else:  # "none"
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    # Optional vertical reference lines at user-selected x-axis (wavenumber) values.
    if x_lines:
        for xv in x_lines:
            ax.axvline(xv, color=UWM_GREY, lw=0.8, ls="--", alpha=0.7, zorder=2)

    # Optional horizontal reference lines at user-selected y-axis values.
    if y_lines:
        for yv in y_lines:
            ax.axhline(yv, color=UWM_GREY, lw=0.8, ls="--", alpha=0.7, zorder=2)

    if title is not None:
        plot_title = title
    else:
        plot_title = _build_plot_title_from_csv(
            dat_file, gamma, raw_activity=raw_activity, temperature=temperature
        )
        # --title line1/line2 selects one line of the auto-generated title; a
        # custom --title is shown as-is.
        plot_title = select_title_line(plot_title, title_line)
    ax.set_title(plot_title)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Plot Raman spectrum from RASCBEC CSV output with Stokes thermal conversion.'
    )
    p.add_argument('--dat', default=None,
                   help='Input CSV file (e.g. Li0.31La0.56TiO3_E0.02.csv). '
                        'If omitted, the single CSV in the current directory is used.')
    p.add_argument('--out', default=None,
                   help='Output PNG filename (default: dat stem + .png)')
    p.add_argument('--gamma', type=float, default=0.25,
                   help='Lorentzian FWHM in THz (default: 0.25 THz); converted '
                        'to cm^-1 internally via THZCM1.')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm-1 (default: auto)')
    p.add_argument('--temperature', type=float, default=DEFAULT_TEMP,
                   help='Sample temperature in K for Bose-Einstein factor (default: 350.0 K)')
    p.add_argument('--laser-wl', type=float, default=DEFAULT_LASER_WL,
                   help='Excitation laser wavelength in nm (default: 632.8 nm)')
    p.add_argument('--raw-activity', action='store_true',
                   help='Plot raw Raman activity S(nu) without 1/nu or thermal Bose factors')
    p.add_argument('--no-sticks', action='store_true',
                   help='Hide stick spectrum')
    p.add_argument('--n-labels', type=int, default=10,
                   help='Number of peak labels to annotate (default: 10)')
    p.add_argument('--min-label-sep', type=float, default=10.0,
                   help='Minimum cm^-1 separation between labelled peaks; a '
                        'closer candidate is merged into the stronger peak '
                        '(default: 10; set 0 to disable).')
    p.add_argument('--prominence', type=float, default=0.01,
                   help='Prominence threshold (fraction of peak-to-baseline '
                        'span) used to detect labellable peaks (default: 0.01)')
    p.add_argument('--title', type=str, default=None,
                   help='Custom plot title (default: auto-generated). Pass '
                        "'line1' or 'line2' to show only the formula line or "
                        "only the parameters line of the auto-generated title.")
    p.add_argument('--grid', choices=['both', 'x', 'y', 'none'], default='both',
                   help='Grid visibility: both (default), x, y, or none')
    p.add_argument('--x-line', type=float, nargs='+', default=None, metavar='X',
                   help='Draw a vertical reference line at each of these '
                        'x-axis (wavenumber) values in cm-1 (e.g. --x-line 420 600)')
    p.add_argument('--y-line', type=float, nargs='+', default=None, metavar='Y',
                   help='Draw a horizontal reference line at each of these '
                        'y-axis values (e.g. --y-line 1.0 0.5)')

    args = p.parse_args()

    dat_file = args.dat if args.dat is not None else str(resolve_default_csv())

    # --title line1/line2 selects a single line of the auto-generated title.
    title, title_line = args.title, 'both'
    if title in ('line1', 'line2'):
        title, title_line = None, title

    plot_raman_spectrum(
        dat_file=dat_file,
        out_png=args.out,
        gamma=args.gamma * THZCM1,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        sticks=not args.no_sticks,
        n_labels=args.n_labels,
        min_label_sep=args.min_label_sep,
        prominence=args.prominence,
        title=title,
        raw_activity=args.raw_activity,
        temperature=args.temperature,
        laser_wl=args.laser_wl,
        grid=args.grid,
        x_lines=args.x_line,
        y_lines=args.y_line,
        title_line=title_line,
    )
