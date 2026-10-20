#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_raman.py -- Overlay Raman spectra from multiple RASCBEC CSV files.

Reads the 4-column CSV format produced by the updated RASCBEC_phonopy.py
(Mode, Freq_cm-1, Activity, Irrep) and is backwards-compatible with the old
3-column format (no Irrep column).

Physical Background and Stokes Conversion:
------------------------------------------
The raw quantity output by RASCBEC is the Placzek isotropic powder Raman activity:

.. math::
    A_s = 45 \\bar{\\alpha}'^2 + 7 {\\gamma'}^2 \\propto \\left| \\frac{\\partial \\alpha}{\\partial Q_s} \\right|^2

In experimental Raman spectroscopy, the detector records the Stokes scattered
photon rate :math:`I_s(\\nu, T)`, which incorporates the harmonic zero-point
coordinate amplitude (:math:`1/\\nu_s`), the thermal Bose-Einstein phonon
population factor (:math:`n(\\nu_s, T) + 1`), and the dipole radiation term
(:math:`(\\nu_L - \\nu_s)^4`):

.. math::
    I_s(\\nu, T) \\propto \\frac{(\\nu_L - \\nu_s)^4}{\\nu_s}
    \\left[ \\frac{1}{1 - \\exp\\left(-\\frac{h c \\nu_s}{k_B T}\\right)} \\right] A_s

Why this matters for multi-composition comparisons:
1. In molecular-like framework compounds such as Na3PS4, the spectra are dominated
   by high-frequency P-S stretching modes (~420 cm^-1). Low-frequency cation
   motions (< 200 cm^-1) have negligible intrinsic activity, so comparing raw
   activity curves S(nu) looks qualitatively similar to experiment.
2. In polar perovskite solid-state electrolytes such as Lithium Lanthanum Titanate
   (LLTO: Li_{3x}La_{(2/3)-x}TiO3), low-frequency modes (< 300 cm^-1) involve
   large-amplitude translations of polarizable La/Li cations coupled to collective
   TiO6 tilts. Omitting the :math:`[n(\\nu_s, T) + 1] / \\nu_s` factor suppresses
   low-frequency intensity by 4x to 7x relative to high-frequency Ti-O stretches
   (500-600 cm^-1).
3. By default, this updated script converts all input CSV activities to physical
   Stokes scattered intensities at T = 300 K (632.8 nm laser line), directly
   matching experimental multi-composition waterfall comparisons. Raw activity
   comparison remains available via ``--raw-activity``.

Backwards Compatibility with Legacy CSVs:
-----------------------------------------
No changes to existing CSV files are needed. The script parses existing 3- or
4-column CSVs, checks for an optional 5th 'Intensity' column if present, and
computes the thermal Stokes weights on the fly if not already present.

Constraints and Layout:
-----------------------
- --offset > 0 automatically applies --normalize-each if no norm flag is given.
- Peak labels are shown ONLY in offset (waterfall) mode; default --n-labels 10.
  Peaks are selected by prominence (not raw height) and kept at least
  --min-label-sep (default 10) cm^-1 apart; a candidate inside that window of a
  stronger peak is merged into it (split-Lorentzian shoulder) and dropped.
- If no input files are given, every *.csv in --dir (default: cwd) is used.
- In --offset (waterfall) mode, trace labels are drawn on the right of each
  trace (no legend), mirroring plot_raman_waterfall.py; without --offset the
  same labels go into a standard legend box in the upper-right corner instead.
  Colours use the hue-sorted tab10 / turbo logic.

Usage:
    # Waterfall with Stokes conversion (T=300 K, default), all *.csv in cwd
    python compare_raman.py --offset 1.2

    # Comparing columnar vs rock-salt LLTO arrangements
    python compare_raman.py C-210-79_E0.02.csv R-616-59_E0.02.csv --offset 1.2

    # Variable temperature comparison
    python compare_raman.py *.csv --offset 1.2 --temperature 100
    python compare_raman.py *.csv --offset 1.2 --temperature 600

    # Legacy raw Raman activity S(nu) mode (no 1/nu or thermal factors)
    python compare_raman.py *.csv --offset 1.2 --raw-activity

    # Explicit global normalisation (preserves relative intensity between systems)
    python compare_raman.py *.csv --normalize --offset 1.2

    # Custom laser excitation wavelength (e.g. 632.8 nm HeNe)
    python compare_raman.py *.csv --offset 1.2 --laser-wl 632.8
"""

import argparse
import colorsys
import glob as _glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from plot_raman import (
    build_raman_title,
    formula_to_mathtext,
    select_labeled_peaks,
    select_title_line,
    STICK_A, STICK_LW, THZCM1, KB_CM1, DEFAULT_TEMP, DEFAULT_LASER_WL, ASPECT_RATIO, UWM_GOLD, UWM_GREY
)

# ---------------------------------------------------------------------------
# Defaults (aligned with plot_raman.py / plot_raman_waterfall.py)
# ---------------------------------------------------------------------------
UPPER_OFFSET_RATIO = 2.0      # Waterfall: upper padding = ratio x --offset (lower = 1 x --offset)

# ---------------------------------------------------------------------------
# Style (publication-ready rcParams; mirrors plot_raman_waterfall.py)
# ---------------------------------------------------------------------------

def set_style() -> None:
    """Publication-ready matplotlib rcParams (same as plot_raman_waterfall)."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.6,
        "legend.facecolor": "#cccccc",
        "legend.edgecolor": "#cccccc",
        "grid.alpha": 1.0,
        "grid.linestyle": "--",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "mathtext.default": "regular",
    })


# ---------------------------------------------------------------------------
# High-contrast colour selection (mirrors plot_raman_waterfall.py)
# ---------------------------------------------------------------------------

def _tab_by_hue():
    """Tableau tab10 colours sorted by HSV hue (rainbow order)."""
    cmap = plt.get_cmap("tab10")
    return sorted(
        (to_rgb(cmap(i / 9)) for i in range(10)),
        key=lambda rgb: colorsys.rgb_to_hsv(*rgb)[0],
    )


def pick_colors(n: int):
    """
    Pick ``n`` high-contrast colours in rainbow order from tab10.

    Indices are evenly spaced across the hue-sorted palette so contrast
    scales with series count. If n > 10, fall back to the ``turbo`` colormap.
    """
    if n <= 0:
        return []

    palette = _tab_by_hue()
    n_pal = len(palette)

    if n == 1:
        return [palette[n_pal // 2]]

    if n >= n_pal:
        cmap = plt.get_cmap("turbo")
        return [cmap(0.08 + 0.84 * i / (n - 1))[:3] for i in range(n)]

    idxs = [round(i * (n_pal - 1) / (n - 1)) for i in range(n)]
    out = []
    used = set()
    for j in idxs:
        k = int(j)
        while k in used and k < n_pal - 1:
            k += 1
        while k in used and k > 0:
            k -= 1
        used.add(k)
        out.append(palette[k])
    return out


# ---------------------------------------------------------------------------
# Physics conversion and Lorentzian broadening
# ---------------------------------------------------------------------------

def lorentzian(x: np.ndarray, x0: float, A: float, gamma: float) -> np.ndarray:
    """
    Area-preserving Lorentzian broadening of a delta-function stick.

    The integrated area under the curve equals A (the mode weight),
    independent of gamma:

    .. math::
        L(x) = \\frac{1}{\\pi} \\frac{A (\\gamma / 2)}{(x - x_0)^2 + (\\gamma / 2)^2}

    Parameters
    ----------
    x : np.ndarray
        Frequency evaluation grid in cm^-1.
    x0 : float
        Peak centre in cm^-1.
    A : float
        Mode intensity / activity stick value.
    gamma : float
        Full width at half maximum (FWHM) in cm^-1.

    Returns
    -------
    np.ndarray
        Broadened Lorentzian intensity on the grid x.
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
    Convert raw Raman activities to experimental Stokes scattered intensities.

    Accounts for the harmonic oscillator coordinate amplitude scaling (1 / nu),
    the thermal Bose-Einstein population factor [n(nu, T) + 1], and the dipole
    radiation factor (nu_L - nu)^4:

    .. math::
        I(\\nu_k, T) = A_k \\times \\left( \\frac{\\nu_L - \\nu_k}{\\nu_L} \\right)^4
                       \\times \\frac{1}{\\nu_k}
                       \\times \\frac{1}{1 - \\exp\\left(-\\frac{h c \\nu_k}{k_B T}\\right)}

    Parameters
    ----------
    freqs_cm1 : np.ndarray
        Phonon frequencies in cm^-1.
    activities : np.ndarray
        Mode activities from RASCBEC (Placzek powder average: 45*a^2 + 7*g^2).
    temperature : float, optional
        Sample temperature in Kelvin (default: 300.0 K).
    laser_wl_nm : float, optional
        Excitation laser wavelength in nm (default: 632.8 nm).
    freq_cutoff_cm1 : float, optional
        Minimum frequency threshold to avoid division by zero (default: 1.0 cm^-1).

    Returns
    -------
    np.ndarray
        Converted Stokes intensities.
    """
    omega_L = 1.0e7 / laser_wl_nm
    intensities = np.zeros_like(activities, dtype=float)

    for i, (f, act) in enumerate(zip(freqs_cm1, activities)):
        if act <= 0.0 or f <= freq_cutoff_cm1:
            continue

        x = f / (KB_CM1 * temperature)
        if x > 700.0:
            bose = 1.0
        else:
            bose = 1.0 / (1.0 - np.exp(-x))

        laser_term = ((omega_L - f) / omega_L) ** 4
        harmonic_term = 1.0 / f

        intensities[i] = act * laser_term * harmonic_term * bose

    return intensities


def build_spectrum(
    freqs_cm1: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Absolute Lorentzian-broadened spectrum on grid x.

    Every mode contributes to the spectrum, including the Lorentzian tails
    of modes whose centres lie outside the displayed frequency interval.
    """
    spectrum = np.zeros_like(x)
    for f, w in zip(freqs_cm1, weights):
        if w > 0:
            spectrum += lorentzian(x, f, w, gamma)
    return spectrum


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def expand_globs(patterns: List[str]) -> List[str]:
    """Expand shell wildcard patterns into file lists."""
    files = []
    for p in patterns:
        expanded = sorted(_glob.glob(p))
        files.extend(expanded if expanded else [p])
    return files


def load_csv(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Load a RASCBEC CSV file. Returns (freqs_cm1, activities, irrep_labels, meta).

    Handles both the legacy 3-column format (Mode, Freq_cm-1, Activity) and
    the 4-column format (Mode, Freq_cm-1, Activity, Irrep). If a 5th column
    exists (e.g. precomputed Stokes Intensity), activities are parsed from column 2
    and can be optionally overridden.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file.

    Returns
    -------
    freqs_cm1 : np.ndarray
        Mode frequencies in cm^-1.
    activities : np.ndarray
        Mode activities from CSV.
    irrep_labels : List[str]
        Mode irreducible representations (empty strings if absent).
    meta : dict
        Metadata dictionary parsed from comment headers (# Formula, # Dopants, etc.).
    """
    meta = {'formula': '', 'dopants': '', 'file_label': '', 'e_field': ''}
    freqs, acts, irreps = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s.startswith('# Formula:'):
                meta['formula'] = s.split(':', 1)[1].strip()
            elif s.startswith('# Dopants:'):
                meta['dopants'] = s.split(':', 1)[1].strip()
            elif s.startswith('# File_Label:'):
                meta['file_label'] = s.split(':', 1)[1].strip()
            elif s.startswith('# E_field:'):
                meta['e_field'] = s.split(':', 1)[1].strip()
            elif s.startswith('#') or not s:
                continue
            else:
                parts = s.split(',')
                freqs.append(float(parts[1]))
                acts.append(float(parts[2]))
                irreps.append(parts[3].strip() if len(parts) > 3 else '')
    return np.array(freqs, dtype=float), np.array(acts, dtype=float), irreps, meta


def filename_fallback(filepath: Union[str, Path]) -> Tuple[str, str]:
    """
    Extract (chem_tag, efield) from a chemistry-based filename as a fallback.

    Examples:
        Na3PS4_E0.02.csv -> ('Na3PS4', '0.02')
        C-210-79_E0.02.csv -> ('C-210-79', '0.02')
    """
    stem = Path(filepath).stem
    inner = stem[6:] if stem.startswith('raman_') else stem
    parts = inner.rsplit('_', 1)
    if len(parts) == 2:
        tag, last = parts
        efield = last.lstrip('Ee') if last[:1].lower() == 'e' else ''
        if efield:
            return tag, efield
    return inner, ''


def chem_label(meta: Dict, filepath: Union[str, Path]) -> str:
    """Chemical part of a trace label.

    When the spectrum is undoped, returns the formula as a bold subscripted
    mathtext string (e.g. ``$\\mathbf{Na}_{3}\\mathbf{P}\\mathbf{S}_{4}$``),
    matching how the figure title renders it; if the formula is missing,
    falls back to the filename tag. When a non-"Undoped" dopant string is
    present, returns ``[Dopants]`` (the main label loop replaces this with the
    subscript dopant-fraction legend).
    """
    formula = meta.get('formula', '')
    dopants = meta.get('dopants', '')
    if dopants and dopants != "Undoped":
        return f"[{dopants}]"
    if formula:
        return formula_to_mathtext(formula)
    return filename_fallback(filepath)[0]


def get_efield(meta: Dict, filepath: Union[str, Path]) -> str:
    """Return E-field string from metadata or filename fallback."""
    return meta.get('e_field', '') or filename_fallback(filepath)[1]


# Subscript letters for dopant fractions, shared with the title builder so a
# trace label and the formula title agree on which letter is which dopant.
DOPANT_LETTERS = "xyzwuv"


def dopant_fraction_label(meta: Dict) -> str:
    """Mathtext dopant-fraction legend for a trace label.

    Parses the ``"Sp=val Sp=val ..."``, ``dopants`` metadata and renders
    ``$x_{Ca}=0.125$, $y_{Cl}=0.0625$`` (the letter is subscripted on the
    dopant element, then its value). Returns ``''`` when there are no dopants.
    """
    letters = DOPANT_LETTERS
    pairs = []
    for entry in (meta.get('dopants', '') or '').split():
        if '=' not in entry:
            continue
        sp, val = entry.split('=', 1)
        try:
            pairs.append((sp, float(val)))
        except ValueError:
            continue
    if not pairs:
        return ''
    return "\n".join(
        f"$\\mathbf{{{letters[i]}}}_{{{sp}}}={val:.2f}$" for i, (sp, val) in enumerate(pairs))
        # f"${letters[i]}={val:.2f}$" for i, (sp, val) in enumerate(pairs))


def build_legend_labels(all_meta: List[Dict], files: List[str]) -> Tuple[List[str], bool]:
    """Build trace labels, appending the E-field when chemistry labels collide.

    The E-field suffix is only added when two or more spectra share the same
    chemistry label; it does not guarantee the final labels are unique (if the
    E-fields also coincide, the labels remain identical).
    """
    chem_labels = [chem_label(m, f) for m, f in zip(all_meta, files)]
    efields = [get_efield(m, f) for m, f in zip(all_meta, files)]
    efield_in_legend = len(set(chem_labels)) < len(chem_labels)
    if efield_in_legend:
        labels = [f"{c} | E={e} eV/Å" if e else c for c, e in zip(chem_labels, efields)]
    else:
        labels = chem_labels
    return labels, efield_in_legend


def build_output_name(all_meta: List[Dict], norm_mode: str, raw_activity: bool) -> str:
    """Generate output filename encoding formula, normalization, and physical mode."""
    formulae = [m.get('formula', '') for m in all_meta]
    unique_f = set(f for f in formulae if f)
    formula = list(unique_f)[0] if len(unique_f) == 1 else 'mixed'
    suffix = "activity" if raw_activity else "stokes"
    return f"compare_{formula}_{suffix}_{norm_mode}.png"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare Raman spectra from multiple RASCBEC CSV files with Stokes thermal conversion.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('files', nargs='*',
                   help='Input CSV files (shell glob patterns accepted). '
                        'If omitted, all *.csv in --dir are used.')
    p.add_argument('--dir', type=str, default='.',
                   help='Directory to search for *.csv input files when no '
                        'files are given on the command line (default: cwd).')
    p.add_argument('--gamma', type=float, default=0.25,
                   help='Lorentzian FWHM in THz (default: 0.25 THz); converted '
                        'to cm^-1 internally via THZCM1.')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm^-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm^-1 (default: auto)')

    norm = p.add_mutually_exclusive_group()
    norm.add_argument('--normalize', action='store_true',
                      help='Normalize all spectra to the global maximum '
                           '(preserves relative intensities between compositions)')
    norm.add_argument('--normalize-each', action='store_true',
                      help='Normalize each spectrum to its own maximum '
                           '(pure peak-shape / position comparison; default when --offset is given)')

    p.add_argument('--offset', type=float, default=0.0,
                   help='Lower padding below each spectrum for waterfall view '
                        f'(default: 0 = overlaid); upper padding above each '
                        f'spectrum is {UPPER_OFFSET_RATIO:g}x this value '
                        f'(see UPPER_OFFSET_RATIO). Automatically applies '
                        '--normalize-each unless --normalize is given.')
    p.add_argument('--sticks', action='store_true',
                   help='Overlay stick (bar) spectrum for each composition')
    p.add_argument('--n-labels', type=int, default=10,
                   help='Number of peak frequency labels per spectrum in '
                        'waterfall mode (default: 10; set 0 to disable).')
    p.add_argument('--min-label-sep', type=float, default=10.0,
                   help='Minimum separation in cm^-1 between labelled peaks '
                        'in waterfall mode (default: 10; set 0 to disable).')
    p.add_argument('--prominence', type=float, default=0.01,
                   help='Prominence threshold (fraction of peak-to-baseline '
                        'span) used to detect labellable peaks (default: 0.01)')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Custom legend labels, one per file in order')
    p.add_argument('--temperature', type=float, default=DEFAULT_TEMP,
                   help='Sample temperature in K for Bose-Einstein factor (default: 300.0 K)')
    p.add_argument('--laser-wl', type=float, default=DEFAULT_LASER_WL,
                   help='Excitation laser wavelength in nm '
                        f'(default: {DEFAULT_LASER_WL:g} nm)')
    p.add_argument('--raw-activity', action='store_true',
                   help='Plot raw Raman activities S(nu) without 1/nu or thermal Bose factors')
    p.add_argument('--title', type=str, default=None,
                   help='Custom figure title (default: auto-generated). Pass '
                        "'line1' or 'line2' to show only the formula line or "
                        "only the parameters line of the auto-generated title, "
                        'or an empty string --title "" to omit the title.')
    p.add_argument('--grid', choices=['both', 'x', 'y', 'none'], default='x',
                   help='Grid visibility: x (default), both, y, or none')
    p.add_argument('--figsize', type=float, nargs='+', default=None,
                   metavar=('WIDTH', 'HEIGHT'),
                   help='Override figure size in inches (default: auto from '
                        'data range / offset). Pass one value for a square '
                        'figure (e.g. --figsize 8) or two for WIDTH HEIGHT '
                        '(e.g. --figsize 6 8).')
    p.add_argument('--x-line', type=float, nargs='+', default=None, metavar='X',
                   help='Draw a vertical reference line at each of these '
                        'x-axis (wavenumber) values in cm-1 (e.g. --x-line 420 600)')
    p.add_argument('--y-line', type=float, nargs='+', default=None, metavar='Y',
                   help='Draw a horizontal reference line at each of these '
                        'y-axis values (e.g. --y-line 1.0 0.5)')
    p.add_argument('--out', default=None,
                   help='Output PNG filename (default: compare_<formula>_<mode>_<norm>.png)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.files:
        files = expand_globs(args.files)
    else:
        files = sorted(_glob.glob(str(Path(args.dir) / "*.csv")))
        if not files:
            raise SystemExit(
                "No input CSV files found. "
                "Pass file paths or use --dir to point at a folder of *.csv.")

    waterfall_mode = args.offset > 0
    if waterfall_mode and not args.normalize and not args.normalize_each:
        args.normalize_each = True
        print("  (auto: --normalize-each applied because --offset > 0)")

    show_labels = args.n_labels > 0 and waterfall_mode
    if args.n_labels > 0 and not waterfall_mode:
        print("  (note: peak labels are shown only in --offset waterfall mode)")

    mode_str = "Raw Activity S(nu)" if args.raw_activity else f"Stokes Intensity (T={args.temperature:g} K, {args.laser_wl:g} nm)"
    print(f"Comparing {len(files)} spectra [{mode_str}]:")
    for f in files:
        print(f"  {Path(f).name}")

    all_freqs, all_acts, all_irreps, all_meta = [], [], [], []
    for f in files:
        freqs, acts, irreps, meta = load_csv(f)
        all_freqs.append(freqs)
        all_acts.append(acts)
        all_irreps.append(irreps)
        all_meta.append(meta)

    gamma = args.gamma * THZCM1

    # Full range spanning every mode (with a 50 cm^-1 margin each side). The
    # broadened spectrum is always built and normalized on this full range, so
    # a custom --freq-min/--freq-max zoom window can never truncate the
    # spectrum (or a Lorentzian peak/tail) before the max is determined.
    full_min = min(fq.min() for fq in all_freqs) - 50.0
    full_max = max(fq.max() for fq in all_freqs) + 50.0
    x = np.linspace(full_min, full_max, 10000)

    # Displayed (zoomed) window
    freq_min = args.freq_min
    freq_max = args.freq_max or full_max
    if not (full_min <= freq_min < freq_max <= full_max):
        print(f"  (note: zoom window [{freq_min:g}, {freq_max:g}] falls outside the "
              f"data range [{full_min:g}, {full_max:g}]; plotting the full range instead)")
        freq_min, freq_max = full_min, full_max

    # Compute mode weights (Stokes intensity or raw activity)
    all_weights = []
    for freqs, acts in zip(all_freqs, all_acts):
        if args.raw_activity:
            weights = np.copy(acts)
        else:
            weights = activity_to_stokes_intensity(
                freqs, acts,
                temperature=args.temperature,
                laser_wl_nm=args.laser_wl,
            )
        all_weights.append(weights)

    # Build broadened spectra from calculated weights
    spectra = [
        build_spectrum(freqs, weights, x, gamma)
        for freqs, weights in zip(all_freqs, all_weights)
    ]

    # Normalisation is always based on the FULL spectrum, even when the plot
    # is zoomed to a custom --freq-min/--freq-max window, so zooming never
    # rescales the traces.
    if args.normalize_each:
        spectra = [s / s.max() if s.max() > 0 else s for s in spectra]
        y_label = 'Intensity (normalized per spectrum)' if not args.raw_activity else 'Activity (normalized per spectrum)'
        norm_mode = 'enorm'
    elif args.normalize:
        gmax = max(s.max() for s in spectra)
        spectra = [s / gmax if gmax > 0 else s for s in spectra]
        y_label = 'Intensity (normalized to global max)' if not args.raw_activity else 'Activity (normalized to global max)'
        norm_mode = 'gnorm'
    else:
        y_label = 'Stokes Intensity (arb. units, absolute)' if not args.raw_activity else 'Raman Activity S(ν) (absolute)'
        norm_mode = 'abs'

    # Mask to the displayed frequency window; all window-dependent quantities
    # (trace-label position, peak labels, sticks) use vis, while normalisation
    # above already used the full spectrum.
    in_window = (x >= freq_min) & (x <= freq_max)
    vis = [s[in_window] for s in spectra]

    auto_labels, efield_in_legend = build_legend_labels(all_meta, files)
    if args.labels:
        for i, lbl in enumerate(args.labels):
            if i < len(auto_labels):
                auto_labels[i] = lbl

    trace_labels = []
    for idx, (meta, lbl) in enumerate(zip(all_meta, auto_labels)):
        # An explicit --labels entry takes precedence over the auto-generated
        # chemistry/E-field label so users can disambiguate identical spectra.
        user_lbl = args.labels[idx] if args.labels and idx < len(args.labels) else None
        if user_lbl:
            trace_labels.append(user_lbl)
            continue
        # lbl already carries the subscripted mathtext formula for undoped
        # spectra (via chem_label) and "[Dopants]" for doped ones.
        fl = lbl if lbl else (meta.get('formula', '') or meta.get('file_label', ''))
        if efield_in_legend:
            e = meta.get('e_field', '') or filename_fallback(files[idx])[1]
            if e and f" | E={e} eV/Å" not in fl:
                fl = f"{fl} | E={e} eV/Å"
        dop_legend = dopant_fraction_label(meta)
        if dop_legend:
            fl = f"{dop_legend}"
        trace_labels.append(fl)

    if len(trace_labels) > 1 and len(set(trace_labels)) == 1:
        print("  (warning: all spectra share the identical trace label "
              f"'{trace_labels[0]}' -- the legend cannot distinguish them. "
              "Pass distinct --labels to disambiguate.)")

    # Figure title
    # --title line1/line2 selects one line of the auto-generated title; it is
    # not itself a custom title string.
    title_line = "both"
    if args.title in ("line1", "line2"):
        title_line = args.title
        args.title = None
    if args.title == "":
        plot_title = None
    elif args.title is not None:
        plot_title = args.title
    else:
        formulae = [m.get('formula', '') for m in all_meta]
        unique_f = set(f for f in formulae if f)
        efields = [get_efield(m, f) for m, f in zip(all_meta, files)]
        unique_ef = set(e for e in efields if e)

        if len(unique_f) == 1:
            formula = list(unique_f)[0]
            # A dopant matrix shares one base host formula but varies the
            # dopant set per file (e.g. an undoped parent plus several doped
            # compositions). Read only file[0]'s dopants would drop the species
            # carried by the other files, so union the species across all files
            # (first-seen order) to recover the full symbolic formula
            # (e.g. Na_{3-2x-y}Ca_xPS_{4-y}Cl_y). The placeholder values are
            # unused: the title builder renders only the species, and the
            # per-file fraction legend is turned off.
            dopant_species = []
            for m in all_meta:
                for entry in re.split(r'\s+', (m.get('dopants') or '').strip()):
                    if '=' in entry:
                        sp = entry.split('=', 1)[0]
                        if sp not in dopant_species:
                            dopant_species.append(sp)
            dopants = " ".join(f"{sp}=0" for sp in dopant_species)
            E_val = float(list(unique_ef)[0]) if len(unique_ef) == 1 else None
            plot_title = build_raman_title(
                chem={"formula": formula, "dopants": dopants},
                E=E_val,
                gamma=gamma,
                temperature=args.temperature,
                raw_activity=args.raw_activity,
                suffix="Raman Comparison",
                show_legend=False,
            )
        else:
            title_line2_parts = []
            if len(unique_ef) == 1:
                title_line2_parts.append(f"E = {list(unique_ef)[0]} eV/Å")
            title_line2_parts.append(f"FWHM = {gamma:.2f} cm$^{{-1}}$")
            if not args.raw_activity:
                title_line2_parts.append(f"T = {args.temperature:g} K")
            else:
                title_line2_parts.append("S(ν)")
            plot_title = "Raman Spectra Comparison\n" + " | ".join(title_line2_parts)

    set_style()
    n = len(files)

    # Waterfall baselines: --offset is the LOWER padding below each trace
    # baseline; the UPPER padding above the baseline is UPPER_OFFSET_RATIO x
    # offset, so --offset 0.5 gives 0.5 below and 1.0 above each baseline.
    if waterfall_mode:
        # Per-trace cell (baseline to next baseline): lower + upper padding.
        cell = args.offset * (1.0 + UPPER_OFFSET_RATIO)
        baselines = [args.offset + i * cell for i in range(n)]
        # Stack extent: lower padding under the first trace + n cells.
        stack_top = args.offset + n * cell
    else:
        baselines = [0.0] * n
        stack_top = max(s.max() for s in vis) if vis else 1.0

    fig_w = max(4, 6 * (freq_max - freq_min) / (full_max - full_min))
    fig_h = max(ASPECT_RATIO * fig_w, args.offset * stack_top) if waterfall_mode else ASPECT_RATIO * fig_w

    # --figsize overrides the auto-computed figure size (1 value = square).
    if args.figsize is not None:
        if len(args.figsize) == 1:
            fig_w = fig_h = float(args.figsize[0])
        elif len(args.figsize) == 2:
            fig_w, fig_h = (float(v) for v in args.figsize)
        else:
            raise SystemExit("--figsize takes 1 (square) or 2 (WIDTH HEIGHT) values.")

    fig, ax = plt.subplots(figsize = (fig_w, fig_h))

    colors = pick_colors(n)

    for i, (spectrum, freqs, weights, irreps, label, color) in enumerate(
            zip(spectra, all_freqs, all_weights, all_irreps, trace_labels, colors)):

        baseline = baselines[i]
        has_irreps = any(lbl != '' for lbl in irreps)

        # In non-waterfall mode the trace label goes into the legend box; in
        # waterfall mode it is drawn inline to the right of each trace.
        ax.plot(x[in_window], spectrum[in_window] + baseline, lw=1.6, color=color, zorder=4,
                label=label if not waterfall_mode else None)

        if waterfall_mode:
            # Trace label sits 20% of the way up from this trace's minimum
            # within the selected frequency window.
            label_dy = float(vis[i].min()) + 0.2 * float(vis[i].max() - vis[i].min()) if args.offset > 0 else 0.2
            ax.text(
                freq_max - 0.2 * (freq_max - freq_min),
                baseline + label_dy,
                f"{label}",
                color=color,
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="left",
                zorder=5,
                bbox=dict(facecolor="#cccccc", edgecolor="#cccccc", alpha=0.4, pad=1.0),
            )

        if args.sticks and freqs.size:
            stick_scale = 0.6 / weights.max() if weights.max() > 0 else 1.0
            for f_val, w_val in zip(freqs, weights):
                if w_val > 0 and freq_min <= f_val <= freq_max:
                    ax.vlines(f_val, baseline, baseline + w_val * stick_scale,
                              color=color, lw=STICK_LW, alpha=STICK_A, zorder=3)

        if show_labels:
            # Prominence-based peak selection with split-Lorentzian-shoulder
            # merging (see select_labeled_peaks); indices are into vis[i].
            top_peaks = select_labeled_peaks(
                vis[i], x[in_window],
                n_labels=args.n_labels,
                min_label_sep=args.min_label_sep,
                prominence_frac=args.prominence,
            )
            for pk in top_peaks:
                peak_freq = x[in_window][pk]
                # if has_irreps and len(freqs) > 0:
                #     closest_idx = int(np.argmin(np.abs(freqs - peak_freq)))
                #     irrep = irreps[closest_idx]
                #     label_text = f"{irrep}\n{peak_freq:.0f}"
                # else:
                label_text = f"{peak_freq:.1f}"
                ax.annotate(
                    label_text,
                    xy=(peak_freq, vis[i][pk] + baseline),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, color=color, linespacing=1.0)

    if not waterfall_mode and n > 1:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
    ax.set_ylabel(y_label)

    # Grid visibility (default 'x' preserves the original x-grid-only look).
    if args.grid == "both":
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
    elif args.grid == "x":
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
    elif args.grid == "y":
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    else:  # "none"
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    # Optional vertical reference lines at user-selected x-axis (wavenumber) values.
    if args.x_line:
        for xv in args.x_line:
            ax.axvline(xv, color="UWM_GREY", lw=0.8, ls="--", alpha=0.8, zorder=2)

    # Optional horizontal reference lines at user-selected y-axis values.
    if args.y_line:
        for yv in args.y_line:
            ax.axhline(yv, color="UWM_GREY", lw=0.8, ls="--", alpha=0.8, zorder=2)

    # --title line1/line2 only affects the auto-generated title; a custom
    # --title is shown as-is, and --title "" still omits the title entirely.
    if plot_title is not None and args.title is None:
        plot_title = select_title_line(plot_title, title_line)
    ax.set_title(plot_title)

    fig.tight_layout()
    out_png = args.out or build_output_name(all_meta, norm_mode, args.raw_activity)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
