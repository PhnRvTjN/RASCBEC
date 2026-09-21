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
    python plot_raman.py                          # auto-uses the only CSV in cwd (T=300K, 632.8nm)
    python plot_raman.py --dat LLTO_E0.02.csv
    python plot_raman.py --dat LLTO_E0.02.csv --temperature 280 --laser-wl 532
    python plot_raman.py --dat LLTO_E0.02.csv --raw-activity     # legacy S(nu) mode
    python plot_raman.py --dat LLTO_E0.02.csv --gamma 0.25
    python plot_raman.py --dat LLTO_E0.02.csv --freq-min 50 --freq-max 900
    python plot_raman.py --dat LLTO_E0.02.csv --no-sticks --n-labels 10
    python plot_raman.py --help
"""

import argparse
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
DEFAULT_TEMP = 300.0          # Default sample temperature in Kelvin
DEFAULT_LASER_WL = 632.8      # Default excitation laser wavelength in nm


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


def _build_plot_title_from_csv(
    dat_file: Union[str, Path],
    gamma: float,
    raw_activity: bool = False,
    temperature: float = DEFAULT_TEMP,
) -> str:
    """
    Rebuild the informative plot title from CSV metadata and physical parameters.

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
    dopants = meta['dopants']
    E = meta['E_field']

    if not formula:
        return Path(dat_file).stem

    chem_part = (
        f"{formula} | {dopants}"
        if dopants and dopants != "Undoped"
        else formula
    )

    t_part = f" | T = {temperature:g} K" if not raw_activity else " | S(ν)"
    if E is not None:
        return f"{chem_part}\nE = {E:g} eV/Å | FWHM = {gamma:.2f} cm$^{{-1}}${t_part}"
    return f"{chem_part} | FWHM = {gamma:.2f} cm$^{{-1}}${t_part}"


def plot_raman_spectrum(
    dat_file: Union[str, Path],
    out_png: Optional[Union[str, Path]] = None,
    gamma: float = 8.3391,
    freq_min: float = 0.0,
    freq_max: Optional[float] = None,
    sticks: bool = True,
    n_labels: int = 10,
    title: Optional[str] = None,
    raw_activity: bool = False,
    temperature: float = DEFAULT_TEMP,
    laser_wl: float = DEFAULT_LASER_WL,
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
    title : str, optional
        Custom plot title (default: auto-generated from CSV metadata).
    raw_activity : bool, optional
        If True, plot raw Raman activity S(nu) instead of Stokes intensity I(nu, T).
    temperature : float, optional
        Sample temperature in Kelvin for Bose-Einstein factor (default: 300.0 K).
    laser_wl : float, optional
        Laser excitation wavelength in nm for (nu_L - nu)^4 factor (default: 532.0 nm).
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

    # Label prominent peaks with overlap guard
    if n_labels > 0:
        min_dist = max(int(5000 / (freq_max - freq_min) * 10), 1)
        peaks, _ = find_peaks(spectrum, height=0.05, distance=min_dist)
        top_peaks = sorted(peaks, key=lambda p: -spectrum[p])[:n_labels]
        top_peaks = sorted(top_peaks)

        dx_guard = float(freq_max - freq_min) * 0.01
        labeled: List[float] = []

        for pk in top_peaks:
            peak_freq = float(x[pk])
            peak_y = float(spectrum[pk])

            if any(abs(peak_freq - ux) < dx_guard for ux in labeled):
                continue

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
            labeled.append(peak_freq)

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(0, y_top)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)")
    ax.grid(axis="both")

    plot_title = title if title is not None else _build_plot_title_from_csv(
        dat_file, gamma, raw_activity=raw_activity, temperature=temperature
    )
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
                   help='Sample temperature in K for Bose-Einstein factor (default: 300.0 K)')
    p.add_argument('--laser-wl', type=float, default=DEFAULT_LASER_WL,
                   help='Excitation laser wavelength in nm (default: 532.0 nm)')
    p.add_argument('--raw-activity', action='store_true',
                   help='Plot raw Raman activity S(nu) without 1/nu or thermal Bose factors')
    p.add_argument('--no-sticks', action='store_true',
                   help='Hide stick spectrum')
    p.add_argument('--n-labels', type=int, default=10,
                   help='Number of peak labels to annotate (default: 10)')
    p.add_argument('--title', type=str, default=None,
                   help='Custom plot title (default: CSV filename stem)')

    args = p.parse_args()

    dat_file = args.dat if args.dat is not None else str(resolve_default_csv())

    plot_raman_spectrum(
        dat_file=dat_file,
        out_png=args.out,
        gamma=args.gamma * THZCM1,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        sticks=not args.no_sticks,
        n_labels=args.n_labels,
        title=args.title,
        raw_activity=args.raw_activity,
        temperature=args.temperature,
        laser_wl=args.laser_wl,
    )
