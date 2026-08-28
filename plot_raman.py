#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Raman spectrum plotter for RASCBEC output.

Also importable:
from plot_raman import plot_raman_spectrum

When called from RASCBEC_phonopy.py, dat_file and out_png are always passed
explicitly and use chemistry-based names (e.g. Na3PS4_Ca12-C06_E0.02.csv).

When run standalone, --dat is optional: if omitted, the single CSV file in
the current working directory is used (exits with an error if none or
multiple CSVs are found, telling you to pass --dat explicitly).
--out defaults to the same stem with .png extension.

The input CSV may have 3 data columns (Mode, Freq_cm-1, Activity) for
backwards compatibility with CSVs produced before the irrep patch, or
4 columns (Mode, Freq_cm-1, Activity, Irrep) as written by the updated
RASCBEC_phonopy.py. When Irrep labels are present, peak annotations
show the symmetry label on the line above the frequency value.

Usage:
python plot_raman.py   # auto-uses the only CSV in the cwd
python plot_raman.py --dat Na3PS4_Ca12-Cl06_E0.02.csv
python plot_raman.py --dat Na3PS4_E0.02.csv --gamma 0.2
python plot_raman.py --dat Na3PS4_E0.02.csv --freq-min 50 --freq-max 600
python plot_raman.py --dat Na3PS4_E0.02.csv --no-sticks --n-labels 5
python plot_raman.py --help
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Defaults (aligned with plot_raman_waterfall.py)
# ---------------------------------------------------------------------------

STICK_A = 0.6    # stick alpha (pre-broadened calculated Raman activity)
STICK_LW = 0.8   # stick linewidth
THZCM1 = 33.356409519815204  # 1 THz in cm^-1 (c = 2.99792458e10 cm/s)


# ---------------------------------------------------------------------------
# Style (publication-ready rcParams; mirrors plot_raman_waterfall.py)
# ---------------------------------------------------------------------------


def set_style() -> None:
    """Publication-ready matplotlib rcParams (same as plot_raman_waterfall)."""
    plt.rcParams.update(
        {
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
        }
    )


def resolve_default_csv() -> Path:
    """
    Auto-resolve the input CSV from the current working directory.

    Globs ``*.csv`` in the cwd and requires exactly one match. Exits with a
    helpful message when zero or multiple files match, telling the user to
    pass ``--dat`` explicitly.
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



def lorentzian(x, x0, A, gamma):
    """
    Area-preserving Lorentzian broadening of a delta-function stick.

    The integrated area under the curve equals A (the Raman activity),
    independent of gamma. This is the standard lineshape used by
    phonopy-spectroscopy and other solid-state spectroscopy packages.

    L(x) = (1/pi) * [A * (gamma/2)] / [(x - x0)^2 + (gamma/2)^2]

    Parameters
    ----------
    x : array-like
        Frequency grid (cm$^{-1}$).
    x0 : float
        Peak centre (cm$^{-1}$).
    A : float
        Raman activity (stick intensity); becomes the integrated area.
    gamma : float
        Full width at half maximum (cm$^{-1}$).

    Returns
    -------
    ndarray of same shape as ``x``.
    """
    half_gamma = gamma / 2.0
    return (A * half_gamma) / (np.pi * ((x - x0)**2 + half_gamma**2))


def _load_raman_csv(dat_file):
    """
    Parse a RASCBEC CSV file.

    Handles both the old 3-column format (Mode, Freq_cm-1, Activity) and the
    new 4-column format (Mode, Freq_cm-1, Activity, Irrep). Comment lines
    beginning with '#' are skipped.

    Parameters
    ----------
    dat_file : str -- path to the CSV file

    Returns
    -------
    freqs_cm1 : ndarray
    activities : ndarray
    irrep_labels : list[str] -- empty strings when Irrep column is absent
    """
    freqs, acts, irreps = [], [], []
    with open(dat_file) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('#') or not s:
                continue
            parts = s.split(',')
            freqs.append(float(parts[1]))
            acts.append(float(parts[2]))
            irreps.append(parts[3].strip() if len(parts) > 3 else '')
    return np.array(freqs), np.array(acts), irreps


def _read_csv_metadata(dat_file):
    """
    Read metadata from commented CSV header lines written by RASCBEC_phonopy.py.

    Recognized keys: Formula, Dopants, File_Label, E_field. Older CSVs that
    predate the File_Label header line simply omit that key.

    Returns
    -------
    meta : dict with keys 'formula', 'dopants', 'file_label', 'E_field'
    """
    meta = {'formula': None, 'dopants': '', 'file_label': '', 'E_field': None}

    with open(dat_file) as fh:
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


def _build_plot_title_from_csv(dat_file, gamma):
    """
    Rebuild the informative title from CSV metadata when available.
    Falls back to the CSV stem for older files.
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

    if E is not None:
        return f"{chem_part}\nE = {E:g} eV/Å | FWHM = {gamma:.4f} cm$^{{-1}}$"
    return f"{chem_part} | FWHM = {gamma:.4f} cm$^{{-1}}$"


def plot_raman_spectrum(dat_file,
                        out_png=None,
                        gamma=0.25,
                        freq_min=0.0,
                        freq_max=None,
                        sticks=True,
                        n_labels=10,
                        title=None):
    """
    Read a RASCBEC CSV output, apply Lorentzian broadening, and save a plot.

    The CSV is expected to have comment lines beginning with '#' (including
    the metadata header written by RASCBEC_phonopy.py) followed by
    comma-separated rows of: mode_index, freq_cm-1, activity[, irrep].

    Every mode contributes to the spectrum evaluated on the requested grid,
    including the Lorentzian tails of modes whose centres lie outside the
    displayed frequency interval. This matches the spectrum construction in
    plot_raman_waterfall.py. Stick marks remain limited to centres visible in
    the plot interval.

    When Irrep labels are present in the CSV, each peak annotation shows
    the symmetry label (e.g. A1, E, B2) on the line above the frequency
    in cm-1. When they are absent the annotation shows the frequency only,
    preserving backwards compatibility.

    The nearest mode by frequency is used to assign an irrep label to each
    detected peak, so annotations remain correct even when broadening shifts
    the apparent peak position slightly away from the stick frequency.

    Parameters
    ----------
    dat_file : str -- input CSV file path (required)
    out_png : str -- output PNG path (default: dat_file stem + .png)
    gamma : float -- Lorentzian FWHM in cm-1 (default: 5.0)
    freq_min : float -- lower x-axis limit in cm-1 (default: 0.0)
    freq_max : float -- upper x-axis limit in cm-1 (default: auto)
    sticks : bool -- overlay stick spectrum (default: True)
    n_labels : int -- number of peak labels to annotate (default: 10)
    title : str -- plot title (default: CSV filename stem)
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
        freq_max = freqs_cm1.max() + 50.0
    if freq_max <= freq_min:
        raise ValueError(
            f"freq_max must exceed freq_min; got {freq_min} and {freq_max}"
        )

    x = np.linspace(freq_min, freq_max, 5000)
    spectrum = np.zeros_like(x)

    # Sum every mode on the evaluation grid. Do not omit modes just because
    # their centres lie outside the displayed range: their Lorentzian tails
    # still contribute inside it.
    for f, A in zip(freqs_cm1, activities):
        if A > 0:
            spectrum += lorentzian(x, f, A, gamma)

    if spectrum.max() > 0:
        spectrum /= spectrum.max()
    ymax = spectrum.max() if spectrum.size else 1.0
    top_pad = 0.14 if n_labels > 0 else 0.06
    y_top = max(1.0, ymax) * (1.0 + top_pad)

    fig, ax = plt.subplots(figsize=(8, 6))

    # post-broadened continuum (tab:blue total line)
    ax.fill_between(x, spectrum, alpha=0.2, color='tab:blue', zorder=2)
    ax.plot(x, spectrum, color='tab:blue', lw=1.6, label="broadened S(ν)", zorder=4)

    # pre-broadened calculated stick activities (tab:red)
    if sticks and freqs_cm1.size:
        stick_scale = 0.6 / activities.max() if activities.max() > 0 else 1.0
        for f, A in zip(freqs_cm1, activities):
            if A > 0 and freq_min <= f <= freq_max:
                ax.vlines(f, 0.0, A * stick_scale,
                          color='tab:red', lw=STICK_LW, alpha=STICK_A, zorder=3)
        ax.legend(handles=[
            Line2D([0], [0], color='tab:blue', lw=1.6, label="broadened S(ν)"),
            Line2D([0], [0], color='tab:red', lw=STICK_LW, alpha=0.8,
                   label="calculated sticks"),
        ], loc='upper right')
    else:
        ax.legend(loc='upper right')

    # Label prominent peaks above the total spectrum (black/bold, matching
    # plot_raman_waterfall's label_peaks_on_ax style), with an overlap guard
    # so closely-spaced labels are suppressed.
    if n_labels > 0:
        min_dist = max(int(5000 / (freq_max - freq_min) * 10), 1)
        peaks, _ = find_peaks(spectrum, height=0.05, distance=min_dist)
        top_peaks = sorted(peaks, key=lambda p: -spectrum[p])[:n_labels]
        # left-to-right so the overlap guard works in order
        top_peaks = sorted(top_peaks)

        # A 5-6 digit number at fontsize ~7 occupies roughly 1% of the x-span;
        # use that as the minimum separation between labelled peaks.
        dx_guard = float(freq_max - freq_min) * 0.01
        labeled: list = []

        for pk in top_peaks:
            peak_freq = float(x[pk])
            peak_y = float(spectrum[pk])
            # Skip if any previously labelled peak is too close
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
    plot_title = title if title is not None else _build_plot_title_from_csv(dat_file, gamma)
    ax.set_title(plot_title)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Plot Raman spectrum from RASCBEC CSV output.'
    )
    p.add_argument('--dat', default=None,
                   help='Input CSV file (e.g. Na3PS4_Ca0.125_E0.02.csv). '
                        'If omitted, the single CSV in the current directory is used '
                        '(errors if none or multiple are found).')
    p.add_argument('--out', default=None,
                   help='Output PNG filename (default: dat stem + .png)')
    p.add_argument('--gamma', type=float, default=0.25,
                   help='Lorentzian FWHM in THz (default: 0.25 THz); converted '
                        'to cm-1 internally via THZCM1.')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm-1 (default: auto)')
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
    )
