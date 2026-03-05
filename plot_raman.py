#!/usr/bin/env python3

"""
Standalone Raman spectrum plotter for RASCBEC output.

Also importable:
    from plot_raman import plot_raman_spectrum

When called from RASCBEC_phonopy.py, dat_file and out_png are always passed
explicitly and use chemistry-based names (e.g. raman_Na3PS4_Ca0.125_E0.02.csv).

When run standalone, --dat must be provided; --out defaults to the same
stem with .png extension.

The input CSV may have 3 data columns (Mode, Freq_cm-1, Activity) for
backwards compatibility with CSVs produced before the irrep patch, or
4 columns (Mode, Freq_cm-1, Activity, Irrep) as written by the updated
RASCBEC_phonopy.py.  When Irrep labels are present, peak annotations
show the symmetry label on the line above the frequency value.

Usage:
    python plot_raman.py --dat raman_Na3PS4_Ca0.125_E0.02.csv
    python plot_raman.py --dat raman_Na3PS4_E0.02.csv --gamma 15
    python plot_raman.py --dat raman_Na3PS4_E0.02.csv --freq-min 50 --freq-max 600
    python plot_raman.py --dat raman_Na3PS4_E0.02.csv --no-sticks --n-labels 5
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


def lorentzian(x, x0, A, gamma):
    """Single Lorentzian peak: A*(gamma/2)^2 / ((x-x0)^2 + (gamma/2)^2)"""
    return A * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)


def _load_raman_csv(dat_file):
    """
    Parse a RASCBEC CSV file.

    Handles both the old 3-column format (Mode, Freq_cm-1, Activity) and the
    new 4-column format (Mode, Freq_cm-1, Activity, Irrep).  Comment lines
    beginning with '#' are skipped.

    Parameters
    ----------
    dat_file : str -- path to the CSV file

    Returns
    -------
    freqs_cm1    : ndarray
    activities   : ndarray
    irrep_labels : list[str]  -- empty strings when Irrep column is absent
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


def plot_raman_spectrum(dat_file,
                        out_png=None,
                        gamma=10.0,
                        freq_min=0.0,
                        freq_max=None,
                        sticks=True,
                        n_labels=15,
                        title=None):
    """
    Read a RASCBEC CSV output, apply Lorentzian broadening, and save a plot.

    The CSV is expected to have comment lines beginning with '#' (including
    the metadata header written by RASCBEC_phonopy.py) followed by
    comma-separated rows of: mode_index, freq_cm-1, activity[, irrep].

    When Irrep labels are present in the CSV, each peak annotation shows
    the symmetry label (e.g. A1, E, B2) on the line above the frequency
    in cm-1.  When they are absent the annotation shows the frequency only,
    preserving backwards compatibility.

    The nearest mode by frequency is used to assign an irrep label to each
    detected peak, so annotations remain correct even when broadening shifts
    the apparent peak position slightly away from the stick frequency.

    Parameters
    ----------
    dat_file : str   -- input CSV file path (required)
    out_png  : str   -- output PNG path (default: dat_file stem + .png)
    gamma    : float -- Lorentzian FWHM in cm-1 (default: 10.0)
    freq_min : float -- lower x-axis limit in cm-1 (default: 0.0)
    freq_max : float -- upper x-axis limit in cm-1 (default: auto)
    sticks   : bool  -- overlay stick spectrum (default: True)
    n_labels : int   -- number of peak labels to annotate (default: 15)
    title    : str   -- plot title (default: CSV filename stem)
    """
    out_png = out_png or Path(dat_file).with_suffix('.png').name

    freqs_cm1, activities, irrep_labels = _load_raman_csv(dat_file)
    has_irreps = any(lbl != '' for lbl in irrep_labels)

    if freq_max is None:
        freq_max = freqs_cm1.max() + 50.0

    x        = np.linspace(freq_min, freq_max, 5000)
    spectrum = np.zeros_like(x)

    for f, A in zip(freqs_cm1, activities):
        if A > 0 and freq_min <= f <= freq_max:
            spectrum += lorentzian(x, f, A, gamma)

    if spectrum.max() > 0:
        spectrum /= spectrum.max()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, spectrum, lw=1.5, color='steelblue', zorder=3)
    ax.fill_between(x, spectrum, alpha=0.15, color='steelblue', zorder=2)

    if sticks:
        stick_scale = 0.6 / activities.max() if activities.max() > 0 else 1.0
        for f, A in zip(freqs_cm1, activities):
            if A > 0 and freq_min <= f <= freq_max:
                ax.vlines(f, 0, A * stick_scale,
                          color='tomato', lw=0.6, alpha=0.5, zorder=1)
        ax.legend(handles=[
            Line2D([0], [0], color='steelblue', lw=1.5, label='Broadened'),
            Line2D([0], [0], color='tomato',    lw=0.8, alpha=0.7, label='Stick'),
        ], fontsize=9, loc='upper right', framealpha=0.7)

    if n_labels > 0:
        min_dist = max(int(5000 / (freq_max - freq_min) * 15), 1)
        peaks, _ = find_peaks(spectrum, height=0.05, distance=min_dist)
        top_peaks = sorted(peaks, key=lambda p: -spectrum[p])[:n_labels]
        for pk in top_peaks:
            peak_freq = x[pk]
            if has_irreps and len(freqs_cm1) > 0:
                closest_idx = int(np.argmin(np.abs(freqs_cm1 - peak_freq)))
                irrep       = irrep_labels[closest_idx]
                label_text  = f"{irrep}\n{peak_freq:.0f}"
            else:
                label_text  = f"{peak_freq:.0f}"
            ax.annotate(label_text,
                        xy=(peak_freq, spectrum[pk]),
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', fontsize=7, color='navy',
                        linespacing=1.3)

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax.set_title(title or Path(dat_file).stem, fontsize=12)
    ax.tick_params(direction='in', top=True, right=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Plot Raman spectrum from RASCBEC CSV output.')
    p.add_argument('--dat', required=True,
                   help='Input CSV file (e.g. raman_Na3PS4_Ca0.125_E0.02.csv)')
    p.add_argument('--out', default=None,
                   help='Output PNG filename (default: dat stem + .png)')
    p.add_argument('--gamma', type=float, default=10.0,
                   help='Lorentzian FWHM in cm-1 (default: 10.0)')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm-1 (default: auto)')
    p.add_argument('--no-sticks', action='store_true',
                   help='Hide stick spectrum')
    p.add_argument('--n-labels', type=int, default=15,
                   help='Number of peak labels to annotate (default: 15)')
    p.add_argument('--title', type=str, default=None,
                   help='Custom plot title (default: CSV filename stem)')
    args = p.parse_args()

    plot_raman_spectrum(
        dat_file = args.dat,
        out_png  = args.out,
        gamma    = args.gamma,
        freq_min = args.freq_min,
        freq_max = args.freq_max,
        sticks   = not args.no_sticks,
        n_labels = args.n_labels,
        title    = args.title,
    )
