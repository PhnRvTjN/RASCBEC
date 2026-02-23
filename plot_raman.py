#!/usr/bin/env python3

"""
Standalone Raman spectrum plotter for RASCBEC output.

Also importable: from plot_raman import plot_raman_spectrum

Output naming:
    When defaults are used, filenames are derived from the directory ONE level
    above cwd.  Example: running from ~/calcs/06-12/9-RASCBEC uses "06-12".

Usage:
    python plot_raman.py                                    # all defaults
    python plot_raman.py --gamma 15                         # wider broadening
    python plot_raman.py --freq-min 50 --freq-max 600       # zoom window
    python plot_raman.py --dat raman_06-12.csv --out compare.png
    python plot_raman.py --no-sticks --n-labels 5
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


def plot_raman_spectrum(dat_file=None,
                        out_png=None,
                        gamma=10.0,
                        freq_min=0.0,
                        freq_max=None,
                        sticks=True,
                        n_labels=15,
                        title=None):
    """
    Read a RASCBEC CSV output, apply Lorentzian broadening, save plot.

    Parameters
    ----------
    dat_file : str  Input CSV file (default: raman_<parent>.csv)
    out_png  : str  Output PNG path (default: raman_<parent>.png)
    gamma    : float  Lorentzian FWHM in cm-1 (default: 10.0)
    freq_min : float  Lower x-axis limit in cm-1 (default: 0.0)
    freq_max : float  Upper x-axis limit in cm-1 (default: auto)
    sticks   : bool   Overlay stick spectrum (default: True)
    n_labels : int    Number of peak labels to annotate (default: 15)
    title    : str    Custom plot title (default: auto-generated from parent dir)
    """
    # Resolve filenames from parent directory name if not explicitly given
    parent_name = Path.cwd().parent.name
    dat_file = dat_file or f"raman_{parent_name}.csv"
    out_png  = out_png  or f"raman_{parent_name}.png"

    # Read CSV: columns are mode, freq_cm-1, activity
    # delimiter=',' handles the comma-separated format; comments='#' skips header
    data       = np.loadtxt(dat_file, delimiter=',', comments='#')
    freqs_cm1  = data[:, 1]
    activities = data[:, 2]

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
        for pk in peaks[:n_labels]:
            ax.annotate(f'{x[pk]:.0f}',
                        xy=(x[pk], spectrum[pk]),
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', fontsize=7, color='navy')

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax.set_title(
        title or f'Raman Spectrum — {parent_name} | Lorentzian FWHM = {gamma} cm$^{{-1}}$',
        fontsize=12)
    ax.tick_params(direction='in', top=True, right=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':

    parent_name = Path.cwd().parent.name

    p = argparse.ArgumentParser(
        description='Plot Raman spectrum from RASCBEC CSV output.')
    p.add_argument('--dat', default=None,
                   help='Input CSV file (default: raman_<parent>.csv)')
    p.add_argument('--out', default=None,
                   help='Output PNG filename (default: raman_<parent>.png)')
    p.add_argument('--gamma', type=float, default=10.0,
                   help='Lorentzian FWHM in cm-1 (default: 10.0)')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm-1 (default: auto)')
    p.add_argument('--no-sticks', action='store_true',
                   help='Hide stick spectrum (default: show sticks)')
    p.add_argument('--n-labels', type=int, default=15,
                   help='Number of peak labels to annotate (default: 15)')
    p.add_argument('--title', type=str, default=None,
                   help='Custom plot title (default: auto from parent dir name)')
    args = p.parse_args()

    plot_raman_spectrum(
        dat_file = args.dat,
        out_png  = args.out,
        gamma    = args.gamma,
        freq_min = args.freq_min,
        freq_max = args.freq_max,
        sticks   = not args.no_sticks,
        n_labels = args.n_labels,
        title    = args.title or parent_name,
    )
