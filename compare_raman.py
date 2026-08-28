#!/usr/bin/env python3

"""
compare_raman.py -- Overlay Raman spectra from multiple RASCBEC CSV files.

Reads the 4-column CSV format produced by the updated RASCBEC_phonopy.py
(Mode, Freq_cm-1, Activity, Irrep) and is backwards-compatible with the old
3-column format (no Irrep column).

In waterfall (--offset) mode, peak labels show the irrep symbol above the
frequency (e.g. A1 / 487) when Irrep data is present in the CSV.

Constraints
-----------
--offset > 0 automatically applies --normalize-each if no norm flag is given
Peak labels shown ONLY in offset (waterfall) mode; default --n-labels 10
If no input files are given, every *.csv in --dir (default: cwd) is used.
Trace labels are drawn on the left of each trace (no legend), mirroring
plot_raman_waterfall.py; colours use the same hue-sorted tab10 / turbo logic.

Output filename (when --out is not given)
-----------------------------------------
compare_<formula>_<norm>.png

formula : shared formula from CSV metadata (e.g. Na3PS4); 'mixed' if differ
norm    : abs | gnorm | enorm

e.g. compare_Na3PS4_enorm.png

Comparing same composition at different E-fields
-------------------------------------------------
When all CSVs share the same chemical label (formula + dopants), the E-field
is automatically appended to each trace label so spectra are distinguishable.
The E-field is then omitted from the title (it lives in the trace labels).

Trace labels and title are read from the CSV metadata header:
# Formula: Na3PS4
# Dopants: Ca=0.0625 Cl=0.0625
# File_Label: Ca01-Cl06
# E_field: 0.02

Usage:
    # Waterfall -- auto normalize-each + labels, all *.csv in cwd (typical)
    python compare_raman.py --offset 1.2

    # Waterfall, all *.csv in a specific folder
    python compare_raman.py --dir results/run1 --offset 1.2

    # Explicit files (still fine)
    python compare_raman.py *.csv --offset 1.2

    # Waterfall with explicit global normalisation
    python compare_raman.py *.csv --normalize --offset 1.2

    # Same composition, compare E-field strength
    python compare_raman.py Na3PS4_E0.01.csv Na3PS4_E0.02.csv Na3PS4_E0.05.csv --offset 1.2

    # Doped series
    python compare_raman.py Na3PS4_E0.02.csv Na3PS4_Ca0.125_Cl0.5_E0.02.csv --offset 1.2

    # Finer control
    python compare_raman.py *.csv --offset 1.2 --n-labels 5
    python compare_raman.py *.csv --offset 1.2 --n-labels 0   # disable labels
    python compare_raman.py *.csv --gamma 8 --freq-min 50 --freq-max 550
    python compare_raman.py *.csv --out my_fig.png
    python compare_raman.py --help
"""

import argparse
import colorsys
import glob as _glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Defaults (aligned with plot_raman.py / plot_raman_waterfall.py)
# ---------------------------------------------------------------------------
STICK_A = 0.6                              # stick alpha (pre-broadened Raman activity)
STICK_LW = 0.8                             # stick linewidth
THZCM1 = 33.356409519815204                # 1 THz in cm^-1 (c = 2.99792458e10 cm/s)


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
            "axes.titlesize": 12,
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
            "grid.alpha": 1.0,
            "grid.linestyle": "--",
            "grid.color": "#cccccc",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "mathtext.default": "regular",
        }
    )


# ---------------------------------------------------------------------------
# High-contrast colour selection (mirrors plot_raman_waterfall.py)
# tab10 sorted by HSV hue (rainbow order); evenly sampled so contrast scales
# with the number of traces. >10 traces fall back to the ``turbo`` colormap.
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
    scales with series count (comfortable up to ~9-10 entries). If n > 10,
    fall back to even samples on the continuous ``turbo`` colormap.
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
# Lorentzian broadening
# ---------------------------------------------------------------------------

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


def build_spectrum(freqs_cm1, activities, x, gamma):
    """
    Absolute Lorentzian-broadened spectrum on grid x.

    Every mode contributes to the spectrum, including the Lorentzian tails
    of modes whose centres lie outside the displayed frequency interval.
    This matches the spectrum construction in plot_raman.py.
    """
    spectrum = np.zeros_like(x)
    for f, A in zip(freqs_cm1, activities):
        if A > 0:
            spectrum += lorentzian(x, f, A, gamma)
    return spectrum

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def expand_globs(patterns):
    files = []
    for p in patterns:
        expanded = sorted(_glob.glob(p))
        files.extend(expanded if expanded else [p])
    return files

def load_csv(filepath):
    """
    Load a RASCBEC CSV.  Returns (freqs_cm1, activities, irrep_labels, meta).

    Handles both the old 3-column format (Mode, Freq_cm-1, Activity) and the
    new 4-column format (Mode, Freq_cm-1, Activity, Irrep).  irrep_labels
    contains empty strings for all modes when the Irrep column is absent.

    Metadata parsed from comment lines (all default to '' if absent):
    # Formula, # Dopants, # E_field
    """
    meta = {'formula': '', 'dopants': '', 'file_label': '', 'e_field': ''}
    freqs, acts, irreps = [], [], []
    with open(filepath) as f:
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
    return np.array(freqs), np.array(acts), irreps, meta

def filename_fallback(filepath):
    """
    Extract (chem_tag, efield) from a chemistry-based filename as a fallback
    when the CSV metadata header is absent.

    Examples
    --------
    Na3PS4_E0.02.csv              -> ('Na3PS4', '0.02')
    Na3PS4_Ca0.125_Cl0.5_E0.02.csv -> ('Na3PS4_Ca0.125_Cl0.5', '0.02')
    """
    stem  = Path(filepath).stem
    inner = stem[6:] if stem.startswith('raman_') else stem
    parts = inner.rsplit('_', 1)
    if len(parts) == 2:
        tag, last = parts
        efield = last.lstrip('Ee') if last[:1].lower() == 'e' else ''
        if efield:
            return tag, efield
    return inner, ''

def chem_label(meta, filepath):
    """
    Chemical part of the label: 'Na3PS4 [Ca(x=0.06)/Cl(x=0.12)]'
    Falls back to filename tag if metadata absent.
    """
    formula = meta.get('formula', '')
    dopants = meta.get('dopants', '')
    tag, _  = filename_fallback(filepath)
    base    = formula or tag
    return f"{base} [{dopants}]" if dopants else base

def get_efield(meta, filepath):
    """Return E-field string from metadata or filename fallback."""
    return meta.get('e_field', '') or filename_fallback(filepath)[1]

def build_legend_labels(all_meta, files):
    """
    Build legend labels with automatic E-field disambiguation.

    - Different compositions, same E -> 'Na3PS4 [Ca(x=0.06)/Cl(x=0.12)]' (E in title)
    - Same composition, different E  -> 'Na3PS4 [...] | E=0.02 eV/Å'   (E in legend)
    - Both differ                    -> 'Na3PS4 [...] | E=0.02 eV/Å'   (E in legend)

    Returns (legend_labels, efield_in_legend)
    efield_in_legend : bool -- True means E is in the legend, not the title
    """
    chem_labels      = [chem_label(m, f) for m, f in zip(all_meta, files)]
    efields          = [get_efield(m, f)  for m, f in zip(all_meta, files)]
    efield_in_legend = len(set(chem_labels)) < len(chem_labels)
    if efield_in_legend:
        labels = [f"{c} | E={e} eV/Å" if e else c
                  for c, e in zip(chem_labels, efields)]
    else:
        labels = chem_labels
    return labels, efield_in_legend

# ---------------------------------------------------------------------------
# Output filename
# ---------------------------------------------------------------------------

def build_output_name(all_meta, norm_mode):
    """
    compare_<formula>_<norm>.png

    formula : shared formula from metadata; 'mixed' if files differ
    norm    : abs | gnorm | enorm
    """
    formulae = [m.get('formula', '') for m in all_meta]
    unique_f = set(f for f in formulae if f)
    formula  = list(unique_f)[0] if len(unique_f) == 1 else 'mixed'
    return f"compare_{formula}_{norm_mode}.png"

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare Raman spectra from multiple RASCBEC CSV files.',
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
                        'to cm-1 internally via THZCM1.')
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm-1 (default: auto)')

    norm = p.add_mutually_exclusive_group()
    norm.add_argument('--normalize', action='store_true',
                      help='Normalise all spectra to the global maximum '
                           '(preserves relative intensities between compositions)')
    norm.add_argument('--normalize-each', action='store_true',
                      help='Normalise each spectrum to its own maximum '
                           '(pure peak-shape / position comparison; '
                           'default when --offset is given)')

    p.add_argument('--offset', type=float, default=0.0,
                   help='Vertical baseline offset between spectra for '
                        'waterfall view (default: 0 = overlaid). '
                        'Automatically applies --normalize-each unless '
                        '--normalize is explicitly given.')
    p.add_argument('--sticks', action='store_true',
                   help='Overlay stick (bar) spectrum for each composition')
    p.add_argument('--n-labels', type=int, default=10,
                   help='Number of peak frequency labels per spectrum in '
                        'waterfall mode (default: 10; set 0 to disable). '
                        'Suppressed in overlaid mode.')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Custom legend labels, one per file in order '
                        '(overrides auto-parsed labels)')
    p.add_argument('--title', type=str, default=None,
                   help='Custom figure title (default: auto-generated)')
    p.add_argument('--out', default=None,
                   help='Output PNG (default: compare_<formula>_<norm>.png)')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()

    if args.files:
        files = expand_globs(args.files)
    else:
        # No explicit files: auto-read every *.csv under --dir.
        files = sorted(_glob.glob(str(Path(args.dir) / "*.csv")))
    if not files:
        raise SystemExit(
            "No input CSV files found. "
            "Pass file paths or use --dir to point at a folder of *.csv.")

    # Resolve normalisation mode
    waterfall_mode = args.offset > 0
    if waterfall_mode and not args.normalize and not args.normalize_each:
        args.normalize_each = True
        print("  (auto: --normalize-each applied because --offset > 0)")

    show_labels = args.n_labels > 0 and waterfall_mode
    if args.n_labels > 0 and not waterfall_mode:
        print("  (note: peak labels are shown only in --offset waterfall mode)")

    print(f"Comparing {len(files)} spectra:")
    for f in files:
        print(f"  {Path(f).name}")

    # Load data
    all_freqs, all_acts, all_irreps, all_meta = [], [], [], []
    for f in files:
        freqs, acts, irreps, meta = load_csv(f)
        all_freqs.append(freqs)
        all_acts.append(acts)
        all_irreps.append(irreps)
        all_meta.append(meta)

    # Convert --gamma from THz to cm-1 (matching plot_raman.py)
    gamma = args.gamma * THZCM1

    # Frequency grid
    freq_min = args.freq_min
    freq_max = args.freq_max or (max(fq.max() for fq in all_freqs) + 50.0)
    x        = np.linspace(freq_min, freq_max, 5000)

    # Build absolute spectra
    spectra = [build_spectrum(freqs, acts, x, gamma)
               for freqs, acts in zip(all_freqs, all_acts)]

    # Apply normalisation
    if args.normalize_each:
        spectra   = [s / s.max() if s.max() > 0 else s for s in spectra]
        y_label   = 'Intensity (normalised per spectrum)'
        norm_mode = 'enorm'
    elif args.normalize:
        gmax      = max(s.max() for s in spectra)
        spectra   = [s / gmax if gmax > 0 else s for s in spectra]
        y_label   = 'Intensity (normalised to global max)'
        norm_mode = 'gnorm'
    else:
        y_label   = 'Raman Activity (arb. units, absolute)'
        norm_mode = 'abs'

    # Per-trace file labels -- the # File_Label metadata written by
    # RASCBEC_phonopy.py (e.g. 'Ca01-Cl06'), shown on the left of each trace
    # in place of a legend.  Custom --labels and the chem/E-field fallback
    # are still honoured when File_Label is absent or overridden.
    # Per-trace file labels -- the # File_Label metadata written by
    # RASCBEC_phonopy.py (e.g. 'Ca01-Cl06'), shown on the left of each trace
    # in place of a legend.  Undoped (File_Label 'Undoped') falls back to the
    # formula; custom --labels and the chem/E-field legend labels are used
    # only when no File_Label is available.
    auto_labels, efield_in_legend = build_legend_labels(all_meta, files)
    if args.labels:
        for i, lbl in enumerate(args.labels):
            if i < len(auto_labels):
                auto_labels[i] = lbl
    trace_labels = []
    for idx, (meta, lbl) in enumerate(zip(all_meta, auto_labels)):
        fl = meta.get('file_label', '')
        if fl == 'Undoped':
            fl = meta.get('formula', '')
        if not fl:
            fl = lbl
        if efield_in_legend:
            e = meta.get('e_field', '') or filename_fallback(files[idx])[1]
            if e:
                fl = f"{fl} | E={e} eV/Å"
        trace_labels.append(fl)

    # Figure title (2 lines, mirroring plot_raman.py):
    #   line 1 : formula / comparison label
    #   line 2 : E-field (if shared) | FWHM
    if args.title:
        plot_title = args.title
    else:
        formulae    = [m.get('formula', '') for m in all_meta]
        unique_f    = set(f for f in formulae if f)
        title_line1 = (f"{list(unique_f)[0]} Raman Comparison"
                       if len(unique_f) == 1 else "Raman Spectra Comparison")
        title_line2_parts = []
        if not efield_in_legend:
            efields   = [get_efield(m, f) for m, f in zip(all_meta, files)]
            unique_ef = set(e for e in efields if e)
            if len(unique_ef) == 1:
                title_line2_parts.append(f"E = {list(unique_ef)[0]} eV/Å")
        title_line2_parts.append(f"FWHM = {args.gamma*THZCM1:.4f} cm$^{{-1}}$")
        plot_title = f"{title_line1}\n" + " | ".join(title_line2_parts)

    # Plot
    set_style()
    n     = len(files)
    fig_h = max(6, n * 1) if waterfall_mode else 6
    fig, ax = plt.subplots(figsize=(6, fig_h))

    colors = pick_colors(n)

    for i, (spectrum, freqs, acts, irreps, label, color) in enumerate(
            zip(spectra, all_freqs, all_acts, all_irreps, trace_labels, colors)):

        # Stack first (alphabetically) input file at the bottom, matching
        # plot_wf in plot_raman_waterfall.py (y0 = i * dy).
        baseline = i * args.offset
        has_irreps = any(lbl != '' for lbl in irreps)

        ax.plot(x, spectrum + baseline, lw=1.6, color=color, zorder=4)

        # Colored label above each trace on the left (mirrors plot_wf in
        # plot_raman_waterfall.py): 25% of the trace height above its baseline.
        label_dy = float(spectrum.max()) * 0.2 if spectrum.max() > 0 else 0.0
        ax.text(
            freq_min,
            baseline + label_dy,
            f"  {label}",
            color=color,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
            zorder=5)

        # Sticks (pre-broadened activities; scaled to the maximum activity,
        # matching the stick logic in plot_raman.py / plot_raman_waterfall.py)
        if args.sticks and freqs.size:
            stick_scale = 0.6 / acts.max() if acts.max() > 0 else 1.0
            for f_val, A in zip(freqs, acts):
                if A > 0 and freq_min <= f_val <= freq_max:
                    ax.vlines(f_val, baseline, baseline + A * stick_scale,
                              color=color, lw=STICK_LW, alpha=STICK_A, zorder=3)

        # Peak labels -- waterfall mode only
        if show_labels:
            min_dist = max(int(5000 / (freq_max - freq_min) * 15), 1)
            peaks, _ = find_peaks(spectrum,
                                  height=0.02 * spectrum.max(),
                                  distance=min_dist)
            top_peaks = sorted(peaks, key=lambda p: -spectrum[p])[:args.n_labels]
            for pk in top_peaks:
                peak_freq = x[pk]
                if has_irreps and len(freqs) > 0:
                    closest_idx = int(np.argmin(np.abs(freqs - peak_freq)))
                    irrep       = irreps[closest_idx]
                    label_text  = f"{irrep}\n{peak_freq:.0f}"
                else:
                    label_text  = f"{peak_freq:.0f}"
                ax.annotate(
                    label_text,
                    xy=(peak_freq, spectrum[pk] + baseline),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=6, color=color, linespacing=1.0)

    # Axes formatting (tick/spine/grid style comes from set_style())
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
    ax.set_ylabel(y_label)
    # Explicit grid state (ax.grid(axis=...) toggles, so set both axes
    # unconditionally): horizontal baselines only, styled by set_style().
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_title(plot_title)
    # No legend -- file labels are drawn on the left of each trace (inside the
    # axes, matching plot_wf in plot_raman_waterfall.py).
    fig.tight_layout()
    out_png = args.out or build_output_name(all_meta, norm_mode)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")

if __name__ == '__main__':
    main()
