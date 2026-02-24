#!/usr/bin/env python3

"""
compare_raman.py — Overlay Raman spectra from multiple RASCBEC CSV files.

Constraints
-----------
  --offset > 0   automatically applies --normalize-each if no norm flag is given
  Peak labels    shown ONLY in offset (waterfall) mode; default --n-labels 10

Output filename (when --out is not given)
-----------------------------------------
  raman_compare_<formula>_<norm>.png

  formula : shared formula from CSV metadata (e.g. Na3PS4); 'mixed' if differ
  norm    : abs | gnorm | enorm

  e.g.  raman_compare_Na3PS4_enorm.png

Comparing same composition at different E-fields
-------------------------------------------------
  When all CSVs share the same chemical label (formula + dopants), the E-field
  is automatically appended to each legend entry so spectra are distinguishable.
  The E-field is then omitted from the title (it lives in the legend instead).

Legend labels and title are read from the CSV metadata header:
  # Formula:     Na3PS4
  # Dopants:     Ca(x=0.06)/Cl(x=0.12)
  # E_field:     0.02
  # Composition: 06-12

Usage:
    # Overlaid absolute (no labels)
    python compare_raman.py raman_*.csv

    # Waterfall — auto normalize-each + labels (typical use)
    python compare_raman.py raman_*.csv --offset 1.2

    # Waterfall with explicit global normalisation
    python compare_raman.py raman_*.csv --normalize --offset 1.2

    # Same composition, compare E-field strength
    python compare_raman.py raman_06-12_0.01.csv raman_06-12_0.02.csv raman_06-12_0.05.csv --offset 1.2

    # Finer control
    python compare_raman.py raman_*.csv --offset 1.2 --n-labels 5
    python compare_raman.py raman_*.csv --offset 1.2 --n-labels 0  # disable labels
    python compare_raman.py raman_*.csv --gamma 8 --freq-min 50 --freq-max 550
    python compare_raman.py raman_*.csv --out my_fig.png
    python compare_raman.py --help
"""

import argparse
import glob as _glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Lorentzian broadening
# ---------------------------------------------------------------------------

def lorentzian(x, x0, A, gamma):
    return A * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)


def build_spectrum(freqs_cm1, activities, x, gamma, freq_min, freq_max):
    """Absolute Lorentzian-broadened spectrum on grid x."""
    spectrum = np.zeros_like(x)
    for f, A in zip(freqs_cm1, activities):
        if A > 0 and freq_min <= f <= freq_max:
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
    Load a RASCBEC CSV.  Returns (freqs_cm1, activities, meta).

    Metadata parsed from comment lines (all default to '' if absent):
      # Formula, # Dopants, # E_field, # Composition
    """
    meta  = {'formula': '', 'dopants': '', 'e_field': '', 'composition': ''}
    freqs, acts = [], []
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if   s.startswith('# Formula:'):
                meta['formula']     = s.split(':', 1)[1].strip()
            elif s.startswith('# Dopants:'):
                meta['dopants']     = s.split(':', 1)[1].strip()
            elif s.startswith('# E_field:'):
                meta['e_field']     = s.split(':', 1)[1].strip()
            elif s.startswith('# Composition:'):
                meta['composition'] = s.split(':', 1)[1].strip()
            elif s.startswith('#') or not s:
                continue
            else:
                parts = s.split(',')
                freqs.append(float(parts[1]))
                acts.append(float(parts[2]))
    return np.array(freqs), np.array(acts), meta


def filename_fallback(filepath):
    """raman_06-12_0.02.csv  →  tag='06-12', efield='0.02'"""
    stem  = Path(filepath).stem
    inner = stem[6:] if stem.startswith('raman_') else stem
    parts = inner.rsplit('_', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (inner, '')


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

    - Different compositions, same E  → 'Na3PS4 [Ca(x=0.06)/Cl(x=0.12)]'  (E in title)
    - Same composition, different E   → 'Na3PS4 [...] | E=0.02 eV/Å'       (E in legend)
    - Both differ                     → 'Na3PS4 [...] | E=0.02 eV/Å'       (E in legend)

    Returns (legend_labels, efield_in_legend)
      efield_in_legend : bool — True means E is in the legend, not the title
    """
    chem_labels = [chem_label(m, f) for m, f in zip(all_meta, files)]
    efields     = [get_efield(m, f)  for m, f in zip(all_meta, files)]

    # If chemical labels are not all unique, differentiate by E-field
    efield_in_legend = len(set(chem_labels)) < len(chem_labels)

    if efield_in_legend:
        labels = [f"{c}  |  E={e} eV/Å" if e else c
                  for c, e in zip(chem_labels, efields)]
    else:
        labels = chem_labels

    return labels, efield_in_legend


# ---------------------------------------------------------------------------
# Output filename
# ---------------------------------------------------------------------------

def build_output_name(all_meta, norm_mode):
    """
    raman_compare_<formula>_<norm>.png

    formula : shared formula from metadata; 'mixed' if files differ
    norm    : abs | gnorm | enorm
    """
    formulae  = [m.get('formula', '') for m in all_meta]
    unique_f  = set(f for f in formulae if f)
    formula   = list(unique_f)[0] if len(unique_f) == 1 else 'mixed'
    return f"raman_compare_{formula}_{norm_mode}.png"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare Raman spectra from multiple RASCBEC CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('files', nargs='+',
                   help='Input CSV files (shell glob patterns accepted)')
    p.add_argument('--gamma', type=float, default=10.0,
                   help='Lorentzian FWHM in cm-1 (default: 10.0)')
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
                   help='Output PNG (default: raman_compare_<formula>_<norm>.png)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    files = expand_globs(args.files)
    if not files:
        raise SystemExit("No input files found.")

    # ── Resolve normalisation mode ────────────────────────────────────────────
    # --offset without an explicit norm flag → silently default to normalize-each
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

    # ── Load data ────────────────────────────────────────────────────────────
    all_freqs, all_acts, all_meta = [], [], []
    for f in files:
        freqs, acts, meta = load_csv(f)
        all_freqs.append(freqs)
        all_acts.append(acts)
        all_meta.append(meta)

    # ── Frequency grid ───────────────────────────────────────────────────────
    freq_min = args.freq_min
    freq_max = args.freq_max or (max(fq.max() for fq in all_freqs) + 50.0)
    x = np.linspace(freq_min, freq_max, 5000)

    # ── Build absolute spectra ────────────────────────────────────────────────
    spectra = [build_spectrum(freqs, acts, x, args.gamma, freq_min, freq_max)
               for freqs, acts in zip(all_freqs, all_acts)]

    # ── Apply normalisation ───────────────────────────────────────────────────
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

    # ── Legend labels — auto E-field disambiguation ───────────────────────────
    auto_labels, efield_in_legend = build_legend_labels(all_meta, files)
    if args.labels:
        for i, lbl in enumerate(args.labels):
            if i < len(auto_labels):
                auto_labels[i] = lbl
    legend_labels = auto_labels

    # ── Figure title ──────────────────────────────────────────────────────────
    if args.title:
        title_parts = [args.title]
    else:
        formulae = [m.get('formula', '') for m in all_meta]
        unique_f  = set(f for f in formulae if f)
        title_parts = [f"{list(unique_f)[0]} Raman Comparison"
                       if len(unique_f) == 1 else "Raman Spectra Comparison"]

    # Only add shared E-field to title when it's NOT already in the legend
    if not efield_in_legend:
        efields   = [get_efield(m, f) for m, f in zip(all_meta, files)]
        unique_ef = set(e for e in efields if e)
        if len(unique_ef) == 1:
            title_parts.append(f"E = {list(unique_ef)[0]} eV/Å")

    title_parts.append(f"FWHM = {args.gamma} cm$^{{-1}}$")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n     = len(files)
    fig_h = max(4.5, 2.5 + n * 0.9) if waterfall_mode else 4.5
    fig, ax = plt.subplots(figsize=(11, fig_h))

    prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors      = [prop_colors[i % len(prop_colors)] for i in range(n)]

    for i, (spectrum, freqs, acts, label, color) in enumerate(
            zip(spectra, all_freqs, all_acts, legend_labels, colors)):

        baseline = i * args.offset

        ax.plot(x, spectrum + baseline, lw=1.5, color=color, label=label, zorder=3)
        ax.fill_between(x, baseline, spectrum + baseline,
                        alpha=0.12, color=color, zorder=2)

        # Sticks
        if args.sticks:
            ref_h = (spectrum.max() or 1.0) * 0.6
            for f_val, A in zip(freqs, acts):
                if A > 0 and freq_min <= f_val <= freq_max:
                    h = A / acts.max() * ref_h if acts.max() > 0 else 0
                    ax.vlines(f_val, baseline, baseline + h,
                              color=color, lw=0.5, alpha=0.4, zorder=1)

        # Peak labels — waterfall mode only
        if show_labels:
            min_dist = max(int(5000 / (freq_max - freq_min) * 15), 1)
            peaks, _ = find_peaks(spectrum,
                                  height=0.02 * spectrum.max(),
                                  distance=min_dist)
            top_peaks = sorted(peaks, key=lambda p: -spectrum[p])[:args.n_labels]
            for pk in top_peaks:
                ax.annotate(
                    f'{x[pk]:.0f}',
                    xy         = (x[pk], spectrum[pk] + baseline),
                    xytext     = (0, 6),
                    textcoords = 'offset points',
                    ha='center', va='bottom',
                    fontsize=6.5, color=color,
                )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=11)
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_title("  |  ".join(title_parts), fontsize=11)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8,
              title="Composition", title_fontsize=9)

    fig.tight_layout()
    out_png = args.out or build_output_name(all_meta, norm_mode)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
