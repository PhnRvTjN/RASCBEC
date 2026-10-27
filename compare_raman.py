#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
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
   Stokes scattered intensities at T = 350 K (632.8 nm laser line), directly
   matching experimental multi-composition waterfall comparisons. Raw activity
   comparison remains available via ``--raw-act``.

Backwards Compatibility with Legacy CSVs:
-----------------------------------------
No changes to existing CSV files are needed. The script parses existing 3- or
4-column CSVs (with or without a trailing comment-header metadata block) and
computes the thermal Stokes weights on the fly.

Constraints and Layout:
-----------------------
- Frequency window: --freq-min defaults to 0 and --freq-max defaults to the
  highest mode rounded up to the nearest 50 cm^-1 (same as
  plot_raman.py). A zoom window must be a valid min < max range that overlaps
  the data; otherwise the auto range is used (a note is printed).
- --offset > 0 automatically applies --norm-each if no norm flag is given.
- Peak labels are shown ONLY in offset (waterfall) mode; default --n-labels 10.
  Peaks are selected by prominence (not raw height) and kept at least
  --min-label-sep (default 10) cm^-1 apart; a candidate inside that window of a
  stronger peak is merged into it (split-Lorentzian shoulder) and dropped.
- If no input files are given, every *.csv in --dir (default: cwd) is used.
- In --offset (waterfall) mode, trace labels are drawn on the right of each
  trace (no legend), mirroring plot_raman_waterfall.py; without --offset the
  same labels go into a standard legend box in the upper-right corner instead.
  In waterfall mode each trace label sits --offset above the spectrum's
  intensity at the label's x position.
  Colours use the hue-sorted tab10 / turbo logic.

Usage:
    # Waterfall with Stokes conversion (T=350 K, default), all *.csv in cwd
    python compare_raman.py --offset 1.2

    # Comparing columnar vs rock-salt LLTO arrangements
    python compare_raman.py C-210-79_E0.02.csv R-616-59_E0.02.csv --offset 1.2

    # Variable temperature comparison
    python compare_raman.py *.csv --offset 1.2 --temp 100
    python compare_raman.py *.csv --offset 1.2 --temp 600

    # Legacy raw Raman activity S(nu) mode (no 1/nu or thermal factors)
    python compare_raman.py *.csv --offset 1.2 --raw-act

    # Explicit global normalisation (preserves relative intensity between systems)
    python compare_raman.py *.csv --norm --offset 1.2

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
    activity_to_stokes_intensity, build_raman_title,
    formula_to_mathtext, lorentzian, read_raman_csv,
    select_labeled_peaks, select_title_line, set_style,
    THZCM1, DEFAULT_LASER_WL, DEFAULT_TEMP,
    STICK_A, STICK_LW, STICK_MAX_FRAC,
    ASPECT_RATIO, UWM_GOLD, UWM_GREY
)

# ---------------------------------------------------------------------------
# Defaults (aligned with plot_raman.py / plot_raman_waterfall.py)
# ---------------------------------------------------------------------------
UPPER_OFFSET_RATIO = 4.0      # Waterfall: upper padding = ratio x --offset (lower = 1 x --offset)

# Note: the publication-ready rcParams come from plot_raman.set_style (see
# import above); compare_raman applies it with compact=True in main().

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
    scales with series count. If n >= 10, fall back to the ``turbo`` colormap.
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
# Spectrum broadening
# ---------------------------------------------------------------------------
# lorentzian() and activity_to_stokes_intensity() are imported from plot_raman
# (see the import block above); they were previously duplicated here.

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

    Thin wrapper over the shared :func:`plot_raman.read_raman_csv`; the only
    adjustment is keeping ``meta['e_field']`` as a string (or ``''`` when
    absent) so the label/title code below can treat it as a plain string.
    """
    freqs, acts, irreps, meta = read_raman_csv(filepath)
    e = meta.get('e_field')
    meta['e_field'] = '' if e is None else str(e)
    return freqs, acts, irreps, meta


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
    present, returns ``[Dopants]`` (``build_trace_labels`` replaces this with
    the subscript dopant-fraction legend).
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


def build_output_name(all_meta: List[Dict], norm_mode: str, raw_act: bool) -> str:
    """Generate output filename encoding formula, normalization, and physical mode."""
    formulae = [m.get('formula', '') for m in all_meta]
    unique_f = set(f for f in formulae if f)
    formula = list(unique_f)[0] if len(unique_f) == 1 else 'mixed'
    suffix = "activity" if raw_act else "stokes"
    return f"compare_{formula}_{suffix}_{norm_mode}.png"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare Raman spectra from multiple RASCBEC CSV files with Stokes thermal conversion.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    # --- input / output ---
    p.add_argument('files', nargs='*',
                   help='Input CSV files (shell glob patterns accepted). '
                        'If omitted, all *.csv in --dir are used.')
    p.add_argument('--dir', type=str, default='.',
                   help='Directory to search for *.csv input files when no '
                        'files are given on the command line (default: cwd).')
    p.add_argument('--out', default=None,
                   help='Output PNG filename (default: compare_<formula>_<mode>_<norm>.png)')

    # --- physical model (what is plotted and under what conditions) ---
    p.add_argument('--temp', type=float, default=DEFAULT_TEMP,
                   help='Sample temperature in K for Bose-Einstein factor (default: 350.0 K)')
    p.add_argument('--laser-wl', type=float, default=DEFAULT_LASER_WL,
                   help='Excitation laser wavelength in nm '
                        f'(default: {DEFAULT_LASER_WL:g} nm)')
    p.add_argument('--raw-act', action='store_true',
                   help='Plot raw Raman activities S(nu) without 1/nu or thermal Bose factors')
    p.add_argument('--gamma', type=float, default=0.25,
                   help='Lorentzian FWHM in THz (default: 0.25 THz); converted '
                        'to cm^-1 internally via THZCM1.')

    # --- displayed frequency window (cm^-1) ---
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm^-1 (default: 0)')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm^-1 (default: auto = highest '
                        'mode rounded up to the nearest 50 cm^-1)')

    # --- normalization ---
    norm = p.add_mutually_exclusive_group()
    norm.add_argument('--norm', action='store_true',
                      help='Normalize all spectra to the global maximum '
                           '(preserves relative intensities between compositions)')
    norm.add_argument('--norm-each', action='store_true',
                      help='Normalize each spectrum to its own maximum '
                           '(pure peak-shape / position comparison; default when --offset is given)')

    # --- waterfall ---
    p.add_argument('--offset', type=float, default=0.0,
                   help='Vertical spacing between stacked spectra for '
                        'waterfall view (default: 0 = overlaid). Lower padding '
                        f'below each spectrum is this value; upper padding '
                        f'above each spectrum is {UPPER_OFFSET_RATIO:g}x this '
                        f'value (see UPPER_OFFSET_RATIO). The trace label is '
                        'placed this amount above its spectrum. Automatically '
                        'applies --norm-each unless --norm is given.')

    # --- stick spectrum ---
    p.add_argument('--sticks', action='store_true',
                   help='Overlay stick (bar) spectrum for each composition')

    # --- peak labels ---
    p.add_argument('--n-labels', type=int, default=10,
                   help='Number of peak frequency labels per spectrum in '
                        'waterfall mode (default: 10; set 0 to disable).')
    p.add_argument('--min-label-sep', type=float, default=10.0,
                   help='Minimum separation in cm^-1 between labelled peaks '
                        'in waterfall mode (default: 10; set 0 to disable).')
    p.add_argument('--prominence', type=float, default=0.01,
                   help='Prominence threshold (fraction of peak-to-baseline '
                        'span) used to detect labellable peaks (default: 0.01)')
    p.add_argument('--label-xfree', type=float, default=0.2,
                   help='Horizontal margin for the waterfall trace label, as a '
                        'fraction of the x-range to inset from the right edge '
                        '(default: 0.2).')

    # --- trace labels ---
    p.add_argument('--labels', nargs='+', default=None,
                   help='Custom legend labels, one per file in order')

    # --- title ---
    p.add_argument('--title', type=str, default=None,
                   help='Custom figure title (default: auto-generated). Pass '
                        "'line1' or 'line2' to show only the formula line or "
                        "only the parameters line of the auto-generated title, "
                        'or an empty string --title "" to omit the title.')

    # --- decorations (grid, figure size & reference lines) ---
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_input_files(args) -> List[str]:
    """Resolve the input CSV files (explicit args or a glob of --dir)."""
    if args.files:
        files = expand_globs(args.files)
    else:
        files = sorted(_glob.glob(str(Path(args.dir) / "*.csv")))
    if not files:
        raise SystemExit(
            "No input CSV files found. "
            "Pass file paths or use --dir to point at a folder of *.csv.")
    return files


def load_all(files: List[str]):
    """Load every CSV into parallel lists of (freqs, acts, irreps, meta)."""
    all_freqs, all_acts, all_irreps, all_meta = [], [], [], []
    for f in files:
        freqs, acts, irreps, meta = load_csv(f)
        all_freqs.append(freqs)
        all_acts.append(acts)
        all_irreps.append(irreps)
        all_meta.append(meta)
    return all_freqs, all_acts, all_irreps, all_meta


def build_all_spectra(all_freqs, all_acts, args):
    """Build broadened spectra on a common grid that contains the display window.

    The resolution grid always spans every mode plus a 50 cm^-1 margin and also
    contains the displayed window, so a custom --freq-min/--freq-max zoom can
    never truncate a Lorentzian peak/tail before the max is determined. When
    --freq-max is not given, the auto upper axis limit mirrors plot_raman.py:
    the highest mode rounded up to the nearest 50 cm^-1.

    Returns
    -------
    spectra, x, in_window, full_min, full_max, freq_min, freq_max, all_weights
    """
    gamma = args.gamma * THZCM1

    # Data range spanning every mode.
    data_min = min(fq.min() for fq in all_freqs)
    data_max = max(fq.max() for fq in all_freqs)

    # Auto (default) range mirrors plot_raman.py: lower bound 0; upper bound the
    # highest mode, rounded up to the nearest 50 cm^-1 so
    # the axis lands on a "nice" value rather than a raw one (e.g. 670.9 -> 700).
    auto_freq_min = 0.0
    auto_freq_max = np.ceil(data_max / 50) * 50

    # Displayed (zoomed) window: the user's --freq-min/--freq-max, defaulting to
    # the auto range above when they are not given.
    freq_min = args.freq_min
    freq_max = args.freq_max if args.freq_max is not None else auto_freq_max

    # Fall back to the auto range on clear user error: an inverted window
    # (freq_min >= freq_max), or an explicit --freq-max that misses the data
    # entirely (almost always a typo). This honours a deliberate partial zoom
    # while keeping a peak from being cut off by accident.
    inverted = not (freq_min < freq_max)
    misses_data = (args.freq_max is not None
                   and not (freq_min <= data_max and freq_max >= data_min))
    if inverted or misses_data:
        reason = ("not a valid window (need --freq-min < --freq-max)" if inverted
                  else f"misses the data range [{data_min:g}, {data_max:g}]")
        print(f"  (note: window [{freq_min:g}, {freq_max:g}] {reason}; "
              f"using the auto range [{auto_freq_min:g}, {auto_freq_max:g}] instead)")
        freq_min, freq_max = auto_freq_min, auto_freq_max

    # Resolution grid: covers every mode (with a 50 cm^-1 margin) and always
    # contains the displayed window. The broadened spectrum is built and
    # normalised on this full range, so a custom --freq-min/--freq-max zoom can
    # never truncate a Lorentzian peak/tail (or change the peak used for
    # normalisation) before the max is determined.
    full_min = min(data_min - 50.0, freq_min)
    full_max = max(data_max + 50.0, freq_max)
    x = np.linspace(full_min, full_max, 10000)

    # Compute mode weights (Stokes intensity or raw activity)
    all_weights = []
    for freqs, acts in zip(all_freqs, all_acts):
        if args.raw_act:
            weights = np.copy(acts)
        else:
            weights = activity_to_stokes_intensity(
                freqs, acts,
                temp=args.temp,
                laser_wl=args.laser_wl,
            )
        all_weights.append(weights)

    # Build broadened spectra from calculated weights
    spectra = [
        build_spectrum(freqs, weights, x, gamma)
        for freqs, weights in zip(all_freqs, all_weights)
    ]

    # Mask to the displayed frequency window; all window-dependent quantities
    # (trace-label position, peak labels, sticks) use in_window, while
    # normalisation uses the full spectrum.
    in_window = (x >= freq_min) & (x <= freq_max)

    return (spectra, x, in_window, full_min, full_max,
            freq_min, freq_max, all_weights)


def norm_spectra(spectra: List[np.ndarray], args) -> Tuple[List[np.ndarray], str, str]:
    """Normalize the spectra and return (spectra, y_label, norm_mode).

    Normalisation is always based on the FULL spectrum, even when the plot is
    zoomed to a custom --freq-min/--freq-max window, so zooming never
    rescales the traces.
    """
    if args.norm_each:
        spectra = [s / s.max() if s.max() > 0 else s for s in spectra]
        y_label = 'Intensity (simulated, normalized per spectrum)' if not args.raw_act else 'Activity (normalized per spectrum)'
        norm_mode = 'enorm'
    elif args.norm:
        gmax = max(s.max() for s in spectra)
        spectra = [s / gmax if gmax > 0 else s for s in spectra]
        y_label = 'Intensity (simulated, normalized to global max)' if not args.raw_act else 'Activity (normalized to global max)'
        norm_mode = 'gnorm'
    else:
        y_label = 'Stokes Intensity (simulated, arb. units, absolute)' if not args.raw_act else 'Raman Activity S(ν) (absolute)'
        norm_mode = 'abs'
    return spectra, y_label, norm_mode


def build_trace_labels(all_meta, files, args) -> List[str]:
    """Build the per-trace labels shown in the legend / beside each trace."""
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

    return trace_labels


def build_plot_title(all_meta, files, args, gamma):
    """Resolve the figure title and the title-line selector.

    --title 'line1'/'line2' selects one line of the auto-generated title (it is
    not itself a custom title string); --title '' omits the title entirely; any
    other --title value is shown as-is.
    """
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
                temp=args.temp,
                raw_act=args.raw_act,
                suffix="Raman Comparison",
                show_legend=False,
            )
        else:
            title_line2_parts = []
            if len(unique_ef) == 1:
                title_line2_parts.append(f"E = {list(unique_ef)[0]} eV/Å")
            title_line2_parts.append(f"FWHM = {gamma:.2f} cm$^{{-1}}$")
            if not args.raw_act:
                title_line2_parts.append(f"T = {args.temp:g} K")
            else:
                title_line2_parts.append("S(ν)")
            plot_title = "Raman Spectra Comparison\n" + " | ".join(title_line2_parts)

    return plot_title, title_line


def compute_layout(args, waterfall_mode, n, freq_min, freq_max, full_min, full_max,
                   vis) -> Tuple[List[float], float, float, float]:
    """Compute waterfall baselines/stack extent and the figure size.

    Returns
    -------
    baselines : list of float
        One baseline offset per trace.
    stack_top : float
        Vertical extent of the stacked traces.
    fig_w, fig_h : float
        Figure size in inches (after any --figsize override).
    """
    # Waterfall baselines: --offset is the LOWER padding below each trace
    # baseline; the UPPER padding above the baseline is UPPER_OFFSET_RATIO x
    # offset, so --offset 0.5 gives 0.5 below and 1.0 above each baseline.
    if waterfall_mode:
        # Per-trace cell (baseline to next baseline): lower + upper padding.
        cell = args.offset * (1.0 + UPPER_OFFSET_RATIO)
        baselines = [args.offset + i * cell for i in range(n)]
        # Stack extent: the final trace's own peak (within the window) plus
        # one plain --offset of headroom above it (not ratio-multiplied).
        final_max = float(max(vis[-1].max(), 0.0)) if vis else 0.0
        stack_top = baselines[-1] + final_max + 2 * args.offset
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

    return baselines, stack_top, fig_w, fig_h


def draw_waterfall(ax, *, spectra, vis, all_freqs, all_weights, all_irreps,
                   trace_labels, in_window, x, baselines, freq_min, freq_max,
                   waterfall_mode, show_labels, y_label, grid, x_lines, y_lines,
                   n, stack_top, args) -> None:
    """Draw the spectra (traces, optional sticks, optional peak labels), the
    legend, axis labels, grid, and reference lines."""
    colors = pick_colors(n)

    for i, (spectrum, freqs, weights, irreps, label, color) in enumerate(
            zip(spectra, all_freqs, all_weights, all_irreps, trace_labels, colors)):

        baseline = baselines[i]

        # In non-waterfall mode the trace label goes into the legend box; in
        # waterfall mode it is drawn inline to the right of each trace.
        ax.plot(x[in_window], spectrum[in_window] + baseline, lw=1.6, color=color, zorder=4,
                label=label if not waterfall_mode else None)

        if waterfall_mode:
            # Trace label sits args.offset above this trace's intensity at
            # label_xfree within the selected frequency window.
            label_x = freq_max - args.label_xfree * (freq_max - freq_min)
            xw = x[in_window]
            v = vis[i]
            idx = int(np.argmin(np.abs(xw - label_x)))
            label_dy = float(v[idx]) + args.offset * (1.0 + 1.0 / UPPER_OFFSET_RATIO)
            ax.text(
                label_x,
                baseline + label_dy,
                f"{label}",
                color=color,
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="left",
                zorder=5,
                bbox=dict(facecolor=UWM_GREY, edgecolor=UWM_GOLD, alpha=0.4, pad=1.0),
            )

        if args.sticks and freqs.size:
            stick_scale = STICK_MAX_FRAC / weights.max() if weights.max() > 0 else 1.0
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
                label_text = f"{peak_freq:.1f}"
                ax.annotate(
                    label_text,
                    xy=(peak_freq, vis[i][pk] + baseline),
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, color=color, linespacing=1.0)

    if not waterfall_mode and n > 1:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)

    ax.set_xlim(freq_min, freq_max)
    if waterfall_mode:
        # Use the computed stack extent (incl. headroom above the final trace)
        # so the top offset is actually rendered; otherwise auto-fit.
        ax.set_ylim(0, stack_top)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
    ax.set_ylabel(y_label)

    # Grid visibility (default 'x' preserves the original x-grid-only look).
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
            ax.axvline(xv, color=UWM_GREY, lw=0.8, ls="--", alpha=0.8, zorder=2)

    # Optional horizontal reference lines at user-selected y-axis values.
    if y_lines:
        for yv in y_lines:
            ax.axhline(yv, color=UWM_GREY, lw=0.8, ls="--", alpha=0.8, zorder=2)


def main():
    args = parse_args()

    waterfall_mode = args.offset > 0
    if waterfall_mode and not args.norm and not args.norm_each:
        args.norm_each = True
        print("  (auto: --norm-each applied because --offset > 0)")

    show_labels = args.n_labels > 0 and waterfall_mode
    if args.n_labels > 0 and not waterfall_mode:
        print("  (note: peak labels are shown only in --offset waterfall mode)")

    files = resolve_input_files(args)

    mode_str = "Raw Activity S(nu)" if args.raw_act else f"Stokes Intensity (T={args.temp:g} K, {args.laser_wl:g} nm)"
    print(f"Comparing {len(files)} spectra [{mode_str}]:")
    for f in files:
        print(f"  {Path(f).name}")

    all_freqs, all_acts, all_irreps, all_meta = load_all(files)

    (spectra, x, in_window, full_min, full_max,
     freq_min, freq_max, all_weights) = build_all_spectra(all_freqs, all_acts, args)

    spectra, y_label, norm_mode = norm_spectra(spectra, args)

    # Masked (zoomed) views drive the window-dependent quantities below.
    vis = [s[in_window] for s in spectra]

    trace_labels = build_trace_labels(all_meta, files, args)
    gamma = args.gamma * THZCM1
    plot_title, title_line = build_plot_title(all_meta, files, args, gamma)

    set_style(compact=True)
    n = len(files)
    baselines, stack_top, fig_w, fig_h = compute_layout(
        args, waterfall_mode, n, freq_min, freq_max, full_min, full_max, vis)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    draw_waterfall(
        ax,
        spectra=spectra, vis=vis, all_freqs=all_freqs, all_weights=all_weights,
        all_irreps=all_irreps, trace_labels=trace_labels, in_window=in_window,
        x=x, baselines=baselines, freq_min=freq_min, freq_max=freq_max,
        waterfall_mode=waterfall_mode, show_labels=show_labels, y_label=y_label,
        grid=args.grid, x_lines=args.x_line, y_lines=args.y_line, n=n,
        stack_top=stack_top, args=args,
    )

    # --title line1/line2 only affects the auto-generated title; a custom
    # --title is shown as-is, and --title "" still omits the title entirely.
    if plot_title is not None and args.title is None:
        plot_title = select_title_line(plot_title, title_line)
    ax.set_title(plot_title or "")

    fig.tight_layout()
    out_png = args.out or build_output_name(all_meta, norm_mode, args.raw_act)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
