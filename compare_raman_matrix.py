#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
compare_raman_matrix.py -- Plot static Raman spectra in a two-dopant
composition matrix.

Sister script to ``compare_raman.py``: where ``compare_raman`` stacks or
overlays arbitrary CSV traces, this script arranges the same CSV files into
a categorical ``x_dopant`` vs ``y_dopant`` grid of shared-scale panels
(mirroring the published two-dopant composition matrices).

Physical Formulation and Stokes Conversion:
-------------------------------------------
Same formulation as ``plot_raman.py`` / ``compare_raman.py``. The raw
quantity output by RASCBEC is the Placzek isotropic powder Raman activity:

.. math::
    A_s = 45 \bar{\alpha}'^2 + 7 {\gamma'}^2 \propto \left| \frac{\partial \alpha}{\partial Q_s} \right|^2

In experimental Raman spectroscopy, the detector records the Stokes
scattered photon rate :math:`I_s(\nu, T)`, which incorporates the harmonic
zero-point coordinate amplitude (:math:`1/\nu_s`), the thermal Bose-Einstein
phonon population factor (:math:`n(\nu_s, T) + 1`), and the dipole radiation
term (:math:`(\nu_L - \nu_s)^4`):

.. math::
    I_s(\nu, T) \propto \frac{(\nu_L - \nu_s)^4}{\nu_s}
    \left[ \frac{1}{1 - \exp\left(-\frac{h c \nu_s}{k_B T}\right)} \right] A_s

By default every mode is converted to a physical Stokes scattered intensity
at T = 350 K (632.8 nm laser line), directly matching experimental
multi-composition comparisons. Raw activity comparison remains available
via ``--raw-act``.

Composition layout
------------------
No concentrations or matrix dimensions are prescribed. In automatic
grouping, complete composition tags in ``File_Label`` or filename define
group identities (e.g. ``Ca00-Cl06``). Their digits are identifiers, NOT
inferred concentrations. If any sample lacks tags, all samples are grouped
by numerical metadata instead. ``--group-by metadata`` always uses exact
numerical compositions.

For tag groups, headings use the most precisely specified metadata value
for each axis level. Differing values are grouped only when consistent
within the coarser value's decimal rounding precision; a warning reports
the values and chosen heading. Thus a group containing 0.06 and 0.0625
can be displayed at 0.0625 without hardcoded aliases. Incompatible
metadata or two distinct tags with the same representative concentration
are errors. This precision check is a formatting-consistency check, not
proof of physical equivalence; use ``--group-by metadata`` when tags do
not uniquely identify intended sample levels. Missing cells are labeled
"Not calculated". Duplicate cells are errors.

Spectra and normalisation
-------------------------
Stokes stick weights are
``A*((nu_L - nu)/nu_L)^4/[nu*(1 - exp(-nu/(kB*T)))]``. Positive-frequency
activities are broadened by area-normalised Lorentzians with a common FWHM.
Stokes modes at/below ``--cutoff`` receive zero weight. Modes outside the
selected window still contribute tails (the grid spans every mode plus a
margin). The resolution grid is determined by all mode frequencies and FWHM,
not the zoom window, so normalisation is unaffected by ``--freq-min``/
``--freq-max``. No fitting, baseline subtraction, offsetting or
composition interpolation is applied.

Default per-spectrum normalisation divides each spectrum by its
full-spectrum maximum; ``--norm global`` uses one common maximum across all
samples; ``--norm none`` preserves numerical units. Reference overlays
reuse the same normalised pristine curve unchanged. Inter-system intensity
comparisons require comparable input normalisation; numerical units are not
absolute Raman cross sections.

Output
------
PNG by default, optional PDF/SVG (``--formats``), plus a JSON manifest.
Optional numerical NPZ exports contain full-grid spectra, window spectra and
mode weights. The common FWHM is an imposed broadening, not a calculated
lifetime linewidth. Panel placement is categorical; gaps are not
proportional to composition.

Dependencies: Python >=3.9, NumPy, Matplotlib, SciPy (via the shared
``plot_raman`` helpers).

Usage
-----
    # Full-spectrum matrix (auto range), default 350 K / 632.8 nm Stokes
    python compare_raman_matrix.py Na3PS4_matrix.csv --norm global

    # Zoom window into the P-S stretch region, shared Ca/Cl axes
    python compare_raman_matrix.py --freq-min 245 --freq-max 305 --dir compare

    # Reference (pristine) overlay on every panel
    python compare_raman_matrix.py *.csv --reference pristine.csv

    # Legacy raw Raman activity S(nu) (no 1/nu or thermal factors)
    python compare_raman_matrix.py *.csv --raw-act --norm each

    # Custom laser excitation wavelength (e.g. 532 nm)
    python compare_raman_matrix.py *.csv --laser-wl 532

    # Only one auto-generated title line (formula only)
    python compare_raman_matrix.py *.csv --title line1
"""

import argparse
import glob as _glob
import json
import re
import warnings
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Shared physical constants, style, CSV reader, Stokes conversion, Lorentzian
# broadening and peak-selection helpers -- imported exactly as in
# compare_raman.py so both scripts produce identical physics and typography.
# STICK_*/UWM_* carry the stick-spectrum and reference-line styling so this
# script matches plot_raman.py / compare_raman.py line-for-line.
from plot_raman import (
    THZCM1, DEFAULT_LASER_WL, DEFAULT_TEMP,
    STICK_A, STICK_LW, STICK_MAX_FRAC, UWM_GOLD, UWM_GREY,
    activity_to_stokes_intensity, lorentzian, read_raman_csv,
    select_labeled_peaks, select_title_line, set_style, build_raman_title,
)
# Shared output nomenclature (compare_<formula>_<stokes|activity>_<norm>) so
# the default output name agrees with compare_raman.py.
from compare_raman import build_output_name

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_POINTS = 2_000_000   # Hard cap on the spectral evaluation grid.

# ---------------------------------------------------------------------------
# Central layout / styling knobs for the matrix figure.
#
# Division of labour with the shared style (plot_raman.set_style, which sets
# the *global* matplotlib rcParams that compare_raman.py / plot_raman.py share):
#   * rcParams own what is truly global -- base/tick/legend font sizes, spine
#     + tick geometry, grid, mathtext weight, UWM palette constants.
#   * What rcParams *cannot* express is this figure's per-figure composition:
#     the subplot margins, inter-panel spacing, how far the axis titles sit
#     from the spines, the suptitle / dopant-heading weight and size, and the
#     element palette for the curves. Those are figure-level decisions, so they
#     are gathered here in ONE place. Edit a single value to retune the look;
#     nothing about them is scattered through draw()/save().
#
# Font sizes: `font_size` is the base; `title_size` and `heading_size` are
# multipliers of it, so raising --font-size scales the whole figure together.
# `--color` still overrides MATRIX_STYLE['color'] when supplied.
# ---------------------------------------------------------------------------
MATRIX_STYLE = {
    # --- figure ---
    'figsize': (6.0, 6.0),           # default figure size (inches)
    'dpi': 600,                      # raster resolution for PNG/SVG

    # --- typography (base + multipliers / offsets) ---
    'font_size': 10.0,               # base text size (points)
    'font_size_minor': 8.0,          # secondary text (tick labels, panel tags,
                                     # peak labels) -- base minus one point
    'title_size': 1.2,               # suptitle size = font_size * title_size
    'heading_size': 1.0,             # dopant-heading size = font_size * heading_size
    'bold_headings': True,           # bold x/y dopant headings (mathtext \mathbf)
    'bold_title': True,              # bold suptitle (matches axes.titleweight)
    'bold_axis_labels': True,        # bold "Intensity" / "Raman Shift"

    # --- subplot geometry (fractions of the figure; *_space in axes-fractions) ---
    'margin_left': 0.12,             # room for the left y-axis title
    'margin_right': 0.96,            # room for the right-edge y-dopant labels
    'margin_bottom': 0.12,           # x-tick labels + x-title + legend
    # When a suptitle is present the top band must fit BOTH the x-dopant column
    # headers (just above the top row) and, above them, the suptitle. The
    # required band height depends on how many title lines are shown, so there
    # is one margin_top per case: none (headers only), one line, two lines.
    # The title itself is anchored at a fixed suptitle_y (see below), so the
    # only thing to nudge per case is how much room the band leaves above the
    # headers for the title to hang into.
    'margin_top_notitle': 1.0,       # x-dopant headers only (no suptitle)
    'margin_top_1line':   0.9,      # one title line + x-dopant headers
    'margin_top_2line':   0.85,      # two title lines + x-dopant headers
    'wspace': 0.1,                   # inter-column gap
    'hspace': 0.1,                   # inter-row gap

    # --- axis-title placement (distance of the titles from the spines) ---
    'xlabel_y': 0.06,                # vertical offset of the x-axis title
    'ylabel_x': 0.04,                # horizontal offset of the y-axis title

    # --- scales, ticks and per-panel decorations ---
    'y_headroom': 1.2,               # shared y-max = max intensity * this
    'n_ticks': 4,                    # target major ticks per axis (MaxNLocator)
    'panel_label_pos': (0.05, 1.0),  # "(a)" tag position (transAxes, ha=left, va=top)
    'panel_label_alpha': 0.6,        # white box behind the tag
    'panel_label_pad': 0.1,
    'x_heading_pos': (0.5, 1.05),    # x-dopant header above the top row
    'y_heading_pos': (1.05, 0.5),    # y-dopant header at the right edge
    'peak_label_offset': 0.05,       # peak label sits this fraction of the y-span above its peak
    'peak_label_cap': 0.95,          # peak label never rises above this fraction of y-max

    # --- legend ---
    'legend_anchor': (0.5, 0.0),     # figure-level anchor (lower center)
    'legend_ncol': None,             # None -> one column per handle; 1 stacks
    'legend_frame': False,           # boxed legend (UWM gold edge, grey face)

    # --- suptitle ---
    # The suptitle is anchored at a single vertical position (figure fraction)
    # and hung with a fixed alignment, so the per-line-count work is entirely
    # done by the margin_top_* values above. Anchor near the top edge (va='top')
    # so the title hangs *downward* from a fixed point: every case shares the
    # same top edge, and you only grow the margin_top_* band until the title
    # bottom clears the column headers.
    'suptitle_y': 0.98,            # suptitle vertical position (figure fraction)
    'suptitle_va': 'top',          # anchor alignment so the title hangs down from suptitle_y

    # --- line / reference palette (UWM house colours from plot_raman) ---
    'color': 'tab:blue',            # composition curve
    'color_ref': 'tab:grey',        # pristine reference line
    'color_grid': UWM_GREY,         # grid lines
    'color_stick': 'tab:red',       # stick overlay (None -> follows 'color')
    'color_metalabel': '0.4',       # "(a)" panel label text
    'color_notcalc': UWM_GOLD,      # missing-cell face colour
    'lw_curve': 1.2,                # composition curve width
    'lw_ref': 0.8,                  # reference line width
    'lw_grid': 0.6,                 # grid line width
    'lw_stick': STICK_LW,           # stick width (shared constant)
    'alpha_ref': 0.8,               # reference line alpha
    'alpha_grid': 0.6,              # grid alpha
    'alpha_stick': STICK_A,         # stick alpha (shared constant)
}


# ---------------------------------------------------------------------------
# Argument parser (mirrors compare_raman.py: grouped, documented flags with
# sensible defaults; mutual-exclusion groups for FWHM and normalisation)
# ---------------------------------------------------------------------------

def _positive(text):
    """Parse a finite positive command-line number."""
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return value


def arguments(argv=None):
    """Read physical, layout and output settings."""
    p = argparse.ArgumentParser(
        description='Plot static Raman spectra in a two-dopant composition matrix.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- input / output ---
    p.add_argument('files', nargs='*',
                   help='CSV files or quoted wildcard patterns. If omitted, '
                        'all matching files in --dir are used.')
    p.add_argument('--dir', default='.',
                   help='Input directory when files are omitted (default: cwd).')
    p.add_argument('--pattern', default='*.csv',
                   help='Input pattern within --dir (default: *.csv).')
    p.add_argument('--reference', type=Path,
                   help='Pristine CSV, also included in the matrix.')
    p.add_argument('--no-reference', action='store_true',
                   help='Disable pristine overlays.')

    # --- composition axes ---
    p.add_argument('--x-dopant', default='Ca',
                   help='Column dopant element (default: Ca).')
    p.add_argument('--y-dopant', default='Cl',
                   help='Row dopant element (default: Cl).')
    p.add_argument('--group-by', choices=['auto', 'tags', 'metadata'], default='auto',
                   help='Grouping strategy: auto (tag if all tagged, else metadata), '
                        'tags, or metadata (default: auto).')

    # --- physical model (what is plotted and under what conditions) ---
    p.add_argument('--temp', type=_positive, default=DEFAULT_TEMP,
                   help='Sample temperature in K for the Bose-Einstein factor '
                        f'(default: {DEFAULT_TEMP:g} K)')
    p.add_argument('--laser-wl', type=_positive, default=DEFAULT_LASER_WL,
                   help='Excitation laser wavelength in nm '
                        f'(default: {DEFAULT_LASER_WL:g} nm)')
    p.add_argument('--raw-act', action='store_true',
                   help='Broaden activities without Stokes conversion '
                        '(legacy S(nu) mode).')
    p.add_argument('--cutoff', type=_positive, default=1.0,
                   help='Stokes zero-weight cutoff in cm^-1 (default: 1.0).')
    fwhm = p.add_mutually_exclusive_group()
    fwhm.add_argument('--fwhm', type=_positive,
                      help='Lorentzian FWHM in cm^-1 (takes precedence over --gamma).')
    fwhm.add_argument('--gamma', type=_positive, default=0.25,
                      help='Lorentzian FWHM in THz (default: 0.25 THz); converted '
                           'to cm^-1 internally via THZCM1.')

    # --- displayed frequency window (cm^-1) ---
    p.add_argument('--freq-min', type=float, default=0.0,
                   help='Lower x-axis limit in cm^-1 (default: 0).')
    p.add_argument('--freq-max', type=float, default=None,
                   help='Upper x-axis limit in cm^-1 (default: auto = highest '
                        'mode rounded up to the nearest 50 cm^-1, mirroring '
                        'compare_raman.py / plot_raman.py).')

    # --- normalisation ---
    norm = p.add_mutually_exclusive_group()
    norm.add_argument('--norm', choices=['each', 'global', 'none'], default='each',
                      help='Normalisation mode: each (default, per-spectrum max), '
                           'global (one common max, preserves relative intensities), '
                           'none (preserve numerical units).')
    norm.add_argument('--norm-each', dest='norm', action='store_const', const='each',
                      help='Alias for --norm each.')
    norm.add_argument('--no-norm', dest='norm', action='store_const', const='none',
                      help='Alias for --norm none.')

    # --- stick spectrum overlay (DEFAULT ON; matches compare_raman.py --sticks) ---
    p.add_argument('--no-sticks', dest='sticks', action='store_false',
                   help='Disable the stick (bar) spectrum overlay (default: on).')
    p.set_defaults(sticks=True)
    p.add_argument('--stick-max', type=float, default=1.0,
                   help='Fraction of the panel max intensity the tallest stick '
                        'reaches (default: 1.0).')

    # --- per-panel peak labels (new, matches compare_raman.py --n-labels) ---
    p.add_argument('--n-labels', type=int, default=4,
                   help='Number of peak frequency labels per panel '
                        '(default: 4; compare_raman.py defaults to 10 '
                        'in waterfall mode).')
    p.add_argument('--min-label-sep', type=float, default=10.0,
                   help='Minimum separation in cm^-1 between labelled peaks '
                        '(default: 10; set 0 to disable merging).')
    p.add_argument('--prominence', type=float, default=0.02,
                   help='Prominence threshold (fraction of peak-to-baseline '
                        'span) used to detect labellable peaks (default: 0.02).')

    # --- layout / styling ---
    p.add_argument('--step', type=_positive, default=0.05,
                   help='Maximum spectral-grid spacing in cm^-1 (default: 0.05).')
    p.add_argument('--grid', choices=['both', 'x', 'y', 'none'], default='x',
                   help='Grid visibility per panel: x (default), both, y, or none '
                        '(mirrors compare_raman.py --grid choices).')
    p.add_argument('--figsize', type=float, nargs='+', default=None,
                   metavar=('WIDTH', 'HEIGHT'),
                   help='Figure size in inches (default: 6 x 6). Pass one value '
                        'for a square figure (e.g. --figsize 8) or two for '
                        'WIDTH HEIGHT (e.g. --figsize 8 10).')
    p.add_argument('--font-size', type=_positive, default=MATRIX_STYLE['font_size'],
                   help=f'Base text size in points (default: {MATRIX_STYLE["font_size"]:g}; '
                        'title/heading sizes scale from this).')
    p.add_argument('--color', default='',
                   help="Composition line color (default: MATRIX_STYLE['color'] = "
                        "'tab:blue'); an empty value uses the config default.")
    p.add_argument('--x-line', type=float, nargs='+', default=[], metavar='X',
                   help='Draw a vertical reference line at each of these '
                        'x-axis (wavenumber) values in cm^-1 '
                        '(e.g. --x-line 420 600).')
    p.add_argument('--y-line', type=float, nargs='+', default=[], metavar='Y',
                   help='Draw a horizontal reference line at each of these '
                        'y-axis values (e.g. --y-line 1.0 0.5). '
                        'New; matches compare_raman.py --y-line.')
    p.add_argument('--ylim', type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'),
                   help='Shared intensity limits; clipping is rejected.')
    p.add_argument('--title', default=None,
                   help='Figure title (default: auto-generated two-line title). '
                        'Pass an empty string --title "" to omit the title '
                        'entirely (matches compare_raman.py), \'line1\' or '
                        "'line2' to show only the formula line or only the "
                        'parameters line of the auto-generated title, or any '
                        'other string for a custom title.')

    # --- output ---
    p.add_argument('--out', type=Path, default=None,
                   help='Output prefix (default: compare_matrix_<formula>_<stokes|'
                        'activity>_<norm>, built on compare_raman.py '
                        'build_output_name but prefixed compare_matrix to avoid '
                        'clashing with compare_raman.py outputs).')
    p.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'],
                   default=['png'],
                   help="Output formats (default: png). Pass 'png pdf' for both.")
    p.add_argument('--dpi', type=int, default=MATRIX_STYLE['dpi'],
                   help=f'Raster resolution for PNG/SVG (default: {MATRIX_STYLE["dpi"]}).')
    p.add_argument('--export-data', action='store_true',
                   help='Write full/window arrays to NPZ.')

    a = p.parse_args(argv)

    # --- validation ---
    if a.x_dopant == a.y_dopant or any(
            not re.fullmatch(r'[A-Z][a-z]?', s) for s in (a.x_dopant, a.y_dopant)):
        p.error('Choose two distinct element symbols.')
    if a.freq_max is not None and a.freq_min >= a.freq_max:
        p.error('--freq-min must be < --freq-max when both are given.')
    if any(not np.isfinite(v) for v in a.x_line):
        p.error('--x-line values must be finite.')
    if any(not np.isfinite(v) for v in a.y_line):
        p.error('--y-line values must be finite.')
    if a.dpi <= 0:
        p.error('dpi must be positive.')
    if a.font_size < 4:
        p.error('font-size must be at least 4 points.')
    if a.figsize is not None and len(a.figsize) not in (1, 2):
        p.error('--figsize takes 1 (square) or 2 (WIDTH HEIGHT) values.')
    if a.stick_max <= 0 or a.stick_max > 1:
        p.error('--stick-max must be in (0, 1].')
    if a.n_labels < 0:
        p.error('--n-labels must be >= 0.')
    if a.min_label_sep < 0:
        p.error('--min-label-sep must be >= 0.')
    if not (0.0 < a.prominence < 1.0):
        p.error('--prominence must be in (0, 1).')
    return a


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def input_paths(a):
    """Expand input patterns and deduplicate repeated paths."""
    paths = []
    for pattern in a.files or [str(Path(a.dir) / a.pattern)]:
        matches = sorted(_glob.glob(pattern))
        if not matches:
            raise ValueError(f'No input matches {pattern!r}.')
        paths.extend(Path(m).resolve() for m in matches)
    if a.reference:
        paths.append(a.reference.resolve())
    paths = list(dict.fromkeys(paths))
    for path in paths:
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
    return paths


def read_csv(path):
    """
    Read 3/4-column mode rows and comment metadata.

    Thin wrapper over the shared :func:`plot_raman.read_raman_csv` so that
    both ``compare_raman.py`` and this script parse the exact same CSV
    dialect (Formula / Dopants / File_Label / E_field comment metadata,
    3- or 4-column rows with an optional ``Mode, Freq_cm-1, Activity, Irrep``
    header line). Returns (frequencies, activities, metadata) where the
    metadata dict keeps all values as strings (compare_raman.py does the
    same, see its ``load_csv`` wrapper).
    """
    freqs, acts, _irreps, meta = read_raman_csv(path)
    e = meta.get('e_field')
    meta['e_field'] = '' if e is None else str(e)
    return freqs, acts, meta


def load_sample(path, a):
    """Determine composition from metadata and recognise pristine tag conventions."""
    frequencies, activities, meta = read_csv(path)
    label_tags = composition_tags(meta.get('file_label', ''), a.x_dopant, a.y_dopant)
    file_tags = composition_tags(path.stem, a.x_dopant, a.y_dopant)
    if label_tags is not None and file_tags is not None and label_tags != file_tags:
        raise ValueError(f'{path.name}: File_Label and filename tags disagree.')
    tags = label_tags if label_tags is not None else file_tags
    explicit = a.reference is not None and path == a.reference.resolve()
    text = meta.get('dopants', '').strip()
    zero_marker = (tags is not None and tags == (Decimal(0), Decimal(0)))
    values = {}
    if text and text.lower() not in {'undoped', 'none', 'pristine'}:
        for token in text.split():
            match = re.fullmatch(r'([A-Z][a-z]?)=(.+)', token)
            if not match:
                raise ValueError(f'{path.name}: invalid Dopants token {token!r}.')
            species, value = match.groups()
            try:
                number = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f'{path.name}: invalid concentration {value!r}.') from exc
            if not number.is_finite() or number < 0 or species in values:
                raise ValueError(f'{path.name}: invalid/repeated concentration {token!r}.')
            values[species] = number
        if any(v != 0 for k, v in values.items() if k not in {a.x_dopant, a.y_dopant}):
            raise ValueError(f'{path.name}: another nonzero dopant lies outside this composition plane.')
    elif not text and not (explicit or zero_marker
                            or meta.get('file_label', '').lower() in {'undoped', 'pristine'}):
        raise ValueError(f'{path.name}: missing Dopants metadata and no pristine identifier.')
    xy = tuple(values.get(k, Decimal(0)) for k in [a.x_dopant, a.y_dopant])
    if (explicit or zero_marker) and any(xy):
        raise ValueError(f'{path.name}: pristine identifier conflicts with nonzero metadata.')
    if tags is not None and any((tag == 0) != (value == 0) for tag, value in zip(tags, xy)):
        raise ValueError(f'{path.name}: zero/nonzero tag disagrees with Dopants metadata.')
    return Sample(path, frequencies, activities, meta, xy, tags)


# ---------------------------------------------------------------------------
# Composition grouping
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """Mode data, metadata concentrations, and optional composition identifiers."""
    path: Path
    frequencies: np.ndarray
    activities: np.ndarray
    metadata: dict
    concentration: tuple
    tags: object


def composition_tags(text, x_species, y_species):
    """Read complete element-code tags without interpreting codes as fractions."""
    pattern = (rf'(?<![A-Za-z0-9]){re.escape(x_species)}(\d+(?:\.\d+)?)-'
               rf'{re.escape(y_species)}(\d+(?:\.\d+)?)(?![A-Za-z0-9])')
    matches = re.findall(pattern, text)
    if len(matches) > 1:
        raise ValueError(f'Multiple composition tags in {text!r}.')
    return tuple(Decimal(t) for t in matches[0]) if matches else None


def check_metadata(samples):
    """Reject inconsistent known host/field metadata; warn about missing fields."""
    for key in ['formula', 'e_field']:
        values = [s.metadata.get(key, '') for s in samples]
        known = [v for v in values if v != '']
        if key == 'e_field':
            try:
                numeric = [float(v) for v in known]
                consistent = not numeric or (
                    np.isfinite(numeric).all()
                    and np.allclose(numeric, numeric[0], rtol=1e-8, atol=1e-12))
            except ValueError:
                consistent = len(set(known)) <= 1
        else:
            consistent = len(set(known)) <= 1
        if not consistent:
            raise ValueError(f'Inconsistent {key} metadata: {sorted(set(known))}')
        if len(known) < len(samples):
            warnings.warn(f'Some {key} metadata are missing; verify input comparability.')


def representative(values, label):
    """Choose the most decimal-precise value after checking rounding compatibility."""
    chosen = min(values, key=lambda v: v.as_tuple().exponent)
    for value in values:
        tolerance = Decimal('0.5') * Decimal(10) ** value.as_tuple().exponent
        if abs(value - chosen) > tolerance:
            raise ValueError(
                f'{label}: inconsistent metadata values {sorted(set(values))}; '
                'inspect inputs or use --group-by metadata.')
    if len(set(values)) > 1:
        warnings.warn(
            f'{label}: metadata values {sorted(set(values))} share a tag; '
            f'heading uses {chosen}. Verify these describe one intended level.')
    return chosen


def arrange(samples, a):
    """Infer axis groups, representatives, available cells and pristine reference."""
    complete = all(s.tags is not None for s in samples)
    if a.group_by == 'tags' and not complete:
        raise ValueError('--group-by tags requires complete tags for every input.')
    mode = 'tags' if (a.group_by == 'tags' or (a.group_by == 'auto' and complete)) else 'metadata'
    if a.group_by == 'auto' and not complete:
        warnings.warn('Some inputs lack composition tags; using numerical metadata for all samples.')
    keys = [s.tags if mode == 'tags' else s.concentration for s in samples]
    axes = [sorted({key[d] for key in keys}) for d in (0, 1)]
    headings = []
    for d, levels in enumerate(axes):
        mapping = {}
        for level in levels:
            vals = [s.concentration[d] for s, key in zip(samples, keys) if key[d] == level]
            mapping[level] = (representative(vals, f'{[a.x_dopant, a.y_dopant][d]} tag {level}')
                              if mode == 'tags' else level)
        if len(set(mapping.values())) != len(mapping):
            raise ValueError('Distinct tags yield identical concentration headings; '
                             'inspect tags or use --group-by metadata.')
        axes[d].sort(key=lambda level: mapping[level])
        headings.append(mapping)
    cells = {}
    for index, key in enumerate(keys):
        if key in cells:
            raise ValueError(f'Duplicate composition cell: {samples[cells[key]].path.name} '
                             f'and {samples[index].path.name}.')
        cells[key] = index
    pristine = [i for i, s in enumerate(samples) if not any(s.concentration)]
    if len(pristine) > 1:
        raise ValueError('Multiple pristine inputs; select one reference.')
    reference = pristine[0] if pristine else None
    if reference is None and not a.no_reference:
        warnings.warn('No pristine spectrum found; reference overlays omitted.')
    return mode, axes, headings, cells, reference


# ---------------------------------------------------------------------------
# Spectral model (Stokes weights + Lorentzian broadening + normalisation)
# ---------------------------------------------------------------------------

def mode_weights(sample, a):
    """
    Return activity or Stokes stick weights.

    Uses the shared :func:`plot_raman.activity_to_stokes_intensity` so the
    physics is identical to ``compare_raman.py`` / ``plot_raman.py`` (same
    cutoff, same laser/temperature handling, same stable Bose evaluation).
    """
    if a.raw_act:
        return sample.activities.copy()
    return activity_to_stokes_intensity(
        sample.frequencies,
        sample.activities,
        temp=a.temp,
        laser_wl=a.laser_wl,
        freq_cutoff_cm1=a.cutoff,
    )


def evaluate(grid, frequencies, weights, fwhm):
    """Sum area-normalised Lorentzians from every positive-weight mode."""
    result = np.zeros_like(grid, dtype=float)
    for frequency, weight in zip(frequencies, weights):
        if weight > 0:
            result += lorentzian(grid, frequency, weight, fwhm)
    return result


def uniform_grid(lower, upper, step):
    """Build a bounded-size grid including both endpoints."""
    count = max(2, int(np.ceil((upper - lower) / step)) + 1)
    if count > MAX_POINTS:
        raise ValueError(
            f'Requested grid exceeds {MAX_POINTS} points; adjust step/window/FWHM.')
    return np.linspace(lower, upper, count)


def spectra(samples, a):
    """
    Compute full-grid normalisation and directly evaluated window spectra.

    The resolution grid spans every mode plus a margin (50 cm^-1 or 8x FWHM)
    and always contains the displayed window, so a custom ``--freq-min``/
    ``--freq-max`` zoom can never truncate a Lorentzian peak/tail (or change
    the peak used for normalisation) before the max is determined -- the
    same design as ``compare_raman.py build_all_spectra``.
    """
    width = a.fwhm if a.fwhm is not None else a.gamma * THZCM1
    frequencies = np.concatenate([s.frequencies for s in samples])
    padding = max(50.0, 8.0 * width)
    step = min(a.step, width / 40.0)
    full = np.unique(np.r_[
        uniform_grid(frequencies.min() - padding, frequencies.max() + padding, step),
        frequencies,
    ])
    # Auto (default) display range mirrors plot_raman.py / compare_raman.py:
    # lower bound 0, upper bound the highest mode rounded up to the nearest
    # 50 cm^-1, so the axis lands on a "nice" value rather than a raw one.
    auto_freq_max = float(np.ceil(frequencies.max() / 50) * 50)
    freq_max = a.freq_max if a.freq_max is not None else auto_freq_max
    if a.freq_min >= freq_max:
        raise ValueError(
            f'--freq-min ({a.freq_min:g}) must be < --freq-max ({freq_max:g}).')
    window = uniform_grid(a.freq_min, freq_max, step)
    weights = [mode_weights(s, a) for s in samples]
    raw = np.array([evaluate(full, s.frequencies, w, width) for s, w in zip(samples, weights)])
    if not np.isfinite(raw).all() or not np.any(raw > 0):
        raise ValueError('Spectra are nonfinite or all zero; check input/model settings.')
    divisors = (raw.max(axis=1) if a.norm == 'each'
                else np.full(len(samples), raw.max() if a.norm == 'global' else 1.0))
    if np.any(divisors == 0):
        warnings.warn('A zero spectrum remains zero; its normalisation divisor is recorded as 1.')
        divisors = divisors.copy()
        divisors[divisors == 0] = 1
    curves = np.array([
        evaluate(window, s.frequencies, w, width) / d
        for s, w, d in zip(samples, weights, divisors)
    ])
    return dict(
        width=width, full=full, raw=raw, weights=weights, divisors=divisors,
        window=window, curves=curves,
        freq_min=a.freq_min, freq_max=freq_max,
    )


# ---------------------------------------------------------------------------
# Title (new: auto-generated two-line title via shared plot_raman builder,
# with compare_raman.py's --title 'line1'/'line2'/'' selection semantics)
# ---------------------------------------------------------------------------

def build_plot_title(samples, a, gamma):
    """
    Resolve the overall figure title.

    Mirrors compare_raman.py ``build_plot_title``: the default (``--title``
    omitted) gives the auto-generated two-line title (formula line +
    parameters line) via the shared :func:`plot_raman.build_raman_title`;
    ``--title ""`` omits the title entirely; ``--title 'line1'`` / ``'line2'``
    selects one line of that auto title; any other value is shown as-is.
    Returns ``(title, title_line)`` where ``title`` is ``None`` when omitted.
    """
    title_line = "both"
    # --title semantics (matching compare_raman.py):
    #   None (default)   -> auto-generated two-line title (built below)
    #   ""               -> omit the title entirely
    #   "line1"/"line2"  -> one line of the auto title (select below)
    #   anything else    -> custom string, shown as-is
    if a.title == "":
        return None, title_line
    if a.title in ("line1", "line2"):
        title_line = a.title
    elif a.title is not None:
        return a.title, title_line
    # a.title is None (default): fall through to the auto-generated title.

    formulae = [s.metadata.get('formula', '') for s in samples]
    unique_f = set(f for f in formulae if f)
    efields = [s.metadata.get('e_field', '') for s in samples]
    unique_ef = set(e for e in efields if e)

    if len(unique_f) == 1:
        formula = list(unique_f)[0]
        # A dopant matrix shares one base host formula but varies the dopant
        # set per file; union the species across all files (first-seen order)
        # to recover the full symbolic formula (same logic as
        # compare_raman.py build_plot_title).
        dopant_species = []
        for s in samples:
            for entry in re.split(r'\s+', (s.metadata.get('dopants') or '').strip()):
                if '=' in entry:
                    sp = entry.split('=', 1)[0]
                    if sp not in dopant_species:
                        dopant_species.append(sp)
        dopants = " ".join(f"{sp}=0" for sp in dopant_species)
        E_val = float(list(unique_ef)[0]) if len(unique_ef) == 1 else None
        title = build_raman_title(
            chem={"formula": formula, "dopants": dopants},
            E=E_val,
            gamma=gamma,
            temp=a.temp,
            raw_act=a.raw_act,
            suffix="Comparison",
            show_legend=False,
        )
    else:
        parts = []
        if len(unique_ef) == 1:
            parts.append(f"E = {list(unique_ef)[0]} eV/Å")
        parts.append(f"FWHM = {gamma:.2f} cm$^{{-1}}$")
        parts.append(f"T = {a.temp:g} K" if not a.raw_act else "S(ν)")
        title = "Raman Composition Matrix\n" + " | ".join(parts)
    return title, title_line


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def panel_letter(index):
    """Return a, ..., z, aa, ... for zero-based panel indices."""
    text = ''
    while True:
        index, remainder = divmod(index, 26)
        text = chr(97 + remainder) + text
        if index == 0:
            return text
        index -= 1


def _resolve_figsize(a):
    """Resolve the figure size; default is MATRIX_STYLE['figsize'], --figsize
    overrides (1 value = square, or 2 = WIDTH HEIGHT)."""
    fig_w, fig_h = MATRIX_STYLE['figsize']
    if a.figsize is not None:
        if len(a.figsize) == 1:
            fig_w = fig_h = float(a.figsize[0])
        elif len(a.figsize) == 2:
            fig_w, fig_h = (float(v) for v in a.figsize)
    return fig_w, fig_h


def draw(samples, axes, headings, cells, reference, result, a, plot_title, title_line):
    """
    Draw shared-scale spectral panels with external composition headings.

    Applies the shared publication-ready style (plot_raman.set_style,
    compact=True -- the same call compare_raman.py makes), then overlays
    optional stick spectra, per-panel peak labels, and x/y reference lines
    (new: --sticks, --n-labels, --y-line).
    """
    # Shared style (identical to compare_raman.py). set_style sets the GLOBAL
    # rcParams -- base/tick/legend font sizes, spine + tick geometry, grid,
    # weights and the UWM palette -- from plot_raman (single source of truth
    # shared by both scripts). What set_style cannot express (this figure's
    # subplot margins, inter-panel spacing, axis-title offsets, heading/title
    # size multipliers and the curve palette) is per-figure and lives in
    # MATRIX_STYLE; the CLI flags that override a single knob (--font-size,
    # --color) are resolved here.
    S = MATRIX_STYLE
    set_style(compact=True)
    fs = a.font_size                       # base size (--font-size > MATRIX_STYLE)
    fs_minor = S['font_size_minor']        # secondary text size (tick/peak/panel tags)
    title_fs = fs * S['title_size']       # suptitle size
    head_fs = fs * S['heading_size']      # dopant-heading size
    bold = S['bold_headings']
    bold_labels = S['bold_axis_labels']    # bold "Intensity" / "Raman Shift"
    # Element palette. A CLI --color wins over the MATRIX_STYLE default.
    color = a.color if a.color else S['color']
    color_ref = S['color_ref']
    color_grid = S['color_grid']
    color_stick = S['color_stick'] or color
    color_metalabel = S['color_metalabel']
    color_notcalc = S['color_notcalc']
    lw_curve, lw_ref, lw_grid = S['lw_curve'], S['lw_ref'], S['lw_grid']
    lw_stick, alpha_stick = S['lw_stick'], S['alpha_stick']
    alpha_ref, alpha_grid = S['alpha_ref'], S['alpha_grid']

    plt.rcParams.update({
        'font.size': fs, 'axes.labelsize': fs,
        'xtick.labelsize': fs_minor, 'ytick.labelsize': fs_minor,
        # set_style already sets the mathtext title/label weights; keep them
        # consistent with the per-figure bold toggles (supxlabel/supylabel and
        # the dopant headings get their weight passed explicitly below, since
        # they do NOT inherit axes.labelweight).
        'axes.titleweight': 'bold' if S['bold_title'] else 'normal',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'grid.color': color_grid, 'grid.linewidth': lw_grid,
        'grid.alpha': alpha_grid,
    })

    xs, ys = axes
    fig_w, fig_h = _resolve_figsize(a)
    fig, panels = plt.subplots(len(ys), len(xs), sharex=True, sharey=True,
                               squeeze=False, figsize=(fig_w, fig_h))
    # Top band is chosen by how many title lines are shown: none when the title
    # is omitted, one for 'line1'/'line2', two for the default auto (or custom)
    # title. The single suptitle_y (anchored with a fixed va) is shared by all
    # cases -- only the band height differs -- so there is one knob to nudge per
    # case instead of pairing margin_top with a per-case suptitle_y.
    if not plot_title:
        margin_top = S['margin_top_notitle']
    elif title_line in ("line1", "line2"):
        margin_top = S['margin_top_1line']
    else:  # 'both' (default auto two-line, or a custom title)
        margin_top = S['margin_top_2line']

    # Tight layout: reserve just enough margin for the decorations around the
    # panel grid -- suptitle + x-dopant column headers above, x-tick labels
    # + x-axis title + legend below, y-axis title on the left, and the
    # y-dopant row labels on the right -- so the panels fill the figure instead
    # of floating in white space. All values come from MATRIX_STYLE.
    fig.subplots_adjust(left=S['margin_left'], right=S['margin_right'],
                        bottom=S['margin_bottom'], top=margin_top,
                        wspace=S['wspace'], hspace=S['hspace'])

    window = result['window']
    curves = result['curves']
    freq_min = result['freq_min']
    freq_max = result['freq_max']
    weights = result['weights']

    maximum = float(curves.max())
    limits = (0.0, max(maximum * S['y_headroom'], 1e-20))
    if a.ylim is not None:
        limits = tuple(a.ylim)
        if not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError('--ylim must be finite and ordered (ymin < ymax).')

    overlay = reference is not None and not a.no_reference
    show_panel_labels = a.n_labels > 0
    # Row order: y-dopant increases DOWNWARD from the pristine (0,0) corner,
    # so the undoped parent composition sits in the upper-left panel and
    # doped levels fall below it (standard composition-matrix reading order).
    for row, y in enumerate(ys):
        for col, ca in enumerate(xs):
            ax = panels[row, col]
            index = cells.get((ca, y))
            if index is None:
                ax.set_facecolor(color_notcalc)
                ax.text(.5, .5, 'Not calculated', transform=ax.transAxes,
                        ha='center', va='center', color=color_metalabel,
                        fontsize=fs_minor)
            else:
                if overlay and index != reference:
                    ax.plot(window, curves[reference], color=color_ref, ls='--',
                            lw=lw_ref, alpha=alpha_ref, zorder=2)
                ax.plot(window, curves[index], color=color, lw=lw_curve, zorder=3)

                # Optional stick spectrum overlay (new, matches
                # compare_raman.py --sticks): scale each stick to a fraction
                # of the panel max so it stays within the shared y-limits.
                if a.sticks:
                    freqs = samples[index].frequencies
                    w = weights[index]
                    wmax = float(w.max()) if w.size else 0.0
                    if wmax > 0:
                        stick_scale = a.stick_max / wmax
                        for f_val, w_val in zip(freqs, w):
                            if w_val > 0 and freq_min <= f_val <= freq_max:
                                ax.vlines(f_val, 0.0, w_val * stick_scale,
                                          color=color_stick, lw=lw_stick,
                                          alpha=alpha_stick, zorder=2)

                # per-panel peak labels (new, matches
                # compare_raman.py --n-labels; prominence-ranked,
                # split-Lorentzian shoulders merged).
                if show_panel_labels:
                    vis = curves[index]
                    for pk in select_labeled_peaks(
                            vis, window,
                            n_labels=a.n_labels,
                            min_label_sep=a.min_label_sep,
                            prominence_frac=a.prominence,
                    ):
                        peak_freq = float(window[pk])
                        peak_y = float(vis[pk])
                        label_y = min(peak_y + S['peak_label_offset'] * (limits[1] - limits[0]),
                                      S['peak_label_cap'] * limits[1])
                        ax.text(peak_freq, label_y, f"{peak_freq:.1f}",
                                color=color, fontsize=fs_minor, fontweight="bold",
                                va="bottom", ha="center", zorder=5, linespacing=1.0)

            for value in a.x_line:
                ax.axvline(value, color=color_ref, ls='--', lw=lw_ref,
                           alpha=alpha_ref, zorder=1)
            for value in a.y_line:
                ax.axhline(value, color=color_ref, ls='--', lw=lw_ref,
                           alpha=alpha_ref, zorder=1)
            ax.set_xlim(freq_min, freq_max)
            ax.set_ylim(*limits)
            ax.xaxis.set_major_locator(MaxNLocator(S['n_ticks']))
            ax.yaxis.set_major_locator(MaxNLocator(S['n_ticks']))
            # Ticks only on the bottom/left spines. Hiding the top/right spines
            # removes their tick marks too (tick_params(top/right=False) only
            # drops the labels), giving a clean panel look. set_style already
            # sets 'in' tick direction and 4pt tick size on the visible spines.
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(top=False, right=False)
            if a.grid == 'both':
                ax.grid(True, alpha=alpha_grid, lw=lw_grid)
            elif a.grid == 'x':
                ax.grid(axis='x', alpha=alpha_grid, lw=lw_grid)
            elif a.grid == 'y':
                ax.grid(axis='y', alpha=alpha_grid, lw=lw_grid)
            # 'none' -> no grid.
            _px, _py = S['panel_label_pos']
            ax.text(_px, _py, f'({panel_letter(row * len(xs) + col)})',
                    transform=ax.transAxes, va='top',
                    bbox={'facecolor': 'white', 'edgecolor': 'none',
                          'alpha': S['panel_label_alpha'],
                          'pad': S['panel_label_pad']}, zorder=6)
            # Composition headings. Both axes use bold mathtext (\mathbf) so the
            # subscripted element symbols are bold (matching the bold formula
            # titles), and both are drawn as plain text with an explicit bold
            # weight (NOT ax.set_title, which would apply axes.titleweight only
            # to the x-axis and leave the two inconsistent). \mathbf is swapped
            # for \mathrm when MATRIX_STYLE['bold_headings'] is False so the
            # headings take a uniform (non-bold) weight.
            mstyle = r'\mathbf' if bold else r'\mathrm'
            _hx, _hy = S['x_heading_pos']
            # x-dopant: centered above each top-row panel.
            if row == 0:
                ax.text(_hx, _hy,
                        rf'$x_{{{mstyle}{{{a.x_dopant}}}}}={float(headings[0][ca]):g}$',
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=head_fs, fontweight='bold' if bold else 'normal',
                        zorder=6)
            # y-dopant: right edge of each row, reading bottom-to-top
            # (rotation=90). With pristine at the top and composition rising
            # downward, bottom-to-top reading matches reading the value column
            # from its smallest to largest member.
            _vyx, _vy = S['y_heading_pos']
            if col == len(xs) - 1:
                ax.text(_vyx, _vy,
                        rf'$y_{{{mstyle}{{{a.y_dopant}}}}}={float(headings[1][y]):g}$',
                        transform=ax.transAxes, rotation=90,
                        ha='left', va='center', fontsize=head_fs,
                        fontweight='bold' if bold else 'normal', zorder=6)

    quantity = 'Activity' if a.raw_act else 'Intensity'
    ylabel = {'each': f'{quantity} (simulated, normalized per spectrum)',
              'global': f'{quantity} (simulated, normalized to global max)',
              'none': f'{quantity} (arb. units, absolute)'}[a.norm]
    # Axis titles use the base font size and sit a MATRIX_STYLE-controlled
    # distance from the spines, so "Intensity" and "Raman Shift" are the same
    # size and consistently placed. fig.supxlabel/supylabel do NOT inherit
    # axes.labelweight (that only styles ax.set_xlabel), so the bold weight is
    # passed explicitly -- that is why it was previously missing here while the
    # \mathbf dopant headings came out bold.
    _lw = 'bold' if S['bold_axis_labels'] else 'normal'
    fig.supylabel(ylabel, x=S['ylabel_x'], fontsize=fs, fontweight=_lw)
    fig.supxlabel(r'Raman Shift (cm$^{-1}$)', y=S['xlabel_y'], fontsize=fs,
                  fontweight=_lw)

    # Legend: reflect the overlays actually drawn.
    handles = [Line2D([], [], color=color, lw=lw_curve, label='Indicated composition')]
    if a.sticks:
        handles.append(Line2D([], [], color=color_stick, lw=lw_stick,
                              alpha=alpha_stick, label='Stick spectrum'))
    if overlay:
        handles.append(Line2D([], [], color=color_ref, ls='--', lw=lw_ref,
                              label='Pristine reference'))
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=S['legend_anchor'],
               frameon=S['legend_frame'],
               ncol=len(handles) if S['legend_ncol'] is None else S['legend_ncol'],
               fontsize=fs_minor)
    if plot_title:
        fig.suptitle(plot_title, fontsize=title_fs, y=S['suptitle_y'],
                     va=S['suptitle_va'],
                     fontweight='bold' if S['bold_title'] else 'normal')
    return fig


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Normalisation-mode codes used by compare_raman.build_output_name for the
# default filename. Each mode maps to the same code compare_raman.py uses when
# it builds its output name (see its norm_spectra / build_output_name).
_NORM_CODE = {'each': 'enorm', 'global': 'gnorm', 'none': 'abs'}


def _default_out(samples, a):
    """
    Default output prefix, built on the shared compare_raman.build_output_name
    but with a ``compare_matrix`` prefix so it cannot clash with
    compare_raman.py's ``compare_<formula>_...`` outputs.

    Produces ``compare_matrix_<formula>_<stokes|activity>_<norm>`` (the
    extensionless prefix; save() appends each requested format). When the user
    passes --out, that value is used verbatim instead.
    """
    if a.out is not None:
        out = Path(a.out)
    else:
        name = build_output_name(
            [s.metadata for s in samples],
            _NORM_CODE[a.norm],
            a.raw_act,
        )
        # build_output_name returns "compare_<formula>_<suffix>_<norm>.png";
        # promote the leading "compare_" to "compare_matrix_" to keep the two
        # scripts' default filenames distinct.
        if name.startswith('compare_'):
            name = 'compare_matrix_' + name[len('compare_'):]
        out = Path(name)
    # Drop any image-file extension so save() can append each format cleanly.
    return out.with_suffix('') if out.suffix.lower() in {'.pdf', '.png', '.svg'} else out


def save(fig, samples, layout, result, a):
    """Write figure files, grouping provenance and optional numerical arrays."""
    mode, axes, headings, cells, reference = layout
    width = result['width']
    full = result['full']
    raw = result['raw']
    weights = result['weights']
    divisors = result['divisors']
    window = result['window']
    curves = result['curves']

    prefix = _default_out(samples, a)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in dict.fromkeys(a.formats):
        path = Path(str(prefix) + '.' + fmt)
        # bbox_inches='tight' crops residual outer white space for a compact
        # figure (compare_raman.py / plot_raman.py do the same).
        fig.savefig(path, dpi=a.dpi, facecolor='white', bbox_inches='tight')
        paths.append(str(path))
    manifest = {
        'script': 'compare_raman_matrix.py',
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'grouping_mode': mode,
        'fwhm_cm1': width,
        'normalization_grid_cm1': [float(full[0]), float(full[-1])],
        'display_window_cm1': [result['freq_min'], result['freq_max']],
        'axis_levels': [[{'group': str(level), 'heading': str(headings[d][level])}
                         for level in axes[d]] for d in (0, 1)],
        'pristine_index': reference,
        'reference_overlaid': reference is not None and not a.no_reference,
        'samples': [
            {'file': str(s.path),
             'metadata': s.metadata,
             'concentration': [str(v) for v in s.concentration],
             'tags': [str(t) for t in s.tags] if s.tags is not None else None,
             'normalization_divisor': float(divisors[i])}
            for i, s in enumerate(samples)
        ],
        'figures': paths,
        'notes': [
            'Equal categorical panel spacing; no composition interpolation.',
            'Common imposed FWHM is not a calculated lifetime linewidth.',
            'Spectra use the shared plot_raman Stokes conversion and Lorentzian '
            'broadening (identical to compare_raman.py / plot_raman.py).',
        ],
    }
    Path(str(prefix) + '.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if a.export_data:
        arrays = {
            'frequency_cm1': full, 'raw_spectra': raw,
            'display_spectra': raw / divisors[:, None],
            'window_frequency_cm1': window, 'window_spectra': curves,
            'normalization_divisors': divisors,
            'files': np.array([str(s.path) for s in samples]),
        }
        for i, (sample, weight) in enumerate(zip(samples, weights)):
            arrays[f'mode_frequency_{i}'] = sample.frequencies
            arrays[f'mode_weight_{i}'] = weight
        np.savez_compressed(str(prefix) + '.npz', **arrays)
    return paths


# ---------------------------------------------------------------------------
# Main orchestration (mirrors compare_raman.py main())
# ---------------------------------------------------------------------------

def main(argv=None):
    """Read spectra, infer composition groups, and save static comparison figures."""
    a = arguments(argv)
    try:
        samples = [load_sample(path, a) for path in input_paths(a)]
        check_metadata(samples)
        layout = arrange(samples, a)
        result = spectra(samples, a)
        _, axes, headings, cells, reference = layout
        plot_title, title_line = build_plot_title(samples, a, result['width'])
        if title_line in ("line1", "line2"):
            plot_title = select_title_line(plot_title, title_line)
        fig = draw(samples, axes, headings, cells, reference, result, a,
                   plot_title, title_line)
        try:
            outputs = save(fig, samples, layout, result, a)
        finally:
            plt.close(fig)
    except (OSError, ValueError, InvalidOperation) as exc:
        raise SystemExit(f'Error: {exc}') from exc

    print(f'Comparing {len(samples)} spectra '
          f'[{("Raw Activity S(nu)" if a.raw_act else f"Stokes Intensity (T={a.temp:g} K, {a.laser_wl:g} nm)")}]:')
    for s in samples:
        print(f'  {s.path.name}')
    print(f'Matrix: {len(axes[1])} rows x {len(axes[0])} columns; grouping: {layout[0]}')
    print(f'Normalisation: {a.norm}; Lorentzian FWHM: {result["width"]:.4f} cm^-1')
    for path in outputs:
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
