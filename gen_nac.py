#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
gen_nac.py -- scaffold the RASCBEC NAC multi-direction phonopy runs.

Run from ``Project/04_phon/<composition>``. The script reads the phonopy
inputs for the composition and builds, under ``Project/05_rascbec/<composition>/``,
one self-contained phonopy project per NAC direction:

    05_rascbec/<composition>/
    ├── nac_directions.json        # manifest: directions, weights, provenance
    ├── dir01/
    │   ├── POSCAR FORCE_CONSTANTS BORN   # symlinks (no file duplication)
    │   ├── direction.conf        # phonopy config: NAC = .TRUE. + Q_DIRECTION
    │   ├── qpoints.yaml          # produced by phonopy (one Gamma point)
    │   └── phonopy.log
    ├── dir02/ ...
    └── dirNN/                    # NN = number of directions for the order

Each direction is one Lebedev-Laikov quadrature point on the unit sphere; the
phonopy ``Q_DIRECTION`` for a direction is the lattice-transform of its
Cartesian unit vector.

This module is also the **single source of truth for the NAC quadrature**.
``RASCBEC_phonopy.py`` imports ``lebedev_points`` / ``lebedev_subset`` /
``ORDER_TO_POINT_COUNT`` / ``POINT_COUNT_TO_ORDER`` from here, so the weights
the generator writes into ``nac_directions.json`` and the weights the consumer
validates against are guaranteed identical (both come from
``scipy.integrate.lebedev_rule``) and can never drift apart.

Quadrature orders
-----------------
The node set and weights are taken from ``scipy.integrate.lebedev_rule``
(Parrish ``getLebedevSphere``); nothing is hand-coded. Supported orders are
odd integers 3 .. 15 (the exact list and point counts are in
``ORDER_TO_POINT_COUNT``). Order 15 (86 dirs) is the cap: it keeps every
direction folder two-digit (``dir01`` ... ``dir86``); higher orders (17 ->
110 dirs, 19 -> 146 dirs, ...) would push the folder names to three digits,
and above order 15 the extra directions add little value for a weighted
spherical average. Direction geometry (verified against scipy's node sets):

    order 3  ->  6 dirs : ±coordinate axes
    order 5  -> 14 dirs : axes + body diagonals
    order 7  -> 26 dirs : axes + face diagonals + body diagonals   [default]
    order 9  -> 38 dirs : axes + 24 in-plane intermediate dirs + body diagonals
    order 11 -> 50 dirs : the above refined with more in-plane / body-diagonal dirs
    order 13 -> 74 dirs : further refinement
    order 15 -> 86 dirs : the highest supported rule

``order 7`` is the classic 26-point rule. Note that containment between
orders is NOT contiguous: e.g. order 7 is a subset of orders 11 and 13 (so a
lower-order rule can be re-weighted from that data), but NOT of orders 9 or
15. Order 13 has one negative weight (still a valid quadrature, but the
weighted average is then an extrapolation, not a convex one).

The independent per-direction phonopy jobs run in parallel (a process pool,
``--jobs``). Each job is an external ``phonopy`` process, so the work is
CPU-parallel regardless of the GIL. Because ``phonopy`` links a threaded BLAS
and by default spawns a thread per core, N parallel phonopy processes would
each grab every core and oversubscribe the machine (often *slower* than a few
well-threaded workers). The script therefore caps threads-per-process so that
``workers * threads ~= core count`` (auto, or set ``--threads`` explicitly).

Every run *rewrites* the target to hold exactly the requested set of
directions. It (re)writes the manifest, re-runs every direction from scratch,
and removes any stale ``dirNN`` folders left over from a previous run with a
different ``--lebedev-order`` (e.g. ``dir15`` ... ``dir26`` when switching from
order 7 to order 5) -- so the target always matches the current inputs and
settings, with nothing extra left behind.

Usage:
    python gen_nac.py                           # order 7 (26 dirs), auto parallelism
    python gen_nac.py --lebedev-order 5         # order 5 (14 dirs)
    python gen_nac.py --lebedev-order 11        # order 11 (50 dirs)
    python gen_nac.py --jobs 8                  # cap parallelism at 8 workers
    python gen_nac.py --jobs 8 --threads 4      # 8 workers x 4 threads (<= cores)

Exit codes: 0 on success, non-zero on any user/configuration error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

#: Manifest schema version (bumped only if the manifest structure changes).
MANIFEST_SCHEMA = 1


# ===========================================================================
# NAC quadrature (Lebedev-Laikov rule) -- SHARED with RASCBEC_phonopy.py
# ===========================================================================
# Imported by RASCBEC_phonopy.py so the generator and the consumer can never
# disagree on the node set, the weights, or the order<->point-count mapping.

#: Default quadrature order (the classic 26-point rule).
DEFAULT_ORDER = 7

#: Supported quadrature orders (odd orders 3 .. 15 that
#: scipy.integrate.lebedev_rule accepts) and their complete-rule point counts.
#: Order 15 (86 dirs) is the cap so every direction folder stays two-digit
#: (dir01 .. dir86). Kept in one place so gen_nac.py and RASCBEC_phonopy.py
#: stay in lockstep.
ORDER_TO_POINT_COUNT = {
    3: 6, 5: 14, 7: 26, 9: 38, 11: 50, 13: 74, 15: 86,
}

#: Inverse of ORDER_TO_POINT_COUNT (point count -> order). Every point count is
#: unique, so this is unambiguous.
POINT_COUNT_TO_ORDER = {v: k for k, v in ORDER_TO_POINT_COUNT.items()}

#: The full supported-order list, for error messages / help text.
SUPPORTED_ORDERS = tuple(sorted(ORDER_TO_POINT_COUNT))


def _scipy_lebedev(order):
    """Fetch the raw (points, weights) node set from scipy for *order*.

    Raises
    ------
    ImportError   -- scipy is not installed.
    NotImplementedError -- the order is not available via scipy.
    """
    try:
        from scipy.integrate import lebedev_rule
    except ImportError as exc:
        raise ImportError(
            'scipy is required for the NAC quadrature '
            '(scipy.integrate.lebedev_rule). Install scipy.') from exc
    return lebedev_rule(int(order))


def lebedev_points(order=DEFAULT_ORDER):
    """Return (points, weights) for a Lebedev-Laikov quadrature of *order*.

    Parameters
    ----------
    order : int
        Any order in ``ORDER_TO_POINT_COUNT`` (odd integers 3 .. 15). 7 (the
        classic 26-point rule) is the default.

    Returns
    -------
    points   : ndarray, shape (3, m) -- unit direction vectors, all octants signed
    weights  : ndarray, shape (m,)   -- normalized to sum to 1 (spherical average)
    """
    if order not in ORDER_TO_POINT_COUNT:
        raise ValueError(
            f'Lebedev order {order!r} is not supported; supported orders are '
            f'{", ".join(map(str, SUPPORTED_ORDERS))} (point counts '
            f'{", ".join(str(ORDER_TO_POINT_COUNT[o]) for o in SUPPORTED_ORDERS)}).')
    x, w = _scipy_lebedev(order)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    x = x / np.linalg.norm(x, axis=0, keepdims=True)
    w = w / w.sum()
    return x, w


def lebedev_subset(lo, hi):
    """True if every point of the order-*lo* rule lies in the order-*hi* rule.

    Lower-order Lebedev-Laikov rules are generally geometric subsets of higher
    ones, but NOT contiguously: e.g. order 7 (26 pts) IS a subset of orders 11
    and 13, but NOT of order 9 or 15, while orders 3 and 5 are subsets of
    almost all higher orders. Used to validate a lower-order selection whose
    order is below the data's own order.
    """
    lx, _ = lebedev_points(lo)
    hx, _ = lebedev_points(hi)
    return all(any(np.allclose(lx[:, i], hx[:, j], atol=1e-7) for j in range(hx.shape[1]))
               for i in range(lx.shape[1]))


# ===========================================================================
# NAC project scaffolding (generator)
# ===========================================================================
def digest(path):
    """Return the SHA-256 hex digest of *path*, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def resolve_project():
    """Validate the working directory and return the project layout.

    The script must be run from ``Project/04_phon/<composition>``. This
    resolves the four path elements the rest of the workflow needs.

    Returns
    -------
    source       : Path -- the ``04_phon/<composition>`` directory (cwd).
    project      : Path -- the ``Project`` root (``04_phon``'s parent).
    composition  : str  -- the composition folder name.
    target       : Path -- ``Project/05_rascbec/<composition>`` (output root).
    """
    source = Path.cwd().resolve()
    if source.parent.name != '04_phon':
        raise SystemExit('Run from Project/04_phon/<composition>.')
    project = source.parent.parent
    composition = source.name
    target = project / '05_rascbec' / composition
    return source, project, composition, target


def read_structure(poscar):
    """Read a VASP POSCAR and return its structural metadata.

    Returns
    -------
    lattice         : ndarray (3, 3) -- lattice vectors (rows).
    nat             : int           -- number of atoms.
    symbols         : list[str]     -- species per atom, in POSCAR order.
    scaled_positions: ndarray (nat, 3) -- fractional coordinates.
    """
    from phonopy.interface.vasp import read_vasp
    cell = read_vasp(str(poscar))
    lattice = np.asarray(cell.cell, dtype=float)
    return (lattice, len(cell), list(cell.symbols),
            np.asarray(cell.scaled_positions, dtype=float))


def verify_target_consistency(target, lattice, nat, symbols, scaled):
    """If the target already holds a POSCAR, confirm it matches the new one.

    Guards against silently regenerating a project whose on-disk structure
    (cell, species order, or positions) differs from the input. Aborts with a
    clear message so the operator can reconcile first.
    """
    existing = target / 'POSCAR'
    if not existing.is_file():
        return
    from phonopy.interface.vasp import read_vasp
    other = read_vasp(str(existing))
    same_cell = np.allclose(np.asarray(other.cell), lattice, atol=1e-7, rtol=0)
    same_symbols = list(other.symbols) == list(symbols)
    delta = (np.asarray(other.scaled_positions) - scaled + 0.5) % 1 - 0.5
    same_pos = delta.shape == scaled.shape and np.allclose(delta, 0, atol=1e-7)
    if not (same_cell and same_symbols and same_pos):
        raise SystemExit(
            '05_rascbec POSCAR differs in cell, species/order or positions; '
            'reconcile first before regenerating.')


def build_directions(order, lattice):
    """Return the list of NAC direction rows for *order* and *lattice*.

    Rows are ordered deterministically (by shell then coordinate) so that a
    re-run produces an identical manifest. Each row carries the data
    ``RASCBEC_phonopy.py`` reads back later.

    Returns
    -------
    rows : list[dict] with keys:
        directory   -- folder name, e.g. ``dir07``.
        yaml        -- path (relative to target) of the phonopy output YAML.
        cartesian   -- unit direction in the *lattice* (fractional) basis.
        q_direction -- the ``Q_DIRECTION`` vector to write to direction.conf
                       (lattice @ cartesian, rescaled so its largest
                       component has unit magnitude -- a direction, not a norm).
        weight      -- quadrature weight (sums to 1 across all rows).
    """
    points, weights = lebedev_points(order)   # points (3, m), weights (m,)
    inv_lattice_t = np.linalg.inv(lattice).T

    raw = []
    for i in range(points.shape[1]):
        cart = points[:, i]                      # unit vector in lattice basis
        q = lattice @ cart                       # Cartesian (crystal) direction
        q = q / max(abs(q))                      # rescale: phonopy wants a direction
        # Round-trip sanity check: q -> lattice basis must recover cart.
        back = q @ inv_lattice_t
        back = back / np.linalg.norm(back)
        if not np.allclose(back, cart, atol=1e-12):
            raise RuntimeError('Direction round-trip check failed (lattice inverse).')
        raw.append({
            'cartesian': cart,
            'q_direction': q,
            'weight': float(weights[i]),
            'nnz': int((np.abs(cart) > 1e-12).sum()),
        })

    # Deterministic ordering: fewer non-zero components first (axes, then
    # face-diagonal, then body-diagonal shells), then lexicographic on the
    # coordinate -- stable across runs and independent of scipy's ordering.
    raw.sort(key=lambda r: (r['nnz'],
                            tuple(round(float(c), 12) for c in r['cartesian'])))

    n_expected = ORDER_TO_POINT_COUNT[order]
    if len(raw) != n_expected:
        raise RuntimeError(
            f'Expected {n_expected} directions for order {order}, got {len(raw)}.')
    if not np.isclose(sum(r['weight'] for r in raw), 1.0):
        raise RuntimeError('Quadrature weights do not sum to 1.')

    out = []
    for idx, r in enumerate(raw, 1):
        name = f'dir{idx:02d}'
        out.append({
            'directory': name,
            'yaml': f'{name}/qpoints.yaml',
            'cartesian': r['cartesian'].tolist(),
            'q_direction': r['q_direction'].tolist(),
            'weight': r['weight'],
        })
    return out


def build_manifest(order, composition, nat, lattice, symbols, scaled,
                   inputs, rows):
    """Assemble the ``nac_directions.json`` manifest (single source of truth).

    The consumer (``RASCBEC_phonopy.py``) validates the manifest's structure,
    directions, weights and per-row ``q_direction``/``cartesian`` fields
    against the same shared quadrature rule, so the two must stay in lockstep.
    """
    return {
        'schema': MANIFEST_SCHEMA,
        'rule': f'Lebedev-Laikov order {order}, '
                f'{ORDER_TO_POINT_COUNT[order]} points, weights sum to 1',
        'composition': composition,
        'nat': nat,
        'lattice': lattice.tolist(),
        'symbols': list(symbols),
        'scaled_positions': scaled.tolist(),
        'input_sha256': {k: digest(v) for k, v in inputs.items()},
        'source_paths': {k: str(v) for k, v in inputs.items()},
        'directions': rows,
    }


def write_manifest(mpath, manifest):
    """Write the ``nac_directions.json`` manifest.

    A re-run simply overwrites any pre-existing manifest with the freshly
    built one, so the on-disk manifest always reflects the current inputs and
    settings.
    """
    mpath.write_text(json.dumps(manifest, indent=2) + '\n')


def prepare_direction(work, inputs, row):
    """Create one direction folder: input symlinks + ``direction.conf``.

    Symlinks are idempotent and refuse to clobber an existing non-symlink, so
    a re-run never destroys the real input files it links to.
    """
    work = Path(work)
    work.mkdir(exist_ok=True)
    for name, path in inputs.items():
        link = work / name
        if link.is_symlink() and link.resolve() == path.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f'Refusing to overwrite {link}')
        link.symlink_to(path)
    q = ' '.join(f'{x:.17g}' for x in row['q_direction'])
    conf = ('DIM = 1 1 1\n'
            'PRIMITIVE_AXES = 1 0 0 0 1 0 0 0 1\n'
            'FORCE_CONSTANTS = READ\n'
            'QPOINTS = 0 0 0\n'
            'EIGENVECTORS = .TRUE.\n'
            'NAC = .TRUE.\n'
            f'Q_DIRECTION = {q}\n')
    (work / 'direction.conf').write_text(conf)


def _is_direction_folder(name):
    """True if *name* looks like a direction folder (``dir`` + digits), e.g. ``dir07``."""
    return name.startswith('dir') and len(name) > 3 and name[3:].isdigit()


def clean_stale_direction_dirs(target, keep):
    """Remove leftover ``dirNN`` folders whose name is not in *keep*.

    A re-run must leave exactly the current direction set behind (not merely
    overwrite the shared ones). This removes stale ``dirNN`` directories from a
    previous run with a different ``--lebedev-order`` -- e.g. ``dir15`` ...
    ``dir26`` left over after switching from order 7 (26) to order 5 (14) -- so
    the target always reflects the requested order.

    Only directories named ``dir<digits>`` that are not in *keep* are removed.
    ``shutil.rmtree`` unlinks the symlinks inside a folder without following
    them, so the shared POSCAR/FORCE_CONSTANTS/BORN targets are never touched,
    and any non-``dirNN`` entry is left alone. Returns the number removed.
    """
    target = Path(target)
    removed = 0
    if not target.is_dir():
        return 0
    for entry in sorted(target.iterdir()):
        if not _is_direction_folder(entry.name) or entry.name in keep:
            continue
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink()
        else:
            shutil.rmtree(entry)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Per-direction phonopy run (worker)
# ---------------------------------------------------------------------------
def _run_direction(payload):
    """Run phonopy for one direction and validate the result.

    Runs as a module-level function so it is picklable for a
    ``ProcessPoolExecutor`` (required for true CPU parallelism of the
    ``phonopy`` subprocess). Each call is independent -- it touches only its
    own ``work/`` directory -- so no cross-process coordination is needed.

    Parameters
    ----------
    payload : tuple
        (work: str, exe: str, nat: int, threads: int)
        ``threads`` caps the BLAS/OpenMP threads each phonopy process may use,
        so that ``workers * threads ~= core count`` (avoids CPU
        oversubscription, see :func:`run_directions`).

    Returns
    -------
    dict -- {'directory': str, 'ok': bool, 'error': str or None}
    """
    work, exe, nat, threads = payload
    work = Path(work)
    directory = work.name
    result = work / 'qpoints.yaml'
    if result.exists():
        result.unlink()

    # Cap this phonopy process's BLAS/OpenMP thread pool so N workers * T
    # threads stays within the machine's core count (each phonopy run would
    # otherwise spawn a thread per core and oversubscribe the box). threads=0
    # means "uncapped" -- leave the variables alone so phonopy uses its own
    # default (note: most BLAS treat "0" as 1 thread, so we must not set it).
    env = os.environ.copy()
    if threads:
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            env[var] = str(threads)

    log = work / 'phonopy.log'
    try:
        with log.open('w') as fh:
            proc = subprocess.run(
                [exe, '-c', 'POSCAR', 'direction.conf'],
                cwd=work, stdout=fh, stderr=subprocess.STDOUT, env=env)
    except Exception as exc:  # noqa: BLE001 -- report any spawn failure per-dir
        return {'directory': directory, 'ok': False, 'error': f'spawn failed: {exc}'}

    if proc.returncode != 0 or not result.is_file():
        return {'directory': directory, 'ok': False,
                'error': f'phonopy failed (returncode={proc.returncode}); see {log}'}

    try:
        data = yaml.safe_load(result.read_text())
    except Exception as exc:  # noqa: BLE001
        return {'directory': directory, 'ok': False,
                'error': f'failed to parse {result}: {exc}'}

    phonon = data.get('phonon', [])
    if len(phonon) != 1:
        return {'directory': directory, 'ok': False,
                'error': f'expected one Gamma point in {result}'}
    bands = phonon[0].get('band', [])
    if len(bands) != 3 * nat or any('eigenvector' not in b for b in bands):
        return {'directory': directory, 'ok': False,
                'error': f'missing modes/eigenvectors in {result}'}

    return {'directory': directory, 'ok': True, 'error': None}


def resolve_worker_count(requested, default):
    """Resolve a ``--jobs`` value to a positive worker count (1 = sequential)."""
    if requested is None:
        return max(1, default)
    if requested < 1:
        raise SystemExit('--jobs must be >= 1 (use --jobs 1 to force sequential).')
    return requested


def resolve_thread_count(requested, workers, cores):
    """Resolve a ``--threads`` value to a per-process BLAS thread cap.

    The cap keeps ``workers * threads`` within the machine's core count so the
    parallel phonopy processes do not oversubscribe the box. With a single
    worker the cap is the full core count (a lone process should use all of
    it). A value of ``0`` is allowed to mean "uncapped" (let phonopy decide).
    """
    if requested is None:
        return cores if workers == 1 else max(1, cores // workers)
    if requested < 0:
        raise SystemExit(
            '--threads must be >= 0 (0 = uncapped, 1 = single-threaded '
            'phonopy).')
    return requested


def _fmt3(vec):
    """Format a 3-vector compactly for log lines, e.g. ``[ 0.000  1.000  0.000]``."""
    return '[' + ' '.join(f'{x:8.5f}' for x in vec) + ']'


def _direction_label(directory, row):
    """One log line per direction: its folder, cartesian unit vector, the
    phonopy ``Q_DIRECTION``, and the quadrature weight.

    ``row`` is a manifest direction (keys ``cartesian``, ``q_direction``,
    ``weight``). If it is missing (shouldn't happen) a minimal line is used.
    """
    if not row:
        return f'{directory}'
    return (f'{directory}  cart={_fmt3(row["cartesian"])} '
            f'q_dir={_fmt3(row["q_direction"])}  w={row["weight"]:.6f}')


class _Progress:
    """Live single-line progress bar (on a TTY) or plain per-item lines (piped).

    Each :meth:`tick` advances the count, logs the current direction, and
    redraws the bar on the last line so the bar always sits at the bottom and
    the per-direction lines accumulate above it (like ``rsync``/``pv``). When
    stdout is not a terminal the bar is suppressed and only the lines are
    emitted, so redirected logs stay clean.
    """

    def __init__(self, total, width=30):
        self.total = total
        self.width = width
        self.done = 0
        self.t0 = time.monotonic()
        self.use_bar = sys.stdout.isatty()
        self._last = 0  # char width of the bar line currently on screen

    def _bar_line(self):
        frac = self.done / self.total if self.total else 1.0
        filled = int(round(self.width * frac))
        bar = '█' * filled + '░' * (self.width - filled)
        dt = time.monotonic() - self.t0
        rate = self.done / dt if dt > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        return (f'  {self.done}/{self.total} [{bar}] {100 * frac:5.1f}%  '
                f'{dt:6.1f}s  eta {eta:5.1f}s')

    def _clear(self):
        if self._last:
            sys.stdout.write('\r' + ' ' * (self._last + 2) + '\r')

    def tick(self, label):
        self.done += 1
        if self.use_bar:
            # Clear the previous bar line, log this direction, redraw the bar.
            self._clear()
            sys.stdout.write(label + '\n')
            bar = self._bar_line()
            self._last = len(bar)
            sys.stdout.write('\r' + bar)
            sys.stdout.flush()
        else:
            sys.stdout.write(label + '\n')
            sys.stdout.flush()

    def finish(self):
        if self.use_bar:
            self._clear()
            self._last = 0
            sys.stdout.flush()


def run_directions(payloads, jobs, threads=None, info=None):
    """Run the per-direction phonopy jobs, in parallel when beneficial.

    Prints the parallelism plan (worker count, per-process thread cap, total
    cores) so a run that is unexpectedly slow can be diagnosed at a glance,
    then streams each direction as it completes (folder, cartesian, phonopy
    ``Q_DIRECTION``, weight) behind a live progress bar. Returns a list of
    status dicts (one per payload, order preserved).

    Parameters
    ----------
    payloads : list[tuple]  -- (work, exe, nat) per direction to run.
    jobs     : int or None  -- ``--jobs`` worker count (None = auto).
    threads  : int or None  -- ``--threads`` per-process BLAS cap (None = auto).
    info     : dict         -- directory -> manifest direction row, for logging.
    """
    if not payloads:
        return []
    info = info or {}
    cores = max(1, mp.cpu_count())
    workers = min(len(payloads), resolve_worker_count(jobs, cores))
    tcap = resolve_thread_count(threads, workers, cores)
    mode = f'{workers} worker(s)' if workers > 1 else 'sequential'
    tshow = 'uncapped' if tcap == 0 else f'{tcap} thread(s)/worker'
    print(f'  parallelism: {mode}, {tshow}, {cores} core(s) '
          f'(workers x threads = {workers} x {tcap or cores})', flush=True)
    payloads = [(w, e, n, tcap) for (w, e, n) in payloads]
    progress = _Progress(len(payloads))

    def _label(res):
        return _direction_label(res['directory'], info.get(res['directory']))

    if workers <= 1 or len(payloads) <= 1:
        results = []
        for p in payloads:
            results.append(_run_direction(p))
            progress.tick(_label(results[-1]))
        progress.finish()
        return results

    try:
        # 'fork' reuses the already-imported module in workers (no re-import of
        # numpy/phonopy); safe here because no threads are running at fork time.
        ctx = mp.get_context('fork')
    except ValueError:  # platforms without fork
        ctx = None
    results: list[dict | None] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        fut_to_idx = {pool.submit(_run_direction, p): i
                      for i, p in enumerate(payloads)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            directory = Path(payloads[i][0]).name
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- keep the run going
                res = {'directory': directory, 'ok': False,
                       'error': f'worker raised: {exc!r}'}
            results[i] = res
            progress.tick(_label(res))
    progress.finish()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _valid_order(order):
    """Validate an order and return it; raise a helpful error if unsupported."""
    if order not in ORDER_TO_POINT_COUNT:
        raise SystemExit(
            f'--lebedev-order {order!r} is not supported; supported orders are '
            f'{", ".join(map(str, SUPPORTED_ORDERS))} '
            f'({ORDER_TO_POINT_COUNT[min(SUPPORTED_ORDERS)]}..'
            f'{ORDER_TO_POINT_COUNT[max(SUPPORTED_ORDERS)]} directions).')
    return order


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--poscar', default='POSCAR',
                   help='POSCAR to use as the structure (default: POSCAR).')
    p.add_argument('--phonopy', default='phonopy',
                   help='Phonopy executable to run per direction (default: phonopy).')
    p.add_argument('--lebedev-order', dest='lebedev_order', type=int,
                   default=DEFAULT_ORDER,
                   help=f'Lebedev-Laikov quadrature order (default: {DEFAULT_ORDER}). '
                        f'Supported orders (odd 3..15) with point counts: '
                        f'3->6, 5->14, 7->26, 9->38, 11->50, 13->74, 15->86. '
                        f'Order 15 is the cap (keeps dirNN two-digit).')
    p.add_argument('--jobs', type=int, default=None,
                   help='Number of parallel workers for the per-direction phonopy '
                        'runs. Default: auto (min(#directions, CPU count)); use '
                        '--jobs 1 to force sequential.')
    p.add_argument('--threads', type=int, default=None,
                   help='Cap on BLAS/OpenMP threads per phonopy process. Default: '
                        'auto = CPU count / workers (so workers x threads ~= cores); '
                        'use --threads 1 to force single-threaded phonopy, or '
                        '--threads 0 to leave it uncapped (phonopy default).')
    return p.parse_args(argv)


def main(argv=None):
    """Entry point: scaffold (and optionally run) the NAC direction projects."""
    args = parse_args(argv)
    order = _valid_order(args.lebedev_order)

    source, project, composition, target = resolve_project()
    inputs = {
        'POSCAR': Path(args.poscar).resolve(),
        'FORCE_CONSTANTS': source / 'FORCE_CONSTANTS',
        'BORN': project / '03_bec' / composition / 'BORN',
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f'Missing {name}: {path}')

    exe = shutil.which(args.phonopy)
    if not exe:
        raise SystemExit(f'Phonopy executable "{args.phonopy}" not found on PATH.')

    lattice, nat, symbols, scaled = read_structure(inputs['POSCAR'])
    if abs(np.linalg.det(lattice)) < 1e-10:
        raise SystemExit('Singular lattice.')

    target.mkdir(parents=True, exist_ok=True)
    verify_target_consistency(target, lattice, nat, symbols, scaled)

    rows = build_directions(order, lattice)
    manifest = build_manifest(order, composition, nat, lattice, symbols, scaled,
                              inputs, rows)
    mpath = target / 'nac_directions.json'
    keep = {r['directory'] for r in manifest['directions']}

    # 0) Rewrite: drop stale dirNN folders left over from a previous run with a
    #    different order so the target holds exactly the current directions.
    removed = clean_stale_direction_dirs(target, keep)
    if removed:
        print(f'Removed {removed} stale direction folder(s) '
              f'(now {len(keep)} directions, Lebedev order {order}).', flush=True)

    write_manifest(mpath, manifest)

    # 1) Prepare every direction folder (symlinks + direction.conf).
    for row in manifest['directions']:
        prepare_direction(target / row['directory'], inputs, row)

    # 2) Run the phonopy job for every direction, from scratch.
    #    Re-runs are idempotent: each direction regenerates its qpoints.yaml,
    #    so the outputs always reflect the current inputs and settings.
    print(f'Running {len(manifest["directions"])} direction(s) '
          f'(Lebedev order {order}) ...', flush=True)
    info = {r['directory']: r for r in manifest['directions']}
    payloads = [(str(target / r['directory']), exe, nat)
                for r in manifest['directions']]
    t0 = time.monotonic()
    results = run_directions(payloads, args.jobs, threads=args.threads, info=info)
    dt = time.monotonic() - t0

    # 3) Report.
    #    A worker result can be None if the future never produced a dict;
    #    treat those as failures so we never index into None.
    failed = [r for r in results if r is None or not r['ok']]
    for r in failed:
        label = r['directory'] if r else 'unknown'
        err = r['error'] if r else 'no result returned by worker'
        print(f'FAILED {label}: {err}', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} direction(s) failed; see messages above.')

    print(f'Manifest: {mpath}')
    print(f'All {len(manifest["directions"])} directional eigensystems completed '
          f'(Lebedev order {order}).')
    t = (f'{dt / 60:.1f} min' if dt >= 90 else f'{dt:.1f} s')
    print(f'Total: {len(results)} direction(s) in {t}.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
