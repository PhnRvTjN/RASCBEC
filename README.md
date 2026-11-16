# RASCBEC - Raman Spectroscopy Calculation via Born Effective Charge

> **This is a community fork** of the original
> [RASCBEC](https://github.com/rruuiizz/RASCBEC) by Rui Zhang et al.
> The underlying physics and RASCBEC method are unchanged. This fork
> refactors the scripts for practical use on large doped supercells, adds a
> complete plotting pipeline, and introduces multi-composition comparison
> tools (single-spectrum, waterfall, and two-dopant composition matrix).

---

## Improvements Over the Original

| Feature | Original | This Fork |
| --- | --- | --- |
| Atomic masses | Hardcoded arrays | Auto-lookup via **pymatgen** |
| OUTCAR layout | Flat files (`OUTCAR1`, `OUTCARm1`, …) | Subdirectory layout (`./1/OUTCAR`, `./m1/OUTCAR`, …) |
| OUTCAR BEC reader | Forward scan (reads entire file) | Reverse scan — finds last BEC block directly, avoids full re-read of multi-GB OUTCARs |
| POSCAR rotation | `rotate.py` outputs 3 flat files (`ex.POSCAR.vasp`, …) | `rotate.py` (improved) creates all 8 subdirectories with correct POSCAR in each |
| Phonopy input | `.dat` intermediate files via `qpoints_to_eigfreq.py` / `mesh_to_eigfreq.py` | Direct YAML reading — `RASCBEC_phonopy.py` reads `qpoints.yaml` (or `mesh.yaml`) and `irreps.yaml` with no intermediate step |
| Mode filtering | Fixed `3N`; frequency-threshold cutoff | Two independent filters: `freq < 0.1` (imaginary modes) and `ir_label is None` (acoustic/unassigned); a mode with a negative frequency but a valid irrep label is still excluded |
| Mode symmetry labels | Not included | `irreps.yaml` integrated; irrep label (A1, B2, E, …) stored per mode in CSV and displayed on plot peak annotations |
| E-field input | Manual `--E` flag required | Auto-read from `./1/OUTCAR` (EFIELD_PEAD); `--E` overrides |
| Output filename | `raman_phonopy.dat` / `raman_vasp.dat` | Chemistry-based: `<formula>[_<dopantpart>]_E<E>.csv`, e.g. `Na3PS4_Ca01-Cl06_E0.02.csv` |
| CSV metadata | None | Header lines: `# Formula`, `# Dopants`, `# File_Label`, `# E_field` (+ `# Phonon_treatment` etc. for NAC outputs) |
| Chemistry roles | Implicit least-abundant-species reference | `--bas` / `--dops` overrides, including `+`-joined basis groups (`--bas Ti+Ga`); **base-formula reconstruction** via `DOPANT_SUBSTITUTIONS` (doped counts are converted back to the undoped host formula) |
| Activity calculation | Nested Python loops O(N·27) | Vectorised `np.einsum('tikj,sti->sjk')` — BLAS-backed, seconds instead of hours for large supercells |
| Stokes intensity | Not included | Physical Stokes scattered intensity by default: `I(ν,T) ∝ (ν_L−ν)⁴/ν·[n(ν,T)+1]·A_s`, with `--temp` and `--laser-wl` controls (defaults 300 K, 632.8 nm); `--raw-act` restores raw S(ν) |
| Plot titles | None | Auto-generated two-line title: bold mathtext formula (e.g. `Na₃₋₂ₓ₋ᵧCaₓPS₄₋ᵧClᵧ` + `x=…, y=…` legend) / run parameters (E, FWHM, T) |
| Plot generation | Not included | Integrated via `plot_raman.py` (Lorentzian broadening, prominence-based peak labels with irreps, two-line auto title) |
| Comparison plots | Not included | `compare_raman.py` — waterfall / overlaid, absolute or normalised, shared physical model; `compare_raman_matrix.py` — two-dopant composition matrix (`x` × `y` grid of shared-scale panels) |
| NAC multi-direction | Not included | `RASCBEC_phonopy.py` auto-detects NAC setups (`nac_directions.json` or 26 `dirNN/direction.conf` folders), validates the full 26-point Lebedev rule (directions, duplicate detection, weights `1/21, 4/105, 9/280`), and emits Lebedev-weighted directional sticks with a JSON provenance audit file |
| CLI arguments | None | `--E`, `--gamma` (THz), `--temp`, `--laser-wl`, `--freq-min/max`, `--phonon-yaml`, `--irreps`, `--bas`, `--dops`, `--no-plot`, `--out-csv/png`, … |
| Code structure | Single monolithic script | Modular functions (BEC reader, derivative builder, activity calculator, writer) shared across both phonon variants; common plotting primitives in `plot_raman.py` |

---

## `rotate.py` — Fixes vs. Original

The following two bugs were fixed relative to the upstream `rotate.py`
(authored by Rui Zhang, 2024-11-06):

### Fix 1 — Incorrect rotation algebra (shearing bug)

**Original:**

```python
lattice_rotated_x = rotation_x @ lattice_vectors
```

**Problem:** VASP POSCARs store lattice vectors as **row vectors** (one
vector per row). The original code applied the rotation matrix on the
**left**, which is the column-vector convention. For non-cubic cells
(e.g., LLTO, Na₃PS₄ doped supercells) this silently shears and distorts
the written cell, causing VASP to receive a non-rigid deformation of the
lattice rather than a pure rotation. This can cause VASP to misclassify
the electronic structure (e.g., fail to identify the system as insulating).

**Fix:** For POSCAR row-vector storage, a rigid Cartesian rotation must
be applied as:

```python
lattice_rotated = lattice_vectors @ R.T
```

This leaves fractional coordinates invariant and correctly rotates every
lattice vector in Cartesian space without shearing.

---

### Fix 2 — INCAR, KPOINTS, POTCAR, and *.sbatch auto-copy

**Original:** Only writes POSCARs; user must manually copy all other
VASP input files into each subdirectory.

**Fix:** The corrected `rotate.py` automatically copies `INCAR`,
`KPOINTS`, `POTCAR`, and any `*.sbatch` job-submission scripts (auto-
discovered from the working directory or supplied explicitly) into all 8
subdirectories. Flags available:

| Flag | Behavior |
| --- | --- |
| `--incar`, `--kpoints`, `--potcar` | Override default paths |
| `--sbatch FILE [FILE ...]` | Explicit sbatch list |
| `--no-sbatch` | Suppress sbatch copying |

---

### Diagnostic output

The corrected script prints the detected row-to-axis mapping and the
resolved legacy routing on every run so you can verify the folder
assignment before submitting VASP jobs:

```bash
Detected lattice-row orientation relative to Cartesian axes:
  row 1 -> +y (alignment = 1.000000)
  row 2 -> +z (alignment = 1.000000)
  row 3 -> +x (alignment = 1.000000)
Legacy folder routing derived from detected row order:
  legacy x -> row 1 (+y) -> physical Cartesian y
  legacy y -> row 2 (+z) -> physical Cartesian z
  legacy z -> row 3 (+x) -> physical Cartesian x
```

Use `--strict-axis-report` to abort if any row-to-axis alignment score
falls below the threshold set by `--axis-tol` (default: 0.90). This is
recommended for oblique cells where the row-to-axis assignment is ambiguous.

---

## Chemistry Metadata, Dopant Roles & Base-Formula Reconstruction

Both RASCBEC scripts derive the chemistry metadata (formula, dopant
fractions, file label, plot titles) **from the POSCAR composition alone**,
with a shared, validated set of role-resolution rules:

- **Automatic inference** — species below 5 % of the supercell atoms
  (`DOPANT_THRESHOLD`) are classified as dopants. Bypass with `--dops`
  where the heuristic is ambiguous (e.g. some LLTO compositions).
- **`--bas` (basis / concentration reference)** — defines both the
  host-formula normalisation and the denominator of the dopant x-values
  (`x = n_dop / n_bas`). A `+`-joined **group** is allowed (e.g.
  `--bas Ti+Ga`): the group's member counts are summed into a single
  reference, so `Ti + Ga = 1`. Example: `Li60La90Ti150O360` with
  `--bas Ti+Ga` → `Li0.38La0.56Ti0.94O2.25`, dopant x-values reported
  per 160 Ti+Ga sites. When omitted, the least-abundant host species is
  used as the legacy reference.
- **Base (undoped) formula** — the `# Formula` value is the **base** host
  composition, not the raw doped host-only count. The supercell contains
  only the host atoms that were *not* displaced, so the base is recovered
  by adding back the atoms each dopant took (or added) at every host
  element using the `DOPANT_SUBSTITUTIONS` table (e.g. `Ca → 2 Na`,
  `Cl → Na + S`, `Ga → Ti − Li`, `Y → La`), then normalising to the
  basis. A Ca-doped Na₃PS₄ supercell whose doped host counts are
  Na20P8S30 (Ca=1, Cl=2) is therefore reported as **Na3PS4**, not
  Na2.5P1S3.75.
- **Validation** — every explicit `--bas` / `--dops` value is checked
  (species present, no duplicates, no empty members) before any
  computation or file naming happens.

> Extend `DOPANT_SUBSTITUTIONS` in `plot_raman.py` with your own
> dopant→host substitution rules (positive = dopant displaces that many
> host atoms; negative = dopant adds to the host count) and the formula
> reconstruction and symbolic plot titles pick it up automatically.

---

## Automatic NAC Multi-Direction Mode (phonopy variant)

`RASCBEC_phonopy.py` supports a **fixed-field NAC (non-analytic
correction) workflow** with automatic spherical averaging. Run it with no
arguments and it looks for one of:

- `nac_directions.json` — a manifest of the 26 NAC directions, or
- 26 sibling folders named `dir01` … `dir26`, each containing a
  `direction.conf` (with `NAC = .TRUE.` and `Q_DIRECTION`) and a
  `qpoints.yaml`.

If neither is found, the script transparently falls back to the original
single-`qpoints.yaml` mode (`--input-mode auto` is the default).

The multi-mode path enforces, before any output is written:

- the **complete 26-point Lebedev rule** — all 26 physical directions
  (1-axis, 2-axis and 3-axis) present exactly once; duplicates, missing
  directions, or non-Lebedev directions abort with no silent fallback;
- the **degree-7 weights** `1/21`, `4/105`, `9/280` (matched by the
  number of non-zero Cartesian components) and a unit sum;
- manifest / `direction.conf` / Cartesian **direction agreement**, plus
  POSCAR ↔ manifest lattice/symbol/position consistency.

Per direction, Gamma-point eigenvectors are validated (unit norm,
real-valued), rigid translations are rejected by mass-weighted overlap
(`--translation-overlap`, default 0.95), and frequencies below
`--nac-min-thz` (default 0.1 THz) are dropped — both recorded in the
provenance file. Activities are computed once with the shared RASCBEC
routine and **weighted by the Lebedev weights before CSV output**; the
CSV rows are directional sticks, not unique modes (declared in the header
via `# Phonon_treatment: NAC_Lebedev26`).

Outputs (suffix `_NAC`):

```bash
<chem>_E<E>_NAC.csv              # weighted directional sticks
<chem>_E<E>_NAC.directions.json  # full provenance: directions, weights,
                                 # retained/skipped modes, thresholds
<chem>_E<E>_NAC.png
```

Existing outputs are never overwritten without `--overwrite`.

> The NAC mode is a phonopy-YAML feature; `RASCBEC_VASP.py` is unaffected.
> No electro-optic correction beyond the fixed-field RASCBEC response is
> applied; downstream Stokes conversion and broadening are unchanged.

---

## Dependencies

```bash
python >= 3.9
numpy
scipy
matplotlib
pymatgen
pyyaml
```

Install all at once:

```bash
pip install numpy scipy matplotlib pymatgen pyyaml
```

---

## Repository Structure

```bash
RASCBEC/
├── rotate.py                - Rotate POSCAR; create all 8 subdirectories for BEC runs
├── RASCBEC_phonopy.py       - Raman activities from phonopy YAML (qpoints.yaml + irreps.yaml);
│                              includes automatic NAC 26-direction Lebedev-averaged mode
├── RASCBEC_VASP.py          - Raman activities using VASP DFPT eigenvectors (same maturity as phonopy variant)
├── plot_raman.py            - Shared plotting library: single-composition spectrum, Stokes conversion,
│                              peak labels, two-line symbolic titles, CSV reader, DOPANT_SUBSTITUTIONS
├── compare_raman.py         - Multi-composition / multi-E-field comparison (waterfall or overlaid)
├── compare_raman_matrix.py  - Two-dopant composition matrix (x × y grid of shared-scale panels)
├── qpoints_to_eigfreq.py    - Legacy: convert qpoints.yaml -> .dat (no longer needed)
├── mesh_to_eigfreq.py       - Legacy: convert mesh.yaml -> .dat (no longer needed)
├── RASCBEC_vasp.py          - Original upstream script (preserved for reference)
├── Code/                    - Original upstream scripts (e.g. rotate.py)
└── Example/                 - GeO2 rutile example inputs and reference outputs
```

> `RASCBEC_phonopy.py` and `RASCBEC_VASP.py` (uppercase) are the improved
> scripts in this fork. `RASCBEC_vasp.py` (lowercase) is the unmodified
> original kept for reference.
>
> `qpoints_to_eigfreq.py` and `mesh_to_eigfreq.py` are kept for reference
> but are no longer part of the workflow. `RASCBEC_phonopy.py` now reads
> `qpoints.yaml` and `irreps.yaml` directly.
>
> `compare_raman.py` and `compare_raman_matrix.py` are both built on the
> shared primitives in `plot_raman.py` (Stokes conversion, Lorentzian
> broadening, peak selection, title builder, CSV reader), so all tools
> agree on physics and formatting.

---

## Workflow

### 1 - Rotate POSCAR

```bash
python rotate.py
```

Creates all 8 subdirectories with the correct POSCAR in each:

```bash
./1/POSCAR   ./m1/POSCAR    — unrotated reference (copied directly)
./x/POSCAR   ./mx/POSCAR   — 45° rotation in xy-plane
./y/POSCAR   ./my/POSCAR   — 45° rotation in yz-plane
./z/POSCAR   ./mz/POSCAR   — 45° rotation in xz-plane
```

Place the appropriate INCAR, KPOINTS, and POTCAR in each subdirectory
(INCAR should set `EFIELD_PEAD`, `LCALCEPS = .TRUE.`; sign of the field
is controlled per subdirectory in the INCAR, not the POSCAR).

### 2 - Run 8 BEC Calculations

After VASP finishes, your working directory should look like:

```bash
9-RASCBEC/
├── POSCAR
├── qpoints.yaml           (phonopy path — generated in step 3)
├── irreps.yaml            (phonopy path — generated in step 3)
├── 1/OUTCAR              — E-field along + (unrotated)
├── m1/OUTCAR             — E-field along − (unrotated)
├── x/OUTCAR              — E-field along +x (rotated)
├── mx/OUTCAR             — E-field along −x
├── y/OUTCAR              — E-field along +y (rotated)
├── my/OUTCAR             — E-field along −y
├── z/OUTCAR              — E-field along +z (rotated)
└── mz/OUTCAR             — E-field along −z
```

#### ⚠️ EFIELD_PEAD sign convention for inactive components

RASCBEC builds ∂Z\*/∂E via central differences between + and − runs
(e.g., `./x` vs `./mx`). For these differences to correctly isolate the
intended field direction, the **inactive** EFIELD_PEAD components must be
**identical** in the + and − runs so they cancel in the subtraction.

**Correct** (inactive components same sign in both runs — cancel exactly):

```bash
./x  INCAR:  EFIELD_PEAD = +E   +δ   +δ
./mx INCAR:  EFIELD_PEAD = -E   +δ   +δ
```

**Incorrect** (inactive components sign-flipped — do NOT cancel, introduces ~δ/E error):

```bash
./x  INCAR:  EFIELD_PEAD = +E   +δ   +δ
./mx INCAR:  EFIELD_PEAD = -E   -δ   -δ
```

> **VASP reset behavior:** if you set an inactive component to `0.0`, VASP
> internally resets it to a small positive constant (commonly `+0.01`) in
> **both** the + and − runs. This is safe — the reset value is the same
> sign in both runs and cancels in the central difference.
>
> If you manually use a small nonzero value (e.g., `δ = 1e-4`) to avoid
> this reset, use `+δ` in **both** the + and − runs for the inactive
> components. Using `−δ` in the negative-field run introduces a
> contamination term of order δ/E in the off-diagonal BEC derivatives
> (~1.4% for δ=1e-4 and E≈0.005 eV/Å). It does not cause catastrophic
> error but is avoidable.

### 3 - Generate Phonopy YAML Files  *(phonopy path only)*

`RASCBEC_phonopy.py` reads the phonopy YAML files directly — no
intermediate conversion step is needed. Generate `qpoints.yaml` and
`irreps.yaml` in your working directory:

```bash
# Generate qpoints.yaml (Gamma-point phonons + eigenvectors)
phonopy --qpoints="0 0 0" --eigenvectors

# Generate irreps.yaml (irreducible representations at Gamma)
phonopy --irreps="0 0 0"
```

Or from a mesh calculation (1×1×1 = Gamma only):

```bash
phonopy -m 1 1 1 --eigenvectors    # generates mesh.yaml
phonopy --irreps="0 0 0"           # generates irreps.yaml
```

Skip this step entirely if you are using `RASCBEC_VASP.py`.

### 4 - Compute Raman Activities

**Phonopy eigenvectors** (frequencies in THz, un-mass-normalised):

```bash
python RASCBEC_phonopy.py
python RASCBEC_phonopy.py --gamma 0.45 --freq-min 50 --freq-max 600   # FWHM in THz

# Use mesh.yaml instead of qpoints.yaml
python RASCBEC_phonopy.py --phonon-yaml mesh.yaml

# Custom irreps file location
python RASCBEC_phonopy.py --irreps path/to/irreps.yaml
```

**VASP DFPT eigenvectors** (frequencies in cm⁻¹, already mass-normalised):

```bash
python RASCBEC_VASP.py
python RASCBEC_VASP.py --gamma 0.25 --no-plot
```

The E-field magnitude is read automatically from `./1/OUTCAR`.
Override only if needed:

```bash
python RASCBEC_phonopy.py --E 0.02
```

> **Key distinction:** VASP eigenvectors are already mass-normalised by
> VASP. `RASCBEC_VASP.py` does *not* divide by √(mass), whereas
> `RASCBEC_phonopy.py` does. Using the wrong script with the wrong
> eigenvector type will give incorrect Raman intensities.

**Chemistry overrides** (both scripts):

```bash
# Explicit dopants + basis reference
python RASCBEC_phonopy.py --dops Ca Cl --bas P

# Doped LLTO: normalise so Ti + Ga = 1
python RASCBEC_phonopy.py --bas Ti+Ga --dops Y Ga
```

Output filenames are derived from the POSCAR chemistry:

```bash
Na3PS4_E0.02.csv                   # undoped
Na3PS4_Ca01-Cl06_E0.02.csv        # Ca+Cl co-doped
Li0.31La0.56TiO3_E0.02.csv        # LLTO with --bas Ti
```

### 5 - Plot a Single Spectrum

`plot_raman.py` is called automatically by both RASCBEC scripts but can
also be run standalone (it auto-resolves the single CSV in the current
directory):

```bash
python plot_raman.py                                  # uses the single CSV in cwd
python plot_raman.py --dat Na3PS4_Ca01-Cl06_E0.02.csv
python plot_raman.py --dat Na3PS4_E0.02.csv --gamma 0.3 --freq-min 0 --freq-max 600
python plot_raman.py --raw-act --temp 300 --laser-wl 632.8
python plot_raman.py --title line1                    # formula line only
```

By default the plot shows the physical **Stokes scattered intensity**
`I(ν,T) ∝ (ν_L−ν)⁴/ν·[n(ν,T)+1]·A_s` at 300 K, 632.8 nm — use
`--raw-act` for the unweighted Raman activity S(ν).

### 6 - Compare Multiple Compositions

**`compare_raman.py`** — waterfall or overlaid traces of arbitrary CSVs:

```bash
# Waterfall (auto-normalises each, 10 prominence-selected labels per spectrum)
python compare_raman.py raman_*.csv --offset 1.2

# Global normalisation waterfall (preserves relative intensities)
python compare_raman.py raman_*.csv --norm --offset 1.2

# Overlaid, absolute scale (no labels)
python compare_raman.py raman_*.csv

# Same composition, different E-field (E auto-appended to legend)
python compare_raman.py Na3PS4_E0.01.csv Na3PS4_E0.02.csv Na3PS4_E0.05.csv --offset 1.2

# Variable-temperature Stokes comparison
python compare_raman.py raman_*.csv --offset 1.2 --temp 600

# Legacy raw Raman activity S(nu)
python compare_raman.py raman_*.csv --offset 1.2 --raw-act

# Custom labels / output name
python compare_raman.py raman_*.csv --offset 1.2 --labels "Undoped" "Ca-doped" "Ca/Cl-doped" --out fig1.png
```

**`compare_raman_matrix.py`** — two-dopant composition matrix (new):
arranges the CSV files into a categorical `x_dopant` × `y_dopant` grid of
shared-scale panels (e.g. `x_Ca` vs `y_Cl`), mirroring the published
two-dopant composition-matrix figures:

```bash
python compare_raman_matrix.py --dir compare --norm global        # full-spectrum matrix, global normalisation
python compare_raman_matrix.py --freq-min 245 --freq-max 305 --dir compare  # zoom into the P-S stretch region
python compare_raman_matrix.py *.csv --reference pristine.csv     # pristine overlay on every panel
python compare_raman_matrix.py *.csv --x-dopant Y --y-dopant Ga   # custom dopant axes
python compare_raman_matrix.py *.csv --raw-act --norm each        # legacy raw Raman activity S(nu)
python compare_raman_matrix.py *.csv --formats png pdf svg --export-data
```

Composition levels are read from the `File_Label` / CSV metadata (e.g.
`Ca01-Cl06`); levels missing from the input set are drawn as
"Not calculated" cells, and duplicate cells abort with an error. Grouping
can be forced to exact numerical metadata with `--group-by metadata` when
filename tags do not uniquely identify intended sample levels.

Auto-generated output filenames encode the run parameters, e.g.:

```bash
Na3PS4_compare_enorm.png   # per-spectrum normalised waterfall
Na3PS4_compare_gnorm.png   # global normalised waterfall
Na3PS4_compare_abs.png     # absolute overlaid
```

---

## CLI Reference

### `rotate.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--poscar` | `POSCAR` | Source POSCAR file |
| `--incar` | `INCAR` | INCAR file to copy into each subdirectory |
| `--kpoints` | `KPOINTS` | KPOINTS file to copy into each subdirectory |
| `--potcar` | `POTCAR` | POTCAR file to copy into each subdirectory |
| `--sbatch [FILE ...]` | auto-discover | *.sbatch files to copy; omit for auto-discovery, pass with no args to suppress |
| `--no-sbatch` | — | Disable sbatch copying entirely |
| `--axis-tol` | `0.90` | Cosine alignment threshold for row-to-axis reporting |
| `--strict-axis-report` | — | Abort if row-to-axis alignment is weak or ambiguous |

### `RASCBEC_phonopy.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--E` | auto | EFIELD_PEAD value — auto-read from `./1/OUTCAR`; override here if needed |
| `--gamma` | `0.25` | Lorentzian FWHM for broadening **in THz** (converted to cm⁻¹ internally) |
| `--temp` | `300.0` | Sample temperature in K for the Bose-Einstein factor |
| `--laser-wl` | `632.8` | Excitation laser wavelength in nm |
| `--freq-min` | `0.0` | Lower plot x-limit (cm⁻¹) |
| `--freq-max` | auto | Upper plot x-limit (cm⁻¹) |
| `--phonon-yaml` | `qpoints.yaml` | Phonopy phonon data file; accepts `qpoints.yaml` or `mesh.yaml` |
| `--irreps` | `irreps.yaml` | Phonopy irreducible representations file |
| `--bas` | auto | Basis species (or `+`-joined group) normalised to coefficient 1; denominator of dopant x-values |
| `--dops` | auto | Explicit dopant species list; overrides the 5 % abundance inference |
| `--input-mode` | `auto` | `auto` (NAC manifest/dir folders first, otherwise single YAML), `single`, or `multi` |
| `--directions-json` | auto | Override the `nac_directions.json` manifest location |
| `--nac-min-thz` | `0.1` | Multi-mode: minimum retained frequency in THz |
| `--translation-overlap` | `0.95` | Multi-mode: rigid-translation rejection threshold (mass-weighted overlap) |
| `--overwrite` | — | Allow replacing existing multi-mode outputs (single-mode behavior unchanged) |
| `--no-plot` | — | Skip PNG generation |
| `--no-sticks` | — | Hide stick spectrum in plot |
| `--n-labels` | `10` | Number of peak frequency labels |
| `--out-csv` | auto | Override output CSV name |
| `--out-png` | auto | Override output PNG name |

### `RASCBEC_VASP.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--E` | auto | EFIELD_PEAD value — auto-read from `./1/OUTCAR`; override here if needed |
| `--gamma` | `0.25` | Lorentzian FWHM for broadening **in THz** (converted to cm⁻¹ internally) |
| `--temp` | `300.0` | Sample temperature in K for the Bose-Einstein factor |
| `--laser-wl` | `632.8` | Excitation laser wavelength in nm |
| `--freq-min` | `0.0` | Lower plot x-limit (cm⁻¹) |
| `--freq-max` | auto | Upper plot x-limit (cm⁻¹) |
| `--bas` | auto | Basis species (or `+`-joined group) normalised to coefficient 1; denominator of dopant x-values |
| `--dops` | auto | Explicit dopant species list; overrides the 5 % abundance inference |
| `--no-plot` | — | Skip PNG generation |
| `--no-sticks` | — | Hide stick spectrum in plot |
| `--n-labels` | `10` | Number of peak frequency labels |
| `--out-csv` | auto | Override output CSV name |
| `--out-png` | auto | Override output PNG name |

### `plot_raman.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--dat` | single CSV in cwd | Input CSV file |
| `--out` | auto | Output PNG (defaults to CSV stem + .png) |
| `--raw-act` | — | Plot raw Raman activity S(ν) without 1/ν or thermal Bose factors |
| `--temp` | `300.0` | Sample temperature in K (Bose-Einstein factor) |
| `--laser-wl` | `632.8` | Excitation laser wavelength in nm |
| `--gamma` | `0.25` | Lorentzian FWHM **in THz** (converted to cm⁻¹ internally) |
| `--freq-min/max` | `0` / auto | Plot x-axis range (cm⁻¹; auto = highest mode rounded up to 50 cm⁻¹) |
| `--no-sticks` | — | Hide stick spectrum |
| `--n-labels` | `10` | Number of peak labels |
| `--min-label-sep` | `10` | Minimum cm⁻¹ separation between labelled peaks (shoulder merging) |
| `--prominence` | `0.01` | Prominence threshold (fraction of peak-to-baseline span) |
| `--title` | auto | Custom title; `'line1'`/`'line2'` select one line of the auto title |
| `--grid` | `both` | Grid visibility: both, x, y, or none |
| `--x-line` / `--y-line` | — | Vertical / horizontal reference lines |

### `compare_raman.py`

| Argument | Default | Description |
| --- | --- | --- |
| `files` (positional) | all `*.csv` in `--dir` | Input CSV files |
| `--dir` | `.` | Directory to scan when no files are given |
| `--temp` | `300.0` | Sample temperature in K for Stokes conversion |
| `--laser-wl` | `632.8` | Excitation laser wavelength in nm |
| `--raw-act` | — | Plot raw Raman activity S(ν) instead of Stokes intensity |
| `--gamma` | `0.25` | Lorentzian FWHM **in THz** |
| `--freq-min/max` | `0` / auto | Plot x-axis range (cm⁻¹) |
| `--norm` | — | Normalise all spectra to the global maximum |
| `--norm-each` | auto with `--offset` | Normalise each spectrum to its own maximum |
| `--offset` | `0.0` | Vertical spacing (waterfall); auto-applies `--norm-each` unless `--norm` is given |
| `--sticks` | — | Show stick spectrum |
| `--n-labels` | `10` | Prominence-selected peak labels per spectrum (waterfall only) |
| `--min-label-sep` | `10` | Minimum cm⁻¹ separation between labelled peaks |
| `--prominence` | `0.01` | Prominence threshold for label selection |
| `--labels` | auto | Override legend / trace labels |
| `--title` | auto | Custom title; `'line1'`/`'line2'` select one auto-title line |
| `--grid` | `x` | Grid visibility |
| `--figsize` | auto | Figure size (width height, in inches) |
| `--x-line` / `--y-line` | — | Vertical / horizontal reference lines |
| `--out` | auto | Output PNG name (encodes formula + normalisation mode) |

### `compare_raman_matrix.py`

| Argument | Default | Description |
| --- | --- | --- |
| `files` (positional) | `--pattern` in `--dir` | Input CSV files |
| `--dir` / `--pattern` | `.` / `*.csv` | Directory and glob used when no files are given |
| `--x-dopant` / `--y-dopant` | `Ca` / `Cl` | Dopant species defining the matrix axes |
| `--group-by` | `auto` | `auto`, `tags` (File_Label/filename), or `metadata` (exact numerical x-values) |
| `--reference` / `--no-reference` | — | Pristine CSV included in the matrix (overlaid on panels); `--no-reference` suppresses overlays |
| `--temp` | `300.0` | Sample temperature in K for Stokes conversion |
| `--laser-wl` | `632.8` | Excitation laser wavelength in nm |
| `--raw-act` | — | Plot raw Raman activity S(ν) instead of Stokes intensity |
| `--cutoff` | `1.0` | Stokes modes at/below this frequency (cm⁻¹) receive zero weight |
| `--fwhm` / `--gamma` | `0.25` | Common broadening FWHM **in THz** |
| `--freq-min/max` | `0` / auto | Display window (cm⁻¹); resolution grid still spans all modes |
| `--norm` | `each` | `each`, `global`, or `none` |
| `--no-sticks` / `--stick-max` | on / `1.0` | Stick spectrum toggle and cap as fraction of panel maximum |
| `--n-labels` | `4` | Prominence-selected peak labels per panel |
| `--min-label-sep` | `10` | Minimum cm⁻¹ separation between labelled peaks |
| `--prominence` | `0.02` | Prominence threshold for label selection |
| `--step` | `0.05` | Axis-step (composition grid resolution) |
| `--grid` | `x` | Grid visibility |
| `--figsize` / `--font-size` | auto | Figure size and base font size |
| `--color` | auto | Base panel colour (matplotlib colour spec) |
| `--ylim` | auto | Shared y-range for all panels (YMIN YMAX) |
| `--title` | auto | Custom title; `'line1'`/`'line2'` select one auto-title line |
| `--x-line` / `--y-line` | — | Vertical / horizontal reference lines |
| `--out` | auto | Output base path (extensions added per `--formats`) |
| `--formats` | `png` | Output formats: png, pdf, svg (plus a JSON manifest) |
| `--dpi` | 600 | Raster resolution for PNG |
| `--export-data` | — | Also write NPZ files (full-grid spectra, window spectra, mode weights) |

---

## Output CSV Format

Every CSV produced by the RASCBEC scripts includes a metadata header and,
on the phonopy path, an Irrep column for mode symmetry labels:

```bash
# Formula: Na3PS4
# Dopants: Ca=0.125 Cl=0.0625
# File_Label: Ca01-Cl06
# E_field: 0.02
# Mode,Freq_cm-1,Activity,Irrep
0001,83.672345,148.546403,A1
0002,91.234567,0.001234,E
...
```

The `#`-prefixed lines are skipped by standard readers.
`plot_raman.py`, `compare_raman.py`, and `compare_raman_matrix.py` parse
them to auto-build legend labels, trace labels, figure titles, and the
matrix composition axes. The `Irrep` column is used by the plotters to
annotate peaks with their symmetry label; it is absent in CSVs produced
by `RASCBEC_VASP.py` (which does not read `irreps.yaml`), and all tools
handle the 3-column format transparently.

NAC multi-direction outputs (`_NAC.csv`) carry additional provenance
lines and Lebedev-weighted activities:

```bash
# Formula: Na3PS4
# E_field: 0.02
# Phonon_treatment: NAC_Lebedev26
# Activity_semantics: quadrature_weight_times_raw_Placzek_activity
# Row_semantics: directional_sticks_not_unique_modes
# Mode,Freq_cm-1,Activity,Irrep
000001,83.672345,12.370534,
...
```

The companion `<name>.directions.json` file records the 26 directions,
their Lebedev weights, retained and skipped modes, and the thresholds
used.

---

## Known Limitations

### A1 and B1 modes may show zero activity in high-symmetry structures

In structures with D2d (point group -42m) site symmetry — such as
tetragonal Na₃PS₄ — all A1 and B1 phonon modes may yield exactly zero
Raman activity from RASCBEC. This is under investigation; it may reflect
a fundamental limitation of the BEC-derivative approximation for symmetric
breathing modes, or a symmetry-related cancellation in the eigenvector
projection. B2 and E modes are unaffected.

In low-symmetry (e.g., P1) doped supercells, strict A1/B1 symmetry is
broken and the affected modes are expected to acquire nonzero activities.

### Zener tunneling limits the safe EFIELD_PEAD range

For small-bandgap inorganic solids, the Zener criterion places a tight
upper bound on the safe field magnitude:

```bash
E_max (eV/Å) ≈ E_gap / (10 × N_k × c)
```

where `E_gap` is the DFT bandgap (eV), `N_k` is the number of k-points
along the field direction, and `c` is the longest lattice parameter (Å).
For Na₃PS₄ with a 5×5×5 Gamma-centered k-mesh, `E_max ≈ 0.006 eV/Å` —
far below the field strengths used in the original GeO₂ benchmark
(E = 0.1 eV/Å). Fields above this limit produce Zener-corrupted spectra
dominated by spurious high-frequency peaks. Fields too far below it may
produce near-zero activities for all modes due to signal falling below
the SCF precision floor (controlled by EDIFF). Recommended starting
point: `EFIELD_PEAD ≈ 0.5 × E_max`, `EDIFF = 1E-7`.

---

## Example: Rutile GeO₂

Input files are in the `Example/` folder (POSCAR, the 8 OUTCARs under
`OUTCAR*`, and the flat VASP DFPT exports `freqs_vasp.dat` /
`eigvecs_vasp.dat`). To run, assemble the expected layout and pass the
field magnitude explicitly (the example OUTCARs contain a single
EFIELD_PEAD entry):

```bash
cd Example
# Assemble subdirectory layout (OUTCAR1 -> ./1/OUTCAR, etc.)
for d in 1 m1 x mx y my z mz; do mkdir -p $d && cp OUTCAR$d $d/OUTCAR; done
python ../RASCBEC_VASP.py --E 0.1          # uses VASP DFPT eigenvectors directly
python ../RASCBEC_phonopy.py --E 0.1       # uses qpoints.yaml + irreps.yaml
```

Expected outputs (chemistry-based names, E supplied):

```bash
GeO2_E0.1.csv
GeO2_E0.1.png
```

Reference activities for cross-checking are in `Example/raman_vasp.dat`
and `Example/raman_phonopy.dat`.

---

## Citation

If you use this software, please cite the original RASCBEC paper:

```text
Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation via born
effective charge." Computer Physics Communications 307 (2025): 109425.
https://doi.org/10.1016/j.cpc.2024.109425
```

---

*Fork maintained by [PhnRvTjN](https://github.com/PhnRvTjN).
Original repository: [RASCBEC](https://github.com/rruuiizz/RASCBEC).*
