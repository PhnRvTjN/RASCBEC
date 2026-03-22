# RASCBEC - Raman Spectroscopy Calculation via Born Effective Charge

> **This is a community fork** of the original
> [RASCBEC](https://github.com/rruuiizz/RASCBEC) by Rui Zhang et al.
> The underlying physics and RASCBEC method are unchanged. This fork
> refactors the scripts for practical use on large doped supercells, adds a
> complete plotting pipeline, and introduces a multi-composition comparison
> tool.

---

## Improvements Over the Original

| Feature | Original | This Fork |
|---|---|---|
| Atomic masses | Hardcoded arrays | Auto-lookup via **pymatgen** |
| OUTCAR layout | Flat files (`OUTCAR1`, `OUTCARm1`, …) | Subdirectory layout (`./1/OUTCAR`, `./m1/OUTCAR`, …) |
| OUTCAR BEC reader | Forward scan (reads entire file) | Reverse scan — finds last BEC block directly, avoids full re-read of multi-GB OUTCARs |
| POSCAR rotation | `rotate.py` outputs 3 flat files (`ex.POSCAR.vasp`, …) | `rotate.py` (improved) creates all 8 subdirectories with correct POSCAR in each |
| Phonopy input | `.dat` intermediate files via `qpoints_to_eigfreq.py` / `mesh_to_eigfreq.py` | Direct YAML reading — `RASCBEC_phonopy.py` reads `qpoints.yaml` (or `mesh.yaml`) and `irreps.yaml` with no intermediate step |
| Mode filtering | Fixed `3N`; frequency-threshold cutoff | Two independent filters: `freq < 0.1` (imaginary modes) and `ir_label is None` (acoustic/unassigned); a mode with a negative frequency but a valid irrep label is still excluded |
| Mode symmetry labels | Not included | `irreps.yaml` integrated; irrep label (A1, B2, E, …) stored per mode in CSV and displayed on plot peak annotations |
| E-field input | Manual `--E` flag required | Auto-read from `./1/OUTCAR` (EFIELD_PEAD); `--E` overrides |
| Output filename | `raman_phonopy.dat` / `raman_vasp.dat` | Chemistry-based: `raman_<formula>_<dopants>_E<field>.csv` |
| CSV metadata | None | Header lines: `# Formula`, `# Dopants`, `# E_field`, `# Mode,Freq_cm-1,Activity,Irrep` |
| Activity calculation | Nested Python loops O(N·27) | Vectorised `np.einsum('tikj,sti->sjk')` — BLAS-backed, seconds instead of hours for large supercells |
| Plot generation | Not included | Integrated via `plot_raman.py` (Lorentzian broadening, peak labels with irrep, auto title) |
| Comparison plots | Not included | `compare_raman.py` — waterfall / overlaid, absolute or normalised |
| CLI arguments | None | `--E`, `--gamma`, `--freq-min/max`, `--phonon-yaml`, `--irreps`, `--no-plot`, `--out-csv/png`, … |
| Code structure | Single monolithic script | Modular functions (BEC reader, derivative builder, activity calculator, writer) |

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
|---|---|
| `--incar`, `--kpoints`, `--potcar` | Override default paths |
| `--sbatch FILE [FILE ...]` | Explicit sbatch list |
| `--no-sbatch` | Suppress sbatch copying |

---

### Diagnostic output

The corrected script prints the detected row-to-axis mapping and the
resolved legacy routing on every run so you can verify the folder
assignment before submitting VASP jobs:

```
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

## Dependencies

```
python >= 3.8
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

```
RASCBEC/
├── rotate.py               - Rotate POSCAR; create all 8 subdirectories for BEC runs
├── RASCBEC_phonopy.py      - Raman activities from phonopy YAML (qpoints.yaml + irreps.yaml)
├── RASCBEC_VASP.py         - Raman activities using VASP DFPT eigenvectors (refactored)
├── plot_raman.py           - Single-composition Raman spectrum plotter
├── compare_raman.py        - Multi-composition / multi-E-field comparison plotter
├── qpoints_to_eigfreq.py   - Legacy: convert qpoints.yaml -> .dat (no longer needed)
├── mesh_to_eigfreq.py      - Legacy: convert mesh.yaml -> .dat (no longer needed)
├── RASCBEC_vasp.py         - Original upstream script (preserved for reference)
├── Code/                   - Original upstream scripts
└── Example/                - GeO2 rutile example inputs and outputs
```

> `RASCBEC_phonopy.py` and `RASCBEC_VASP.py` (uppercase) are the improved
> scripts in this fork. `RASCBEC_vasp.py` (lowercase) is the unmodified
> original kept for reference.
>
> `qpoints_to_eigfreq.py` and `mesh_to_eigfreq.py` are kept for reference
> but are no longer part of the workflow. `RASCBEC_phonopy.py` now reads
> `qpoints.yaml` and `irreps.yaml` directly.

---

## Workflow

### 1 - Rotate POSCAR

```bash
python rotate.py
```

Creates all 8 subdirectories with the correct POSCAR in each:

```
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

```
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
```
./x  INCAR:  EFIELD_PEAD = +E   +δ   +δ
./mx INCAR:  EFIELD_PEAD = -E   +δ   +δ
```

**Incorrect** (inactive components sign-flipped — do NOT cancel, introduces ~δ/E error):
```
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
python RASCBEC_phonopy.py --gamma 15 --freq-min 50 --freq-max 600

# Use mesh.yaml instead of qpoints.yaml
python RASCBEC_phonopy.py --phonon-yaml mesh.yaml

# Custom irreps file location
python RASCBEC_phonopy.py --irreps path/to/irreps.yaml
```

**VASP DFPT eigenvectors** (frequencies in cm⁻¹, already mass-normalised):
```bash
python RASCBEC_VASP.py
python RASCBEC_VASP.py --gamma 15 --no-plot
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

Output filenames are derived from the POSCAR chemistry:
```
raman_Na3PS4_E0.02.csv                   # undoped
raman_Na3PS4_Ca0.125_Cl0.5_E0.02.csv    # Ca+Cl co-doped
```

### 5 - Plot a Single Spectrum

`plot_raman.py` is called automatically by both RASCBEC scripts but can
also be run standalone:

```bash
python plot_raman.py --dat raman_Na3PS4_Ca0.125_Cl0.5_E0.02.csv
python plot_raman.py --dat raman_Na3PS4_E0.02.csv --gamma 10 --freq-min 0 --freq-max 600
```

### 6 - Compare Multiple Compositions

```bash
# Waterfall (auto-normalises each, 10 peak labels per spectrum)
python compare_raman.py raman_*.csv --offset 1.2

# Global normalisation waterfall
python compare_raman.py raman_*.csv --normalize --offset 1.2

# Overlaid, absolute scale (no labels)
python compare_raman.py raman_*.csv

# Same composition, different E-field (E auto-appended to legend)
python compare_raman.py raman_Na3PS4_E0.01.csv raman_Na3PS4_E0.02.csv raman_Na3PS4_E0.05.csv --offset 1.2

# Custom labels / output name
python compare_raman.py raman_*.csv --offset 1.2 --labels "Undoped" "Ca-doped" "Ca/Cl-doped" --out fig1.png
```

Auto-generated output filename encodes the run parameters:
```
raman_compare_Na3PS4_enorm.png   # per-spectrum normalised waterfall
raman_compare_Na3PS4_gnorm.png   # global normalised waterfall
raman_compare_Na3PS4_abs.png     # absolute overlaid
```

---

## CLI Reference

### `rotate.py`

| Argument | Default | Description |
|---|---|---|
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
|---|---|---|
| `--E` | auto | EFIELD_PEAD value — auto-read from `./1/OUTCAR`; override here if needed |
| `--gamma` | `10.0` | Lorentzian FWHM for broadening (cm⁻¹) |
| `--freq-min` | `0.0` | Lower plot x-limit (cm⁻¹) |
| `--freq-max` | auto | Upper plot x-limit (cm⁻¹) |
| `--phonon-yaml` | `qpoints.yaml` | Phonopy phonon data file; accepts `qpoints.yaml` or `mesh.yaml` |
| `--irreps` | `irreps.yaml` | Phonopy irreducible representations file |
| `--no-plot` | — | Skip PNG generation |
| `--no-sticks` | — | Hide stick spectrum in plot |
| `--n-labels` | `20` | Number of peak frequency labels |
| `--out-csv` | auto | Override output CSV name |
| `--out-png` | auto | Override output PNG name |

### `RASCBEC_VASP.py`

| Argument | Default | Description |
|---|---|---|
| `--E` | auto | EFIELD_PEAD value — auto-read from `./1/OUTCAR`; override here if needed |
| `--gamma` | `10.0` | Lorentzian FWHM for broadening (cm⁻¹) |
| `--freq-min` | `0.0` | Lower plot x-limit (cm⁻¹) |
| `--freq-max` | auto | Upper plot x-limit (cm⁻¹) |
| `--no-plot` | — | Skip PNG generation |
| `--no-sticks` | — | Hide stick spectrum in plot |
| `--n-labels` | `20` | Number of peak frequency labels |
| `--out-csv` | auto | Override output CSV name |
| `--out-png` | auto | Override output PNG name |

### `plot_raman.py`

| Argument | Default | Description |
|---|---|---|
| `--dat` | required | Input CSV file |
| `--out` | auto | Output PNG (defaults to CSV stem + .png) |
| `--gamma` | `10.0` | Lorentzian FWHM (cm⁻¹) |
| `--freq-min/max` | `0` / auto | Plot x-axis range |
| `--no-sticks` | — | Hide stick spectrum |
| `--n-labels` | `15` | Number of peak labels |
| `--title` | auto | Custom plot title |

### `compare_raman.py`

| Argument | Default | Description |
|---|---|---|
| `--gamma` | `10.0` | Lorentzian FWHM (cm⁻¹) |
| `--freq-min/max` | `0` / auto | Plot x-axis range |
| `--normalize` | — | Normalise all to global maximum |
| `--normalize-each` | auto with `--offset` | Normalise each to its own maximum |
| `--offset` | `0.0` | Vertical offset per spectrum (waterfall); auto-applies `--normalize-each` |
| `--sticks` | — | Show stick spectrum |
| `--n-labels` | `10` | Peak labels per spectrum (waterfall only) |
| `--labels` | auto | Override legend labels |
| `--out` | auto | Output PNG name |

---

## Output CSV Format

Every CSV produced by the RASCBEC scripts includes a metadata header and
an Irrep column (phonopy path) for mode symmetry labels:

```
# Formula: Na3PS4
# Dopants: Ca(x=0.125)/Cl(x=0.5)
# E_field: 0.02
# Mode,Freq_cm-1,Activity,Irrep
0001,83.672345,148.546403,A1
0002,91.234567,0.001234,E
...
```

The `#`-prefixed lines are skipped by standard readers. `compare_raman.py`
parses them to auto-build legend labels and figure titles. The `Irrep`
column is used by `plot_raman.py` and `compare_raman.py` to annotate
peaks with their symmetry label; it is absent in CSVs produced by
`RASCBEC_VASP.py` (which does not read `irreps.yaml`), and both plotters
handle the 3-column format transparently.

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

```
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

Input files are in the `Example/` folder. Run:

```bash
cd Example
python ../RASCBEC_VASP.py     # uses VASP DFPT eigenvectors directly
python ../RASCBEC_phonopy.py  # uses qpoints.yaml + irreps.yaml
```

Expected outputs (chemistry-based names, E auto-read from OUTCAR):
```
raman_GeO2_E0.02.csv
raman_GeO2_E0.02.png
```

---

## Citation

If you use this software, please cite the original RASCBEC paper:

```
Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation via born
effective charge." Computer Physics Communications 307 (2025): 109425.
https://doi.org/10.1016/j.cpc.2024.109425
```

---

*Fork maintained by [PhnRvTjN](https://github.com/PhnRvTjN).
Original repository: [RASCBEC](https://github.com/rruuiizz/RASCBEC).*
