# RASCBEC - Raman Spectroscopy Calculation via Born Effective Charge

> **This is a community fork** of the original
> [RASCBEC](https://github.com/RZhang05/RASCBEC) by Rui Zhang et al.
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
| POSCAR rotation | No subdirectory generation | `rotate.py` generates all 8 subdirectories (`./1/`, `./m1/`, `./x/`, …) |
| Mode count | Fixed `3N` | Variable — filtered imaginary / acoustic modes handled |
| E-field input | Manual `--E` flag required | Auto-read from `./1/OUTCAR` (EFIELD_PEAD); `--E` overrides |
| Output filename | `raman_phonopy.dat` / `raman_vasp.dat` | Chemistry-based: `raman_<formula>_<dopants>_E<field>.csv` |
| CSV metadata | None | Header lines: `# Formula`, `# Dopants`, `# E_field` |
| Plot generation | Not included | Integrated via `plot_raman.py` (Lorentzian broadening, peak labels, auto title) |
| Comparison plots | Not included | `compare_raman.py` — waterfall / overlaid, absolute or normalised |
| CLI arguments | None | `--E`, `--gamma`, `--freq-min/max`, `--no-plot`, `--out-csv/png`, … |
| Code structure | Single monolithic script | Modular functions (BEC reader, derivative builder, activity calculator, writer) |

---

## Dependencies

```
python >= 3.8
numpy
scipy
matplotlib
pymatgen
```

Install all at once:
```bash
pip install numpy scipy matplotlib pymatgen
```

---

## Repository Structure

```
RASCBEC/
├── rotate.py            - Rotate POSCAR and create all 8 subdirectories for BEC runs
├── RASCBEC_phonopy.py   - Raman activities using phonopy eigenvectors (refactored)
├── RASCBEC_VASP.py      - Raman activities using VASP DFPT eigenvectors (refactored)
├── plot_raman.py        - Single-composition Raman spectrum plotter
├── compare_raman.py     - Multi-composition / multi-E-field comparison plotter
├── RASCBEC_vasp.py      - Original upstream script (preserved for reference)
├── Code/                - Original upstream scripts
└── Example/             - GeO2 rutile example inputs and outputs
```

> `RASCBEC_phonopy.py` and `RASCBEC_VASP.py` (uppercase) are the improved
> scripts in this fork. `RASCBEC_vasp.py` (lowercase) is the unmodified
> original kept for reference.

---

## Workflow

### 1 — Rotate POSCAR

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

### 2 — Run 8 BEC Calculations

After VASP finishes, your working directory should look like:

```
9-RASCBEC/
├── POSCAR
├── freqs_phonopy.dat      (or freqs_vasp.dat)
├── eigvecs_phonopy.dat    (or eigvecs_vasp.dat)
├── 1/OUTCAR              — E-field along + (unrotated)
├── m1/OUTCAR             — E-field along − (unrotated)
├── x/OUTCAR              — E-field along +x (rotated)
├── mx/OUTCAR             — E-field along −x
├── y/OUTCAR              — E-field along +y (rotated)
├── my/OUTCAR             — E-field along −y
├── z/OUTCAR              — E-field along +z (rotated)
└── mz/OUTCAR             — E-field along −z
```

### 3 — Compute Raman Activities

**Phonopy eigenvectors** (frequencies in THz, un-mass-normalised):
```bash
python RASCBEC_phonopy.py
python RASCBEC_phonopy.py --gamma 15 --freq-min 50 --freq-max 600
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

Output filenames are derived from the POSCAR chemistry — no directory
naming convention required:
```
raman_Na3PS4_E0.02.csv                   # undoped
raman_Na3PS4_Ca0.125_Cl0.5_E0.02.csv    # Ca+Cl co-doped
```

### 4 — Plot a Single Spectrum

`plot_raman.py` is called automatically by both RASCBEC scripts but can
also be run standalone:

```bash
python plot_raman.py --dat raman_Na3PS4_Ca0.125_Cl0.5_E0.02.csv
python plot_raman.py --dat raman_Na3PS4_E0.02.csv --gamma 10 --freq-min 0 --freq-max 600
```

### 5 — Compare Multiple Compositions

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

### `RASCBEC_phonopy.py` / `RASCBEC_VASP.py`

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

Every CSV produced by the RASCBEC scripts includes a metadata header:

```
# Formula: Na3PS4
# Dopants: Ca(x=0.125)/Cl(x=0.5)
# E_field: 0.02
# Mode,Freq_cm-1,Activity
0001,83.672345,148.546403
0002,91.234567,0.001234
...
```

The `#`-prefixed lines are skipped by `numpy.loadtxt` and other standard
readers. `compare_raman.py` parses them to auto-build legend labels and
figure titles.

---

## Example: Rutile GeO₂

Input files are in the `Example/` folder. Run:

```bash
cd Example
python ../RASCBEC_VASP.py     # uses freqs_vasp.dat + eigvecs_vasp.dat
python ../RASCBEC_phonopy.py  # uses freqs_phonopy.dat + eigvecs_phonopy.dat
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
Original repository: [RZhang05/RASCBEC](https://github.com/RZhang05/RASCBEC).*
