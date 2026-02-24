# RASCBEC — Raman Spectroscopy Calculation via Born Effective Charge

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
| Mode count | Fixed `3N` | Variable — filtered imaginary / acoustic modes handled |
| Output filename | `raman_phonopy.dat` / `raman_vasp.dat` | `raman_<composition>_<E>.csv` (E-field encoded) |
| CSV metadata | None | Header lines: `# Formula`, `# Dopants`, `# E_field`, `# Composition` |
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
├── rotate.py            — Rotate POSCAR for ±x/y/z E-field directions
├── RASCBEC_phonopy.py   — Raman activities using phonopy eigenvectors (refactored)
├── RASCBEC_VASP.py      — Raman activities using VASP DFPT eigenvectors (refactored)
├── plot_raman.py        — Single-composition Raman spectrum plotter
├── compare_raman.py     — Multi-composition / multi-E-field comparison plotter
├── RASCBEC_vasp.py      — Original upstream script (preserved for reference)
├── Code/                — Original upstream scripts
└── Example/             — GeO2 rutile example inputs and outputs
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

Generates `ex.POSCAR.vasp`, `ey.POSCAR.vasp`, `ez.POSCAR.vasp` for
electric-field directions along ±x, ±y, ±z.

### 2 — Run 8 BEC Calculations

Organise VASP calculations in subdirectories.
The scripts expect the following layout in your working directory:

```
9-RASCBEC/
├── POSCAR
├── freqs_phonopy.dat      (or freqs_vasp.dat)
├── eigvecs_phonopy.dat    (or eigvecs_vasp.dat)
├── 1/OUTCAR              — E-field along +z (unrotated)
├── m1/OUTCAR             — E-field along −z (unrotated)
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
python RASCBEC_phonopy.py --E 0.02 --gamma 15 --freq-min 50 --freq-max 600
```

**VASP DFPT eigenvectors** (frequencies in cm⁻¹, already mass-normalised):
```bash
python RASCBEC_VASP.py
python RASCBEC_VASP.py --E 0.02 --gamma 15 --no-plot
```

> **Key distinction:** VASP eigenvectors are already mass-normalised by
> VASP. `RASCBEC_VASP.py` does *not* divide by √(mass), whereas
> `RASCBEC_phonopy.py` does. Using the wrong script with the wrong
> eigenvector type will give incorrect Raman intensities.

Output (run from `.../06-12/9-RASCBEC/` with `--E 0.02`):
```
raman_06-12_0.02.csv
raman_06-12_0.02.png
```

### 4 — Plot a Single Spectrum

`plot_raman.py` is used internally by both RASCBEC scripts but can also
be called standalone:

```bash
python plot_raman.py --dat raman_06-12_0.02.csv --gamma 10 --freq-min 0 --freq-max 600
```

### 5 — Compare Multiple Compositions

```bash
# Waterfall (auto applies --normalize-each, 10 peak labels per spectrum)
python compare_raman.py raman_*.csv --offset 1.2

# Global normalisation waterfall
python compare_raman.py raman_*.csv --normalize --offset 1.2

# Overlaid, absolute scale (no labels)
python compare_raman.py raman_*.csv

# Same composition, different E-field (E auto-appended to legend)
python compare_raman.py raman_06-12_0.01.csv raman_06-12_0.02.csv raman_06-12_0.05.csv --offset 1.2

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
| `--E` | `0.02` | EFIELD_PEAD value used in BEC calculations (eV/Å) |
| `--gamma` | `10.0` | Lorentzian FWHM for broadening (cm⁻¹) |
| `--freq-min` | `0.0` | Lower plot x-limit (cm⁻¹) |
| `--freq-max` | auto | Upper plot x-limit (cm⁻¹) |
| `--no-plot` | — | Skip PNG generation |
| `--no-sticks` | — | Hide stick spectrum in plot |
| `--n-labels` | `20` | Number of peak frequency labels |
| `--out-csv` | auto | Override output CSV name |
| `--out-png` | auto | Override output PNG name |

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
# Dopants: Ca(x=0.06)/Cl(x=0.12)
# E_field: 0.02
# Composition: 06-12
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

Expected outputs:
- `raman_Example_0.02.csv`
- `raman_Example_0.02.png`

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
