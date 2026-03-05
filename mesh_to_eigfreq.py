#!/usr/bin/env python3
"""
Convert phonopy mesh.yaml → freqs_phonopy.dat + eigvecs_phonopy.dat
for RASCBEC post-processing.

Output shapes (RASCBEC convention):
  freqs_phonopy.dat  : (N_modes,)       — one frequency per line, THz
  eigvecs_phonopy.dat: (3N x N_modes)   — column s = eigenvector of mode s
                       (i.e. eigvecs_clean is saved TRANSPOSED)
"""

import yaml
import numpy as np

# ── User settings ────────────────────────────────────────────────────────────
MESH_YAML    = 'mesh.yaml'
FREQ_CUTOFF  = 0.1   # THz — remove modes below this (acoustic + imaginary)
# ─────────────────────────────────────────────────────────────────────────────

with open(MESH_YAML, 'r') as f:
    data = yaml.safe_load(f)

natom  = data['natom']               # 125
nmodes = 3 * natom                   # 375
bands  = data['phonon'][0]['band']   # Gamma point = phonon[0]
assert len(bands) == nmodes, f"Expected {nmodes} bands, got {len(bands)}"

# ── Frequencies (THz) ────────────────────────────────────────────────────────
freqs_all = np.array([b['frequency'] for b in bands])   # shape (375,)

# ── Eigenvectors (raw, NOT mass-weighted) ────────────────────────────────────
# eigvecs_all[s, :] = mode s displacement vector, length 3N
# ordering: [x1, y1, z1, x2, y2, z2, ..., x_N, y_N, z_N]
eigvecs_all = np.zeros((nmodes, nmodes))
for s, band in enumerate(bands):
    idx = 0
    for atom_vecs in band['eigenvector']:   # natom entries
        for comp in atom_vecs:              # 3 components [real, imag]
            eigvecs_all[s, idx] = comp[0]   # real part (Gamma: imag ≈ 0)
            idx += 1

# ── Filter modes ─────────────────────────────────────────────────────────────
keep          = freqs_all >= FREQ_CUTOFF       # boolean mask
freqs_clean   = freqs_all[keep]                # shape (N_modes,)
eigvecs_clean = eigvecs_all[keep, :]           # shape (N_modes, 3N)

n_modes   = keep.sum()
n_removed = nmodes - n_modes
print(f"Total modes : {nmodes}")
print(f"Removed     : {n_removed}  (freq < {FREQ_CUTOFF} THz)")
print(f"Retained    : {n_modes}")
print(f"Freq range  : {freqs_clean.min():.4f} - {freqs_clean.max():.4f} THz")
print(f"             ({freqs_clean.min()*33.356:.2f} - {freqs_clean.max()*33.356:.2f} cm⁻¹)")

# ── Norm check (rows of eigvecs_clean should be unit vectors) ────────────────
norms = np.sum(eigvecs_clean**2, axis=1)
print(f"Eigvec norm : min={norms.min():.5f}  max={norms.max():.5f}  (should be ~1.0)")

# ── Write files ───────────────────────────────────────────────────────────────
# freqs : (N_modes,)  — straightforward
np.savetxt('freqs_phonopy.dat', freqs_clean, fmt='%.10f')

# eigvecs: RASCBEC reads eigvecs[:, s] as mode s  →  must be (3N × N_modes)
#          eigvecs_clean is (N_modes × 3N), so we TRANSPOSE before saving
eigvecs_out = eigvecs_clean.T                  # shape (3N, N_modes) = (375, N_modes)
np.savetxt('eigvecs_phonopy.dat', eigvecs_out, fmt='%.10f')

# ── Final shape verification ──────────────────────────────────────────────────
check_freqs  = np.loadtxt('freqs_phonopy.dat')
check_eigv   = np.loadtxt('eigvecs_phonopy.dat')

print(f"\nFile shapes written to disk:")
print(f"  freqs_phonopy.dat   : {check_freqs.shape}   ← should be ({n_modes},)")
print(f"  eigvecs_phonopy.dat : {check_eigv.shape}  ← should be ({natom*3}, {n_modes})")

assert check_freqs.shape == (n_modes,),          "freqs shape mismatch!"
assert check_eigv.shape  == (natom*3, n_modes),  "eigvecs shape mismatch!"
assert check_eigv.shape[1] == len(check_freqs),  "cols of eigvecs != len(freqs)!"

print("\n✓ All shape checks passed — ready for RASCBEC.")
