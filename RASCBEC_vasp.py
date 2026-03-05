#!/usr/bin/env python
"""
Title: Calculation of Raman Activities from VASP Output Files
Author: Rui Zhang
Date: 04/28/2025
License: GNU GENERAL PUBLIC LICENSE
Cite: Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation 
via born effective charge." Computer Physics Communications 307 (2025): 109425

Description:
 
This script reads input files generated from VASP simulations 
to calculate Raman activities for each phonon mode using RASCBEC method.

It requires:

(1) A POSCAR file (structure information).
(2) Eight OUTCAR files (OUTCAR1, OUTCARm1, OUTCARx, OUTCARmx, OUTCARy, OUTCARmy, 
    OUTCARz, OUTCARmz) containing Born Effective Charge data.
(3) Two additional phonon property files:
    freqs_vasp.dat: Phonon frequencies stored as a 3N×1 array, where N is the number of atoms.
    eigvecs_vasp.dat: Phonon eigenvectors stored as a 3N×3N array.

Dependencies:
- numpy

Usage:

Prepare the following files in the same directory:
POSCAR
OUTCAR1, OUTCARm1, OUTCARx, OUTCARmx, OUTCARy, OUTCARmy, OUTCARz, OUTCARmz
freqs_vasp.dat
eigvecs_vasp.dat
Run the script using Python 3 and NumPy installed:
python RASCBEC_vasp.py

Output:

raman_vasp.dat:
A file containing the calculated Raman activities for each phonon mode.

"""
#
import numpy as np

#get the structure information from POSCAR
def structure_info(poscar):
    
    #lattice constant
    box = np.loadtxt(poscar, skiprows=2, max_rows=3)
    
    #volume
    vol = np.linalg.det(box)
    
    #ion species and numbers
    species_n = np.loadtxt(poscar, skiprows=6, max_rows=1)
    
    #number of species
    ntype = len(species_n)

    #number of atoms
    nat = int(sum(species_n))
    
    #number of phonon modes
    modes = nat * 3

    return vol, species_n, ntype, nat, modes

#get the Born Effective Charge from OUTCAR 
def get_charges_from_OUTCAR(outcar, nat):
    
    #n*3 tensor
    charges = [[0.0] * 3 for _ in range(nat)]
    
    for line in outcar:
        if "BORN EFFECTIVE CHARGES (including local field effects)" in line:
            outcar.readline()
            for i in range(nat):
                outcar.readline()
                charges[i] = [list(map(float, outcar.readline().split()[1:4])) for _ in range(3)]
    
    return charges

#derivative of Born Effective charge with respect to electric field
def charge_derivative(charge1, chargem1, chargex, chargey, chargez, chargemx, chargemy, chargemz, E):

    #charges
    c1, cm1, cx, cy, cz, cmx, cmy, cmz = (np.array(ch).T for ch in (charge1, chargem1, chargex, chargey, chargez, chargemx, chargemy, chargemz))
    
    #charge derivative 3*3*3 tensor
    dq = np.zeros((3, 3, 3))

    for i in range(3):
        for j in range(3):
             dq[i,j,j] = (c1[i,j]-cm1[i,j])/E
    
    dq[2,0,1] = dq[2,1,0] = 0.5*(np.sqrt(2)*(cx[2,0]-cmx[2,0])-(c1[2,0]-cm1[2,0])-(c1[2,1]-cm1[2,1]))/E
    dq[0,0,1] = dq[0,1,0] = 0.5*(cx[0,0]-cmx[0,0]-cx[1,0]+cmx[1,0]-c1[0,0]+cm1[0,0]-c1[0,1]+cm1[0,1])/E
    dq[1,1,0] = dq[1,0,1] = 0.5*(cx[0,0]-cmx[0,0]+cx[1,0]-cmx[1,0]-c1[1,0]+cm1[1,0]-c1[1,1]+cm1[1,1])/E

    dq[0,1,2] = dq[0,2,1] = 0.5*(np.sqrt(2)*(cy[0,1]-cmy[0,1])-(c1[0,1]-cm1[0,1])-(c1[0,2]-cm1[0,2]))/E
    dq[1,1,2] = dq[1,2,1] = 0.5*(cy[1,1]-cmy[1,1]-cy[2,1]+cmy[2,1]-c1[1,1]+cm1[1,1]-c1[1,2]+cm1[1,2])/E
    dq[2,2,1] = dq[2,1,2] = 0.5*(cy[1,1]-cmy[1,1]+cy[2,1]-cmy[2,1]-c1[2,1]+cm1[2,1]-c1[2,2]+cm1[2,2])/E

    dq[1,2,0] = dq[1,0,2] = 0.5*(np.sqrt(2)*(cz[1,2]-cmz[1,2])-(c1[1,0]-cm1[1,0])-(c1[1,2]-cm1[1,2]))/E
    dq[2,2,0] = dq[2,0,2] = 0.5*(cz[2,2]-cmz[2,2]-cz[0,2]+cmz[0,2]-c1[2,2]+cm1[2,2]-c1[2,0]+cm1[2,0])/E
    dq[0,0,2] = dq[0,2,0] = 0.5*(cz[2,2]-cmz[2,2]+cz[0,2]-cmz[0,2]-c1[0,2]+cm1[0,2]-c1[0,0]+cm1[0,0])/E

    return dq


if __name__ == '__main__':
    
    #Parameters need to adjust
    ####################################################################
    #strength of electric field
    E = 0.1

    ####################################################################
    
    #read POSCAR
    vol, species_n, ntype, nat, modes = structure_info('POSCAR')
    
    #read OUTCAR
    charge_files = ['OUTCAR1', 'OUTCARm1', 'OUTCARx', 'OUTCARy', 'OUTCARz', 'OUTCARmx', 'OUTCARmy', 'OUTCARmz']
    
    charges = {}
    for file in charge_files:
        with open(file, 'r') as outcar:
            charges[file] = get_charges_from_OUTCAR(outcar, nat)
    
    #charge derivative
    dq = [charge_derivative(charges['OUTCAR1'][j], charges['OUTCARm1'][j], charges['OUTCARx'][j], charges['OUTCARy'][j], charges['OUTCARz'][j], charges['OUTCARmx'][j], charges['OUTCARmy'][j], charges['OUTCARmz'][j], E) for j in range(nat)]
    
    #read phonon frequency (3nat*1 array)

    #unit: cm^-1
    freqs = np.loadtxt('freqs_vasp.dat')
    
    #in case the phonon frequency needs to be outputted in a different unit
    #unit: cm^-1
    eigvals = freqs
    
    #unit: meV
    hws = freqs * 4.136 / 33.356
    
    #read phonon eigenvector (3nat*3nat array)
    eigvecs = np.loadtxt('eigvecs_vasp.dat')

    #other parameters
    #unit: meV
    kT = 8.617333e-2 * 298
    
    #unit: e^2/(eV*A)
    epslon_0 = 55.2635e-4
    
    #Raman activity
    activity = [ 0.0 for _ in range(modes) ]
    
    for s in range(modes):

        #eigvector divided by square root of mass
        eigvec = eigvecs[:,s].reshape((nat,3))
        
        #polariazation tensor
        ra_tot = np.zeros((3,3))
        for t in range(nat):
            dqt = dq[t]
            eigvect = eigvec[t,:]
            
            #Eq.7
            act = np.zeros((3,3,3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        act[i,j,k] = dqt[i,k,j]*eigvect[i]

            rat = act[0,:,:] + act[1,:,:] + act[2,:,:]
            ra_tot += rat
        
        #global constant
        ra = ra_tot /(4.0*np.pi*epslon_0)
        
        #mean polarizability derivative
        alpha = (ra[0][0] + ra[1][1] + ra[2][2])/3.0
        
        #anisotropy of the polarizability tensor derivative
        beta2 = ((ra[0][0] - ra[1][1])**2 + (ra[0][0] - ra[2][2])**2 + (ra[1][1] - ra[2][2])**2 + 6.0 * (ra[0][1]**2 + ra[0][2]**2 + ra[1][2]**2))/2.0
        
        #raman secattering activity
        activity[s] = 45.0*alpha**2 + 7.0*beta2
    
    ##################################################################
    
    #output file
    with open('raman_vasp.dat', 'w') as output_fh:

        output_fh.write("#  freq(cm-1)   activity\n")
        for i, (eigval, act) in enumerate(zip(eigvals, activity), 1):
            output_fh.write(f"{i:03d}  {eigval:10.7f}  {act:10.7f}\n")
    
    ###################################################################
