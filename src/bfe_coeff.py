"""
Code to compute the BFE coefficients from 
a N-body simulated halo.

Dependencies:
------------
1. Biff : Compute the BFE coefficients, and accelerations.
2. Scipy : interpolation routines.
3. pygadgetreader : read Gadget snapshots.
4. Astropy : units and constans.
5. Octopus : to compute COM of the halo.

Code structure:
---------------

- Compute the Snlm, Tnlm Coefficients from a Snapshot. (The snapshot
  is re-centered to the halo COM.) The centering is done here by using
  octopus.
- Interpolate the coefficients.
- Integrate orbits with the interpolated coeffciients.
- The resulting orbit is in galactocentric orbtis!

Input parameters:
-----------------

path : path to simulations
snap_name : name of the snapshots without the _XXX

Output:
-------
Orbit.


Important:
----------

If comparison wants to be done with Gadget orbits
please use the gravitational constant G of Gadget
G=43007.1kpc3/(Gyr2Msun)/1E10. This is what the code is using right
now!


to-do:
------
- Compute coefficients separately.
- Truncate halo
- Include the LMC

"""

import numpy as np
import biff
from scipy import interpolate
from pygadgetreader import *
from astropy import constants
from astropy import units as u
import sys
import octopus
import leapfrog_bfe

def re_center_halo(pos, r_cm):
    """
    Re-center a halo.
    """
    for i in range(3):
        pos[:,i] = pos[:,i] - r_cm[i]
    return pos

def snap_times_nbody(path, snap_name, N_initial):
    """
    Computed the times between snapshots of the n-body simulation.

    """
    for i in range(N_initial, N_initial+1):
        dt = readheader(path+snap_name+'_{:03d}'.format(i+1), 'time') - readheader(path+snap_name+'_{:03d}'.format(i), 'time')
        return dt

def write_coefficients(S, T, times, file_name, t_max, nmax, lmax, r_s, path):
    """
    Writes the coefficients Snlm and Tnlm in a file.
    for this the coefficients are flattened into a 1d array.
    Parameters:
    -----------
    S : Matrix of coefficients Snlm.
    T : Matrix of coefficients Tnlm.
    """


    S_flat = S.flatten()
    T_flat = T.flatten()

    f = open('../coefficients/ST_'+file_name, 'w')
    f.write('# number of time steps : {:.3f} \n'.format(t_max))
    f.write('# nmax = {:0>2d}, lmax = {:0>2d} \n'.format(nmax, lmax))
    f.write('# orginal matrix shape [{:.1f},{:0>1d}, {:0>1d}, {:0>1d} ] \n'.format(t_max, nmax, lmax, lmax))
    f.write('# Snlm, Tnlm \n')
    f.write('# r_s = {:.2f} \n'.format(r_s))
    f.write('# ICs from {} \n'.format(path))
    for i in range(len(S_flat)):
        f.write("{:.3f} {:.3f} \n".format(S_flat[i], T_flat[i]))

    f.close()

    f = open('../coefficients/times_' + file_name, 'w')

    f.write('# Times in Gyrs \n')
    for i in range(len(times)):
        f.write('{:.3f}\n'.format(times[i]))
    f.close()

def disk_bulge(path,  snap_name, N_initial):
    """
    See is there are disk or bulge particles!

    """
    n_diskpart = readheader(path + snap_name + "_{:03d}".format(N_initial), 'diskcount')
    n_bulgepart = readheader(path + snap_name + "_{:03d}".format(N_initial), 'bulgecount')

    if (n_diskpart == 0):
        disk = 0
    else:
        disk = 1

    if (n_bulgepart == 0):
        bulge = 0
    else:
        bulge = 1

    return disk, bulge

def compute_coeffs_from_snaps(path, snap_name, N_initial, \
                              N_final, Nmax, Lmax, r_s, disk=0):
    """
    Compute the coefficients from a series of snapshots of N-body
    simulations.
    Dependecies: pygadgetreader and octopus.

    Input:
    ------

    path : string
        path to snapshots
    snap_name : string
        snapshots base name
    N_intitial : int
        initial number of the snapshot.
    N_final : int
        final number of the snaphot in the simulation.
    Nmax : int
        Max number of expansion terms in the radial terms of the BFE.
    Lmax:
        Max number of expansion terms in the angular terms of the BFE.
    r_s: float
        Halo scale length in kpc.
    disk : int
        If disk is present disk==1, otherwise disk==0 (default)

    Return:
    -------

    S : matrix of shape [t, nmax+1, lmax+1, lmax+1]
    T : matrix of shape [t, nmax+1, lmax+1, lmax+1]

    """
    t = N_final - N_initial

    S = np.zeros((t, Nmax+1, Lmax+1, Lmax+1))
    T = np.zeros((t, Nmax+1, Lmax+1, Lmax+1))

    for i in range(N_initial, N_final, 1):
        pos = readsnap(path+snap_name+'_{:03d}'.format(i), 'pos', 'dm')
        mass = readsnap(path+snap_name+'_{:03d}'.format(i), 'mass', 'dm')

        ## Computing the COM.

        if disk==1:
            pos_disk = readsnap(path+snap_name+'_{:03d}'.format(i), 'pos', 'disk')
            pot_disk = readsnap(path+snap_name+'_{:03d}'.format(i), 'pot', 'disk')
            rcm, vcm = octopus.CM_disk_potential(pos_disk, pos_disk, pot_disk)
        elif disk==0:
            rcm, vcm = octopus.CM(pos, pos) # not using velocities!

        ## Re-centering halo.
        pos_cm = re_center_halo(pos, rcm)

        # Computing Coefficients.
        S[i], T[i] = biff.compute_coeffs_discrete(np.ascontiguousarray(pos_cm.astype(np.double)), mass.astype(np.double)*1E10, Nmax, Lmax, r_s)

    return S, T






if __name__ == "__main__":

    if(len(sys.argv)!=9):
        print('///////////////////////////////////////////////////////////////')
        print('')
        print('Usage:')
        print('------')
        print('path : path to nbody simulations snapshots')
        print('snap_name: name of the snapshot base name')
        print('init_snap : number of the initial snapshot')
        print('final_snap : number of the final snapshot')
        print('nmax')
        print('lmax')
        print('r_s : scale lenght of the halo in kpc')
        print('out name : name of the output file with the coefficients')
        print('')
        print('////////////////////////////////////////////////////////////////')
        exit(0)


    path = sys.argv[1]
    snap_name = sys.argv[2]

    init_snap = int(sys.argv[3])
    final_snap = int(sys.argv[4])

    nmax = int(sys.argv[5])
    lmax = int(sys.argv[6])

    r_s = float(sys.argv[7])

    out_name = sys.argv[8]

    N_snaps = final_snap - init_snap
    dt_nbody = snap_times_nbody(path, snap_name, init_snap)

    times = np.linspace(init_snap*dt_nbody, final_snap*dt_nbody, N_snaps)

    disk, bulge = disk_bulge(path, snap_name, init_snap)

    print('disk, bulge', disk, bulge)

    # Computing coefficients.
    print('Computing BFE coefficients')
    S_nlm, T_nlm = compute_coeffs_from_snaps(path, snap_name, init_snap, final_snap, nmax, lmax, r_s, disk)

    print('Writting coefficients')
    write_coefficients(S_nlm, T_nlm, times, out_name, N_snaps ,nmax, lmax, r_s, path)

