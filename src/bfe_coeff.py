"""
Code to compute the BFE coefficients from
a N-body simulated halo.

Dependencies:
------------
1. Biff : Compute the BFE coefficients, and accelerations.
2. Scipy : interpolation routines.
3. pygadgetreader : read Gadget snapshots.
4. Astropy : units and constants.
5. Octopus : to compute COM of the halo.

Code structure:
---------------

- Compute the Snlm, Tnlm Coefficients from a Snapshot. (The snapshot
  is re-centered to the halo COM.) The centering is done here by using
  octopus.
- Interpolate the coefficients.
- Integrate orbits with the interpolated coefficients.
- The resulting orbit is in galactocentric coordinates.

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
G=43007.1kpc3/(Gyr2Msun)/1E10. 
This is what the code uses as default


to-do:
------
- Truncate halo
- Different selection for the LMC

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

## Code main functions:

def re_center_halo(pos, r_cm):
    """
    Re-center a halo positions or velocities.
    """

    for i in range(3):
        pos[:,i] = pos[:,i] - r_cm[i]
    return pos

def snap_times_nbody(path, snap_name, N_initial):
    """
    Computed the times between snapshots of the n-body simulation.

    """
    for i in range(N_initial, N_initial+1):
        dt = readheader(path+snap_name+'_{:03d}'.format(i+1), 'time')\
             - readheader(path+snap_name+'_{:03d}'.format(i), 'time')
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


    # from 3d to 1d coefficients.

    S_flat = S.flatten()
    T_flat = T.flatten()


    f = open('../coefficients/ST_'+file_name, 'w')
    f.write('# number of time steps : {:.3f} \n'.format(t_max))
    f.write('# nmax = {:0>2d}, lmax = {:0>2d} \n'.format(nmax, lmax))
    f.write('# orginal matrix shape [{:.1f},{:0>1d}, {:0>1d},'\
            ' {:0>1d} ] \n'.format(t_max, nmax, lmax, lmax))

    f.write('# Snlm, Tnlm \n')
    f.write('# r_s = {:.2f} \n'.format(r_s))
    f.write('# ICs from {} \n'.format(path))

    for i in range(len(S_flat)):
        f.write("{:.3f} {:.3f} \n".format(S_flat[i], T_flat[i]))

    f.close()

    # Write a file with the times between coefficients.
    f = open('../coefficients/times_' + file_name, 'w')

    f.write('# Times in Gyrs \n')
    for i in range(len(times)):
        f.write('{:.3f}\n'.format(times[i]))
    f.close()

def disk_bulge(path,  snap_name, N_initial):
    """
    Check if the there is a disk or a bulge in the simulation

    Parameters:
    -----------
    path : str
        path to simulation.

    snap_name : str
        snapshot base name.

    N_initial : int
        Number of the snapshot.

    Returns:
    --------

    bulge : int
        If no bulge particles it returns 0, otherwise it returns 1.
    disk : int
        If no disk particles it returns 0, otherwise it returns 1.

    """

    n_diskpart = readheader(path + snap_name + "_{:03d}"\
                            .format(N_initial), 'diskcount')
    n_bulgepart = readheader(path + snap_name + "_{:03d}"\
                             .format(N_initial), 'bulgecount')

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
                              N_final, Nmax, Lmax, r_s, Nmax_lmc=0,\
                              Lmax_lmc=0, r_s_lmc=3,\
                              disk=0, LMC=0, Nhalo=0):
    """
    Compute the coefficients from a series of snapshots of N-body
    simulations.
    Dependecies: pygadgetreader and octopus.

    Input:
    ------

    path : str
        path to snapshots
    snap_name : str
        snapshots base name
    N_intitial : int
        initial number of the snapshot.
    N_final : int
        final number of the snaphot in the simulation.
    Nmax : int
        Maximum number of terms in the radial terms of the BFE.
    Lmax:
        Maximum number of terms in the angular terms of the BFE.
    r_s: float
        Halo scale length in kpc.
    disk : int
        If disk is present disk==1, otherwise disk==0 (default).
    LMC : int
        Is the LMC present? Yes=1, No=0.
    Nhalo : int
        Number of MW DM particles.

    Returns:
    -------

    S : matrix of shape [t, nmax+1, lmax+1, lmax+1]
    T : matrix of shape [t, nmax+1, lmax+1, lmax+1]

    """

    # total time.
    t = N_final - N_initial

    # Snlm and Tnlm matrices initialization for the MW.
    S_mw = np.zeros((t, Nmax+1, Lmax+1, Lmax+1))
    T_mw = np.zeros((t, Nmax+1, Lmax+1, Lmax+1))

    # Lmax
    S_lmc = np.zeros((t, Nmax_lmc+1, Lmax_lmc+1, Lmax_lmc+1))
    T_lmc = np.zeros((t, Nmax_lmc+1, Lmax_lmc+1, Lmax_lmc+1))

    # computing the coefficients for all the snapshots.
    for i in range(N_initial, N_final, 1):
        pos = readsnap(path+snap_name+'_{:03d}'.format(i), 'pos', 'dm')
        mass = readsnap(path+snap_name+'_{:03d}'.format(i), 'mass', 'dm')
        pids = readsnap(path+snap_name+'_{:03d}'.format(i), 'pid', 'dm')

        # If the LMC is present:
        if (LMC==1):
            # selecting MW and LMC particles.

            pos_MW, mass_MW, pos_LMC, mass_LMC \
            = octopus.orbit_cm.MW_LMC_particles(pos, mass, pids, Nhalo)

        ## Computing the COM.

        if (disk==1):
            pos_disk = readsnap(path+snap_name+'_{:03d}'.format(i),\
                                'pos', 'disk')

            pot_disk = readsnap(path+snap_name+'_{:03d}'.format(i),\
                                'pot', 'disk')

            rcm, vcm = octopus.CM_disk_potential(pos_disk, pos_disk,\
                                                 pot_disk)
            # Computing the LMC COM.

            if (LMC==1):
                rlmc, vlmc = octopus.CM(pos_LMC, pos_LMC) # not using velocities!

        # If halo with no disk:
        elif ((disk==0) & (LMC==0)):
            rcm, vcm = octopus.CM(pos, pos) # not using velocities!

        # if halo with LMC halo:
        elif ((disk==0) & (LMC==1)):
            rcm, vcm = octopus.CM(pos_MW, pos_MW) # not using velocities!
            rlmc, vlmc = octopus.CM(pos_LMC, pos_LMC) # not using velocities!


        ## Centering the halos & computing the Snlm Tnlm coefficients.

        if (LMC==1):
            ## Centering halos
            pos_mw_cm = re_center_halo(pos_MW, rcm)
            pos_lmc_cm = re_center_halo(pos_LMC, rlmc)

            ## Compute Snlm and Tnlm for the MW halo particles.
            S_mw[i-N_initial], T_mw[i-N_initial]\
            = biff.compute_coeffs_discrete(np.ascontiguousarray(pos_mw_cm.astype(np.double)), mass_MW.astype(np.double)*1E10, Nmax, Lmax, r_s)

            ############################# change nmax and lmax lMC

            ## Compute Snlm and Tnlm for the LMC halo particles.
            S_lmc[i-N_initial], T_lmc[i-N_initial]\
            = biff.compute_coeffs_discrete(np.ascontiguousarray(pos_lmc_cm.astype(np.double)),\
                                           mass_LMC.astype(np.double)*1E10,\
                                           Nmax_lmc, Lmax_lmc, r_s_lmc)

        ## Without the LMC:

        elif (LMC==0):
            pos_cm = re_center_halo(pos, rcm)
            S_mw[i-N_initial], T_mw[i-N_initial] \
            = biff.compute_coeffs_discrete(np.ascontiguousarray(pos_cm.astype(np.double)), mass.astype(np.double)*1E10, Nmax, Lmax, r_s)
            # Computing Coefficients.

    if (LMC==0):
        return S_mw, T_mw

    elif (LMC==1):
        return S_mw, T_mw, S_lmc, T_lmc



if __name__ == "__main__":

    if(len(sys.argv)!=11):
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
        print('LMC')
        print('Nhalo')
        print('')
        print('////////////////////////////////////////////////////////////////')
        exit(0)

    ## Reading variables from argv!

    path = sys.argv[1]
    snap_name = sys.argv[2]

    init_snap = int(sys.argv[3])
    final_snap = int(sys.argv[4])

    nmax = int(sys.argv[5])
    lmax = int(sys.argv[6])
    r_s_mw = float(sys.argv[7])

    nmax_lmc = int(sys.argv[8])
    lmax_lmc = int(sys.argv[9])
    r_s_lmc = float(sys.argv[10])

    out_name = sys.argv[11]
    LMC = int(sys.argv[12])
    Nhalo = int(sys.argv[13])

    ## Total number of snapshots:

    N_snaps = final_snap - init_snap

    ## Time between snapshots:

    dt_nbody = snap_times_nbody(path, snap_name, init_snap)

    ## Total time between the first and the initial snapshot.

    times = np.linspace(init_snap*dt_nbody, final_snap*dt_nbody,\
                        N_snaps)

    ## Look for the presence of the disk or the bulge.
    ## this is done for the computation of the COM.

    disk, bulge = disk_bulge(path, snap_name, init_snap)

    print('disk, bulge', disk, bulge)



    ## Computing coefficients.

    print('Computing BFE coefficients')

    if (LMC==0):
        S_mw, T_mw = compute_coeffs_from_snaps(path, snap_name,\
                                               init_snap, final_snap,\
                                               nmax, lmax, r_s_mw,\
                                               disk=disk, LMC=LMC,\
                                               Nhalo=Nhalo)

        print('Writting coefficients')

        write_coefficients(S_mw, T_mw, times, 'MW'+out_name, N_snaps,\
                           nmax, lmax, r_s, path)

    if (LMC==1):
        ##  Computing coefficients.
        ## *****
        S_mw, T_mw, S_lmc, T_lmc = compute_coeffs_from_snaps(path,\
                                                             snap_name,\
                                                             init_snap,\
                                                             final_snap,\
                                                             nmax,lmax,\
                                                             r_s_mw,\
                                                             nmax_lmc,\
                                                             lmax_lmc,\
                                                             r_s_lmc,\
                                                             disk,\
                                                             LMC,\
                                                             Nhalo)

        ## writing the coefficients!

        print('Writing coefficients')
        write_coefficients(S_mw, T_mw, times, 'MW'+ out_name, N_snaps,\
                           nmax, lmax, r_s_mw, path)

        write_coefficients(S_lmc, T_lmc, times, 'LMC'+ out_name,\
                           N_snaps, nmax, lmax, r_s_lmc, path)

