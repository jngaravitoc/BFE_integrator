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

Usage:
------

python bfe_coeff_parallel.py


path to n-body:
snap_name
init_snap
final_snap
nmax
lmax
r_s_mw
nmax_lmc
lmax_lmc
r_s_lmc
out_name
LMC
N_halo
ncores
-mpi

Important:
----------

If comparison wants to be done with Gadget orbits
please use the gravitational constant G of Gadget
G=43007.1 km^2 * kpc/(s^2 Msun * 1E10).
This is what the code uses as default
see: https://github.com/jngaravitoc/MW_anisotropy/blob/master/code/equilibrium/G_units_gadget.ipynb


TODO:
- free memory
- Truncate halo

"""

import numpy as np
import biff
from scipy import interpolate
from pygadgetreader import *
from astropy import constants
from astropy import units as u
#import sys
import octopus
import leapfrog_bfe
#rom multiprocessing import Pool
from os.path import join
import schwimmbad



## Code main functions:

def re_center_halo(pos, r_cm):
    """
    Re-center halo positions or velocities.
    """

    for i in range(3):
        pos[:,i] = pos[:,i] - r_cm[i]
    return pos

def snap_times_nbody(path, snap_name, N_initial):
    """
    Compute the times between snapshots of the n-body simulation.

    """
    for i in range(N_initial, N_initial+1):
        print('here\n')
        dt = readheader(path+snap_name+'_{:03d}'.format(i+1), 'time')\
             - readheader(path+snap_name+'_{:03d}'.format(i), 'time')
        return dt

def write_coefficients(S, T, file_name, nmax, lmax, r_s, n):
    """
    Writes the coefficients Snlm and Tnlm in a file.
    for this the coefficients are flattened into a 1d array.
    Parameters:
    -----------
    S : Matrix of coefficients Snlm.
    T : Matrix of coefficients Tnlm.

    to-do:
    add LMC and times.
    """
    print('writing snapshot {:d}'.format(n))

    # from 3d to 1d coefficients.

    S_flat = S.flatten()
    T_flat = T.flatten()


    f = open('ST_'+file_name+'{:d}'.format(n), 'w')
    #f.write('# number of time steps : {:.3f} \n'.format(t_max))
    f.write('# nmax = {:0>2d}, lmax = {:0>2d} \n'.format(nmax, lmax))
    f.write('# original matrix shape [{:.1f},{:0>1d}, {:0>1d},'\
            ' {:0>1d} ] \n'.format(n, nmax, lmax, lmax))

    f.write('# Snlm, Tnlm \n')
    f.write('# r_s = {:.2f} \n'.format(r_s))
    #f.write('# ICs from {} \n'.format(path))

    for i in range(len(S_flat)):
        f.write("{:.3f} {:.3f} \n".format(S_flat[i], T_flat[i]))

    f.close()
    print('finished writing snapshot {:d}'.format(n))

    # Write a file with the times between coefficients.
    #f = open('times_' + file_name, 'w')

    #f.write('# Times in Gyrs \n')
    #for i in range(len(times)):
        #f.write('{:.3f}\n'.format(times[i]))
    #f.close()

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

class Worker(object):

    def __init__(self, path, snap_name, init_snap, final_snap, Nmax,
                 Lmax, r_s_mw, Nmax_lmc, Lmax_lmc, r_s_lmc,  out_name,  LMC, Nhalo):
        self.path = path
        self.snap_name = snap_name
        self.init_snap = init_snap
        self.final_snap = final_snap
        self.Nmax = Nmax
        self.Lmax = Lmax
        self.r_s_mw = r_s_mw
        self.Nmax_lmc = Nmax_lmc
        self.Lmax_lmc = Lmax_lmc
        self.r_s_lmc = r_s_lmc
        self.out_name = out_name
        self.LMC = LMC
        self.Nhalo = Nhalo
        self.disk = 1


    def compute_coeffs_from_snaps(self, n):
        print('n=', n)
        pos = readsnap(self.path+self.snap_name+'_{:03d}'.format(n), 'pos', 'dm')
        mass = readsnap(self.path+self.snap_name+'_{:03d}'.format(n), 'mass', 'dm')
        pids = readsnap(self.path+self.snap_name+'_{:03d}'.format(n), 'pid', 'dm')

        # If the LMC is present:
        if (self.LMC==1):
        # selecting MW and LMC particles.
            pos_MW, mass_MW, pos_LMC, mass_LMC \
            = octopus.orbit_cm.MW_LMC_particles(pos, mass, pids, self.Nhalo)

            ## Computing the COM.

        if (self.disk==1):
            pos_disk = readsnap(self.path+self.snap_name+'_{:03d}'.format(n),\
                                                        'pos', 'disk')

            rcm, vcm = octopus.CM(pos_disk, pos_disk)
            # Computing the LMC COM.

            if (self.LMC==1):
                rlmc, vlmc = octopus.CM(pos_LMC, pos_LMC) # not using velocities!

                # If halo with no disk:
        elif ((self.disk==0) & (self.LMC==0)):
            rcm, vcm = octopus.CM(pos, pos) # not using velocities!

        # if halo with LMC halo:
        elif ((self.disk==0) & (self.LMC==1)):
            rcm, vcm = octopus.CM(pos_MW, pos_MW) # not using velocities!
            rlmc, vlmc = octopus.CM(pos_LMC, pos_LMC) # not using velocities!


        ## Centering the halos & computing the Snlm Tnlm coefficients.

        if (self.LMC==1):
            ## Centering halos
            pos_mw_cm = np.ascontiguousarray(re_center_halo(pos_MW, rcm))
            pos_lmc_cm = np.ascontiguousarray(re_center_halo(pos_LMC, rlmc))

            ## Compute Snlm and Tnlm for the MW halo particles.

            S_mw, T_mw = biff.compute_coeffs_discrete(pos_mw_cm.astype(np.double),
                                                      mass_MW.astype(np.double)*1E10, self.Nmax, Lmax,
                                                      self.r_s_mw)
            ############################# change nmax and lmax lMC
            ## Compute Snlm and Tnlm for the LMC halo particles.
            S_lmc, T_lmc = biff.compute_coeffs_discrete(pos_lmc_cm.astype(np.double),\
                                                                          mass_LMC.astype(np.double)*1E10,\
                                                                          self.Nmax_lmc,
                                                                          self.Lmax_lmc,
                                                                          self.r_s_lmc)

        elif (self.LMC==0):
            pos_cm = np.ascontiguousarray(re_center_halo(pos, rcm))
            S_mw, T_mw = biff.compute_coeffs_discrete(pos_cm.astype(np.double),\
                                                      mass.astype(np.double)*1E10,\
                                                      self.Nmax,
                                                      self.Lmax,
                                                      self.r_s_mw)

        print("Done computing coefficients from snapshot {:d}\n".format(n))
        #if fail:
        #   return None

        if (self.LMC==0):
            return n, S_mw, T_mw

        elif (self.LMC==1):
            return n, S_mw, T_mw, S_lmc, T_lmc

    @classmethod

    def callback(self, result):

        if result is not None:
            print('here in writing')
            n, Snlm, Tnlm = result
            write_coefficients(Snlm, Tnlm, self.out_name,
                               self.Nmax, self.Lmax, self.r_s_mw, n)
            # write to file
            #output_file = # something
            print(n)
            # write to "output_file"

    def __call__(self, task):
        print('hello')
        print(task)
        self.compute_coeffs_from_snaps(task)
        self.callback(task)

def main(pool, path, snap_name, init_snap, final_snap, nmax, lmax,
        r_s_mw, nmax_lmc, lmax_lmc, r_s_lmc, out_name, LMC, Nhalo):

    """
    Function to parallelize

    """
    ##############################################################
    # Reading arguments.

    # total number of snapshots
    N_snaps = final_snap - init_snap

    ## Time between snapshots:
    dt_nbody = snap_times_nbody(path, snap_name, init_snap)

    ## Total time between the first and the initial snapshot.

    times = np.linspace(init_snap*dt_nbody, final_snap*dt_nbody,\
                        N_snaps)


    disk, bulge = disk_bulge(path, snap_name, init_snap)

    print('disk, bulge', disk, bulge)


    ###############################################################

    worker = Worker(path, snap_name, init_snap, final_snap, nmax,
                    lmax, r_s_mw, nmax_lmc, lmax_lmc, r_s_lmc,
                    out_name, LMC, Nhalo)

    tasks = list(range(N_snaps+init_snap))
    print('tasks', tasks)
    for r in pool.map(worker, tasks, callback=Worker.callback):
        pass

    pool.close()


if __name__ == "__main__":

    """
    if(len(sys.argv)!=14):
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
     """
        ## writing the coefficients!

        #print('Writing coefficients')
        #write_coefficients(S_mw, T_mw, times, 'MW'+ out_name, N_snaps,\
        #                   nmax, lmax, r_s_mw, path)
        #
        #write_coefficients(S_lmc, T_lmc, times, 'LMC'+ out_name,\
        #                   N_snaps, nmax_lmc, lmax_lmc, r_s_lmc, path)
    from argparse import ArgumentParser
    parser = ArgumentParser(description="")

    parser.add_argument("--path", dest="path", default=1,
                       type=str, help="path to n-body simulations.")

    parser.add_argument("--snap_name", dest="snap_name", default=1,
                       type=str, help="path to n-body simulations.")

    parser.add_argument("--init_snap", dest="init_snap", default=1,
                       type=int, help="path to n-body simulations.")

    parser.add_argument("--final_snap", dest="final_snap", default=1,
                       type=int, help="path to n-body simulations.")

    parser.add_argument("--nmax", dest="nmax", default=1,
                       type=int, help="path to n-body simulations.")

    parser.add_argument("--lmax", dest="lmax", default=1,
                       type=int, help="path to n-body simulations.")

    parser.add_argument("--r_s_mw", dest="r_s_mw", default=1,
                       type=float, help="path to n-body simulations.")

    parser.add_argument("--nmax_lmc", dest="nmax_lmc", default=0,
                       type=int, help="path to n-body simulations.")

    parser.add_argument("--lmax_lmc", dest="lmax_lmc", default=0,
                       type=int, help="path to n-body simulations.")

    parser.add_argument("--r_s_lmc", dest="r_s_lmc", default=0,
                       type=float, help="path to n-body simulations.")

    parser.add_argument("--out_name", dest="out_name", default='coefficients',
                       type=str, help="path to n-body simulations.")

    parser.add_argument("--LMC", dest="LMC", default=0,
                       type=int, help="path to n-body simulations.")

    parser.add_argument("--Nhalo", dest="Nhalo", default=0,
                       type=int, help="path to n-body simulations.")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                        action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    kw = vars(args)
    kw.pop('mpi')
    kw.pop('n_cores')

    main(pool, **kw)
