"""
Code to compute the BFE coefficients from 
a N-body simulated halo.

Note: The halo have to be centered.

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
  is re-centered to the halo COM.

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


def interpolate_coeff(S, T, dt_nbody, dt_int, N_initial, N_final, nmax, lmax):
    """
    Interpolate the BFE coefficients.

    Parameters:
    -----------
        Snlm : float
            The value of the cosine expansion coefficient for the desired number of snapshots to be interpolated.
        Tnlm : float
            The value of the sine expansion coefficient for the desired number of snapshots to be interpolated.
        dt_nbody : float
            Time bet snapshot in the n-body simulation.
        dt_int : float
            Time
        N_initial : int
            Inital snapshot number.
        N_final : int
            Final snapshot number.
        nmax :
            Maximum value of ``n`` for the radial expansion.
        lmax :
            Maximum value of ``l`` for the spherical expansion.
    Returns:
    --------
        Snlm_interpolate : float
            The value of the cosine expansion coefficient interpolated for different dt values.
        Tnlm_interpolate : float
            The value of the sine expansion coefficient interpolated for different dt values.
    """

    # total time
    time = (N_final - N_initial) * dt_nbody

    # time arrays
    time_array = np.linspace(0, time, time/dt_nbody)
    time_array_new = np.linspace(0, time, time/dt_int)

    ## Coefficient Matrices size: [time, nmax+1, lmax+1, lmax+1] 
    S_new = np.zeros((int(time/dt_int), nmax+1, lmax+1, lmax+1))
    T_new = np.zeros((int(time/dt_int), nmax+1, lmax+1, lmax+1))
    # Interpolating the coefficients.
    for i in range(nmax+1):
        for j in range(lmax+1):
            for k in range(lmax+1):
                if k<=j:
                    # put the contrain k<j ? 
                    f = interpolate.interp1d(time_array, S[:,i,j,k])
                    S_new[:,i,j,k] = f(time_array_new)

    return S_new, T_new


def print_orbit(t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb, file_name):
    f = open(file_name, 'w')
    f.write('#t_orb (Gyrs), x (kpc), y(kpc), z(kpc), vx(km/s), vy(km/s), vz(km/s) \n')
    for i in range(len(t_orb)):
        print(i)
        f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} \n".format(t_orb[i], x_orb[i], y_orb[i], z_orb[i], vx_orb[i], vy_orb[i], vz_orb[i]))

    f.close()


if __name__ == "__main__":

    if(len(sys.argv)!=17):
        print('///////////////////////////////////////////////////////////////')
        print('')
        print('Usage:')
        print('------')
        print('x_init : cartesian initial x-coordinate of the test particle')
        print('y_init : cartesian initial y-coordinate of the test particle')
        print('z_init : cartesian initial z-coordinate of the test particle')
        print('vx_init : cartesian initial vx-coordinate of the test particle')
        print('vy_init : cartesian initial vy-coordinate of the test particle')
        print('vz_init : cartesian initial vz-coordinate of the test particle')
        print('Time of integration: total time for the orbit integration in Gyrs. e.g: 2')
        print('r_s : scale lenght of the halo')
        print('nmax')
        print('lmax')
        print('path : path to nbody simulations snapshots')
        print('snap_name: name of the snapshot base name')
        print('init_snap : number of the initial snapshot')
        print('final_snap : number of the final snapshot')
        print('interp_dt : interpolated dt time this is going to be the orbit dt')
        print('orbit name : name of the orbit output file')
        print('')
        print('////////////////////////////////////////////////////////////////')
        exit(0)

    x_init = float(sys.argv[1])
    y_init = float(sys.argv[2])
    z_init = float(sys.argv[3])
    vx_init = float(sys.argv[4])
    vy_init = float(sys.argv[5])
    vz_init = float(sys.argv[6])

    time = float(sys.argv[7])
    r_s = float(sys.argv[8])

    nmax = int(sys.argv[9])
    lmax = int(sys.argv[10])

    path = sys.argv[11]
    snap_name = sys.argv[12]

    init_snap = int(sys.argv[13])
    final_snap = int(sys.argv[14])

    interp_dt = float(sys.argv[15])
    orbit_name = sys.argv[16]


    M = 1
    G_c = constants.G
    G_c2 = G_c.to(u.kiloparsec**3 / (u.Gyr**2 * u.Msun))
    g_fact = 43007.1/(G_c2.value*1E10)

    # Computing coefficients.
    print('Computing BFE coefficients')
    S_nlm, T_nlm = compute_coeffs_from_snaps(path, snap_name, init_snap, final_snap, nmax, lmax, r_s)

    dt_nbody = snap_times_nbody(path, snap_name, init_snap)
    print('dt: ', dt_nbody)
    print('Interpolating coefficients')
    S_interp, T_interp = interpolate_coeff(S_nlm, T_nlm, dt_nbody, interp_dt, init_snap, final_snap, nmax, lmax)

    ## Integrating orbit in evolving potential.
    print('integrating orbit')
    t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb = leapfrog_bfe.integrate_biff_t(x_init, y_init, z_init, vx_init, vy_init, vz_init, time, S_interp, T_interp, G_c.value*g_fact, M, r_s, interp_dt)
    print(t_orb[0], x_orb[0])
    print_orbit(t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb, orbit_name)
    ## Integrating orbit in the potential of a given time -> Generalize this to any time.
    #t_orb_st, x_orb_st, y_orb_st, z_orb_st, vx_orb_st, vy_orb_st, vz_orb_biff_st = leapfrog_bfe.integrate_biff(x_init, y_init, z_init, vx_init, vy_init, vz_init, time, S_interp[0], T_interp[0], G_c.value*g_fact, M , r_s)


