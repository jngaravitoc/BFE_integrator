"""
Example of computing and orbit using BFE

"""

import numpy as np
from astropy import constants
from astropy import units as u
import sys
import leapfrog_bfe


def print_orbit(t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb,\
               file_name):

    f = open(file_name, 'w')
    f.write('#t_orb (Gyrs), x (kpc), y(kpc), z(kpc), vx(km/s),'\
            'vy(km/s), vz(km/s) \n')


    for i in range(len(t_orb)):
        f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} \n"\
                .format(t_orb[i], x_orb[i], y_orb[i], z_orb[i],\
                        vx_orb[i], vy_orb[i], vz_orb[i]))

    f.close()


if __name__ == "__main__":


    if(len(sys.argv)!=23):

        print('///////////////////////////////////////////////////////////////')
        print('')
        print('Usage:')
        print('------')
        print('x_init : cartesian initial x-coordinate of the test particle in kpc')
        print('y_init : cartesian initial y-coordinate of the test particle in kpc')
        print('z_init : cartesian initial z-coordinate of the test particle in kpc')
        print('vx_init : cartesian initial vx-coordinate of the test particle in km/s')
        print('vy_init : cartesian initial vy-coordinate of the test particle in km/s')
        print('vz_init : cartesian initial vz-coordinate of the test particle in km/s')
        print('Time of integration: total time for the orbit integration in Gyrs. e.g: 2')
        print('interp_dt : this is the time of the integration time step')
        print('static : No (0), yes (1)')
        print('r_s : scale lenght of the halo in kpc')
        print('nmax')
        print('lmax')
        print('path_coeff: path to the BFE coefficients')
        print('path_times: path to times')
        print('orbit name : name of the orbit output file')
        print('disk : 1 if disk, 0 if not')
        print('LMC : 1 if LMC, 0 if not')
        print('path_coeff_lmc : path to the LMC BFE coefficients')
        print('nmax_lmc : nmax for the lmc')
        print('lmax_lmc : lmax for the lmc')
        print('r_s_lmc : r_s for the lmc')
        print('backwards : 0 (no) 1 (yes)')
        print('')
        print('////////////////////////////////////////////////////////////////')
        exit(0)


    # defining variables:

    ## particles orbit initial conditions.
    x_init = float(sys.argv[1])
    y_init = float(sys.argv[2])
    z_init = float(sys.argv[3])
    vx_init = float(sys.argv[4])
    vy_init = float(sys.argv[5])
    vz_init = float(sys.argv[6])


    # Times:
    time = float(sys.argv[7])  #total time of integration?
    interp_dt = float(sys.argv[8])

    #
    static = int(sys.argv[9])
    r_s = float(sys.argv[10])

    # Size of BFE expansion.
    nmax  = int(sys.argv[11])
    lmax = int(sys.argv[12])

    # Paths to coefficients.
    path_coeff = sys.argv[13]
    path_times = sys.argv[14]

    orbit_name = sys.argv[15]
    disk = int(sys.argv[16])


    LMC = int(sys.argv[17])
    path_coeff_lmc = sys.argv[18]
    nmax_lmc = int(sys.argv[19])
    lmax_lmc = int(sys.argv[20])
    r_s_lmc = float(sys.argv[21])

    backwards = int(sys.argv[22])

    M = 1
    G_c = constants.G
    G_c = G_c.to(u.kiloparsec*u.km**2/ (u.s**2 * u.Msun))*1E10
    print(G_c)
    N_snaps=1

    if (static==0):
        #times_nbody = np.loadtxt(path_times)
        dt_nbody = 0.02#times_nbody[1] - times_nbody[0]
        N_snaps = 108 #len(times_nbody)
        t_nbody = 2.16#times_nbody[-1] - times_nbody[0]

        if (t_nbody < time):
             print('Integration time requested {:.2f} larger than the'\
                   'possible time {:.2f}'.format(time, t_nbody))
             exit(0)

    print(N_snaps)
    #S_nlm, T_nlm = read_coefficients(path_coeff, N_snaps, nmax, lmax)
    #print(np.shape(S_nlm), np.shape(T_nlm))
    #S_nlm, T_nlm = read_coeff_files(path_coeff, 0, 114,\
    #                                        N_snaps, nmax,\
    #                                        lmax)

    #S_nlm, T_nlm = read_coeff_files_smooth(path_coeff, 0, 108,\
    #                                       nmax,\
    #                                       lmax, backwards)
    # TODO: detect snap numbers automatically from the shape of the coefficients!

    S_nlm, T_nlm = read_coeff_files(path_coeff, 0, 108, tmax, nmax, lmax)


    if (static==0):
        print('Interpolating coefficients')
        S_interp, T_interp = interpolate_coeff(S_nlm, T_nlm, dt_nbody, interp_dt,
                                               t_nbody, nmax, lmax)


    ## Integrating orbit in time-evolving potential.

    #print('Integrating orbit')


    if (LMC==1):
        S, T = read_coeff_files(path_coeff_lmc, 0, 108,\
                                                N_snaps, nmax_lmc,\
                                                lmax_lmc)
        S_nlm_lmc = S/1E10
        T_nlm_lmc = T/1E10
        if (static==1):
            t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb\
            = leapfrog_bfe.integrate_biff(x_init, y_init, z_init,\
                                          vx_init, vy_init, vz_init,\
                                          time, S_nlm[0], T_nlm[0], \
                                          G_c.value, M, r_s,\
                                          interp_dt, disk, LMC=LMC,\
                                          Slmc=S_nlm_lmc[0],\
                                          Tlmc=T_nlm_lmc[0], x_lmc=-1,\
                                          y_lmc=-44, z_lmc=-28,\
                                          R_s_lmc = r_s_lmc)

        elif (static==0):
            S_interp_lmc, T_interp_lmc = interpolate_coeff(S_nlm_lmc,\
                                                           T_nlm_lmc,\
                                                           dt_nbody,\
                                                           interp_dt,\
                                                           t_nbody,\
                                                           nmax_lmc,\
                                                           lmax_lmc)


            t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb\
            = leapfrog_bfe.integrate_biff_t(x_init, y_init, z_init,\
                                            vx_init, vy_init, vz_init,\
                                            time, S_interp, T_interp,\
                                            G_c.value, 1, r_s,\
                                            interp_dt, disk, LMC=LMC,\
                                            Slmc=S_interp_lmc,\
                                            Tlmc=T_interp_lmc,\
                                            x_lmc=-1, y_lmc=-44,\
                                            z_lmc=-28, R_s_lmc\
                                            = r_s_lmc)

    elif (LMC==0):
        if (static==0):
            ### TODO : make inter_dt as an input parameter! this controls time
            ### TODO : directon
            print('Integrating orbit')
            t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb\
            = leapfrog_bfe.integrate_biff_t(x_init, y_init, z_init,\
                                            vx_init, vy_init, vz_init,\
                                            time, S_interp, T_interp,\
                                            G_c.value, M, r_s,\
                                            -interp_dt, disk, LMC=0)

        elif (static==1):
            print('Integrating orbit')
            t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb\
            = leapfrog_bfe.integrate_biff(x_init, y_init, z_init,\
                                           vx_init, vy_init, vz_init,\
                                           time, S_nlm[-1], T_nlm[-1],\
                                           G_c.value, M, r_s,\
                                           interp_dt, disk, LMC=0)

    print('Writing data')

    print_orbit(t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb,\
                orbit_name)
