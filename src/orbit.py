import numpy as np
import biff
from scipy import interpolate
from astropy import constants
from astropy import units as u
import sys
import leapfrog_bfe


def interpolate_coeff(S, T, dt_nbody, dt_int, time, nmax, lmax):
    """
    Interpolate the BFE coefficients.

    Parameters:
    -----------
        Snlm : float
            The value of the cosine expansion coefficient for the
            desired number of snapshots to be interpolated.
        Tnlm : float
            The value of the sine expansion coefficient for the
            desired number of snapshots to be interpolated.
        dt_nbody : float
            Time bet snapshot in the n-body simulation.
        dt_int : float
            dt for the integration.
        time: float
            total time covered by the coefficients.
        nmax :
            Maximum value of ``n`` for the radial expansion.
        lmax :
            Maximum value of ``l`` for the spherical expansion.

    Returns:
    --------
        Snlm_interpolate : float
            The value of the cosine expansion coefficient interpolated
            for different dt values.
        Tnlm_interpolate : float
            The value of the sine expansion coefficient interpolated
            for different dt values.
    """

    # time arrays
    print(time, dt_nbody)
    time_array = np.linspace(0, time, time/dt_nbody+2)
    time_array_new = np.linspace(0, time, time/dt_int+2)

    ## Coefficient Matrices size: [time, nmax+1, lmax+1, lmax+1]
    S_new = np.zeros((int(time/dt_int)+2, nmax+1, lmax+1, lmax+1))
    T_new = np.zeros((int(time/dt_int)+2, nmax+1, lmax+1, lmax+1))
    # Interpolating the coefficients.
    for i in range(nmax+1):
        for j in range(lmax+1):
            for k in range(lmax+1):
                if k<=j:
                    # put the contrain k<j ?·
                    print(len(time_array), len(S[:,i,j,k]))
                    f = interpolate.interp1d(time_array, S[:,i,j,k])
                    S_new[:,i,j,k] = f(time_array_new)

    return S_new, T_new


def read_coefficients(path, tmax, nmax, lmax):

    ST = np.loadtxt(path)

    S = ST[:,0]
    T = ST[:,1]

    S_nlm = S.reshape(tmax, nmax+1, lmax+1, lmax+1)
    T_nlm = S.reshape(tmax, nmax+1, lmax+1, lmax+1)

    return np.ascontiguousarray(S_nlm), np.ascontiguousarray(T_nlm)


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


    if(len(sys.argv)!=22):

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
    interp_dt = float(sys.argv[8])
    static = int(sys.argv[9])
    r_s = float(sys.argv[10])
    nmax  = int(sys.argv[11])
    lmax = int(sys.argv[12])

    path_coeff = sys.argv[13]
    path_times = sys.argv[14]
    orbit_name = sys.argv[15]
    disk = int(sys.argv[16])

    LMC = int(sys.argv[17])
    path_coeff_lmc = sys.argv[18]
    nmax_lmc = int(sys.argv[19])
    lmax_lmc = int(sys.argv[20])
    r_s_lmc = float(sys.argv[21])

    M = 1
    G_c = constants.G
    G_c = G_c.to(u.kiloparsec**3 / (u.s**2 * u.Msun))
    G_c2 = G_c.to(u.kiloparsec**3 / (u.Gyr**2 * u.Msun))
    g_fact = 43007.1/(G_c2.value*1E10)

    N_snaps=1

    if (static==0):
        times_nbody = np.loadtxt(path_times)
        dt_nbody = times_nbody[1] - times_nbody[0]
        N_snaps = len(times_nbody)
        t_nbody = times_nbody[-1] - times_nbody[0]

        if (t_nbody < time):
             print('Integration time requested {:.2f} larger than the'\
                   'possible time {:.2f}'.format(time, t_nbody))
             exit(0)

    print(N_snaps)
    S_nlm, T_nlm = read_coefficients(path_coeff, N_snaps, nmax, lmax)
    print(np.shape(S_nlm), np.shape(T_nlm))

    if (static==0):
        print('Interpolating coefficients')
        S_interp, T_interp = interpolate_coeff(S_nlm, T_nlm, dt_nbody,\
                                               interp_dt, t_nbody,\
                                               nmax, lmax)
    ## Integrating orbit in time-evolving potential.

    #print('Integrating orbit')


    if (LMC==1):
        S_nlm_lmc, T_nlm_lmc = read_coefficients(path_coeff_lmc,\
                                                 N_snaps, nmax_lmc,\
                                                 lmax_lmc)

        if (static==1):
            t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb\
            = leapfrog_bfe.integrate_biff(x_init, y_init, z_init,\
                                          vx_init, vy_init, vz_init,\
                                          time, S_nlm[0], T_nlm[0], \
                                          G_c.value*g_fact, M, r_s,\
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
                                            G_c.value*g_fact, M, r_s,\
                                            interp_dt, disk, LMC=LMC,\
                                            Slmc=S_interp_lmc,\
                                            Tlmc=T_interp_lmc,\
                                            x_lmc=-1, y_lmc=-44,\
                                            z_lmc=-28, R_s_lmc\
                                            = r_s_lmc)

    elif (LMC==0):
        if (static==0):
            print('Integrating orbit')
            t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb\
            = leapfrog_bfe.integrate_biff_t(x_init, y_init, z_init,\
                                            vx_init, vy_init, vz_init,\
                                            time, S_interp, T_interp,\
                                            G_c.value*g_fact, M, r_s,\
                                            interp_dt, disk)

        elif (static==1):
            print('Integrating orbit')
            t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb\
            = leapfrog_bfe.integrate_biff_t(x_init, y_init, z_init,\
                                            vx_init, vy_init, vz_init,\
                                            time, S_nlm, T_nlm,\
                                            G_c.value*g_fact, M, r_s,\
                                            interp_dt, disk)

    print('Writing data')

    print_orbit(t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb,\
                orbit_name)




