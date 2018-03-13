import numpy as np
from astropy import units, constants
import biff
from soda.profiles import a_hernquist, a_mn


def extract(dct, namespace=None):
    # function that extracts variables from kwargs
    # from:
    # http://stackoverflow.com/questions/4357851/creating-or-assigning-variables-from-a-dictionary-in-python

    if not namespace: namespace = globals()
    namespace.update(dct)


def disk_bulge_a(x, y, z):
    a_bulge = a_hernquist(0.7, x, y, z, 1.4E10)
    a_disk = a_mn(0.638, 1.7, x, y, z, 5.78E10)
    ax = a_bulge[0] + a_disk[0]
    ay = a_bulge[1] + a_disk[1]
    az = a_bulge[2] + a_disk[2]
    return ax, ay, az

def integrate_hern(x_i, y_i, z_i, vx_i, vy_i, vz_i, time, Mass, R_s, disk=0):
    """
    Orbit integrator around a Hernquist potential using the
    leapfdrog algorithm.

    Input:
    ------
        x_i: float
            x component of the intial position of the test particle.
        y_i: float
            y component of the intial position of the test particle.
        z_i: float
            z component of the intial position of the test particle.
        vx_i: float
            vx component of the intial velocity of the test particle.
        vy_i: float
            vy component of the intial velocity of the test particle.
        vz_i: float
            vz component of the intial velocity of the test particle.
    """

    # h is the time step
    h = -0.001
    n_points = int(time * 1000.0)

    t = np.zeros(n_points)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)

    vx = np.zeros(n_points)
    vy = np.zeros(n_points)
    vz = np.zeros(n_points)

    ax = np.zeros(n_points)
    ay = np.zeros(n_points)
    az = np.zeros(n_points)

    t[0] = 0
    x[0] = x_i
    y[0] = y_i
    z[0] = z_i

    vx[0] = vx_i
    vy[0] = vy_i
    vz[0] = vz_i

    r = np.zeros((1,3))
    r[0] = np.array([x[0], y[0], z[0]])

    if (disk==1):
        ax[0] = a_hernquist(R_s, x[0], y[0], z[0], Mass)[0] + disk_bulge_a(x[0], y[0], z[0])[0]
        ay[0] = a_hernquist(R_s, x[0], y[0], z[0], Mass)[1] + disk_bulge_a(x[0], y[0], z[0])[1]
        az[0] = a_hernquist(R_s, x[0], y[0], z[0], Mass)[2] + disk_bulge_a(x[0], y[0], z[0])[2]

    elif (disk==0):
        ax[0] = a_hernquist(R_s, x[0], y[0], z[0], Mass)[0]
        ay[0] = a_hernquist(R_s, x[0], y[0], z[0], Mass)[1]
        az[0] = a_hernquist(R_s, x[0], y[0], z[0], Mass)[2]

    # half step
    # Here I assume the host galaxy starts at position (0, 0, 0) and then its
    # initial v[1] is (0, 0, 0)
    t[1] = t[0] - h
    x[1] = x[0] - h * vx[0]
    y[1] = y[0] - h * vy[0]
    z[1] = z[0] - h * vz[0]

    vx[1] = vx[0] - h * ax[0]
    vy[1] = vy[0] - h * ay[0]
    vz[1] = vz[0] - h * az[0]

    if (disk==1):
        ax[1] = a_hernquist(R_s, x[1], y[1], z[1], Mass)[0] + disk_bulge_a(x[1], y[1], z[1])[0]
        ay[1] = a_hernquist(R_s, x[1], y[1], z[1], Mass)[1] + disk_bulge_a(x[1], y[1], z[1])[1]
        az[1] = a_hernquist(R_s, x[1], y[1], z[1], Mass)[2] + disk_bulge_a(x[1], y[1], z[1])[2]

    if (disk==0):
        ax[1] = a_hernquist(R_s, x[1], y[1], z[1], Mass)[0]
        ay[1] = a_hernquist(R_s, x[1], y[1], z[1], Mass)[1]
        az[1] = a_hernquist(R_s, x[1], y[1], z[1], Mass)[2]

    for i in range(2, len(x)):
        t[i] = t[i-1] - h
        x[i] = x[i-2] - 2 * h * vx[i-1]
        y[i] = y[i-2] - 2 * h * vy[i-1]
        z[i] = z[i-2] - 2 * h * vz[i-1]

        vx[i] = vx[i-2] - 2 * h * ax[i-1]
        vy[i] = vy[i-2] - 2 * h * ay[i-1]
        vz[i] = vz[i-2] - 2 * h * az[i-1]

        r = np.zeros((1,3))
        r[0] = np.array([x[i], y[i], z[i]])

        if (disk==1):
            ax[i] = a_hernquist(R_s, x[i], y[i], z[i], Mass)[0] + disk_bulge_a(x[i], y[i], z[i])[0]
            ay[i] = a_hernquist(R_s, x[i], y[i], z[i], Mass)[1] + disk_bulge_a(x[i], y[i], z[i])[1]
            az[i] = a_hernquist(R_s, x[i], y[i], z[i], Mass)[2] + disk_bulge_a(x[i], y[i], z[i])[2]

        if (disk==0):
            ax[i] = a_hernquist(R_s, x[i], y[i], z[i], Mass)[0]
            ay[i] = a_hernquist(R_s, x[i], y[i], z[i], Mass)[1]
            az[i] = a_hernquist(R_s, x[i], y[i], z[i], Mass)[2]

    return t, x, y, z, vx, vy, vz

def integrate_biff(x_i, y_i, z_i, vx_i, vy_i, vz_i, time, S, T, G, Mass, R_s, dt, disk=0, **kwargs):
    """
    Orbit integration function for the BFE methods.
    the time evolution uses a leapfrog algorithm.
    Accelerations from the coefficients are computed with biff.

    Parameters:
    -----------
    x_i : initial particle coordinate in kpc.
    x_i : initial particle coordinate in kpc.
    x_i : initial particle coordinate in kpc.
    vx_i : initial particle coordinate in km/s.
    vy_i : initial particle coordinate in km/s.
    vz_i : initial particle coordinate in km/s.
    time : total time of integration.
    S : Matrix with the coefficients Snlm
    T : Matrix with the coefficients Tnlm
    G : gravitational constant in units of kpc/s
    Mass : 
    R_s : Dark Matter halo scale lenght
    dt : time step between integration points.
    disk : if a disk is present disk=1, default disk=0

    kwargs:
    -------
    LMC : 1, 0
    Slmc
    Tlmc
    x_lmc
    y_lmc
    z_lmc
    R_s_lmc

    Returns:
    --------

    to - do:
    --------
    1. Backwards integration.
    2. Combine both the orbit integration functions, the time
       dependent and the time independent.
    3. Include the LMC

    """
    ## put h as an input parameter
    # h is the time step
    h = -dt
    n_points = int(time / np.abs(h))

    #from kpc/gyrs to km/s

    #kpc_s2 = 1 * units.kpc / units.s**2
    #convtokpc_gyr2 = kpc_s2.to(units.kpc/units.Gyr**2)

    t = np.zeros(n_points)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)

    vx = np.zeros(n_points)
    vy = np.zeros(n_points)
    vz = np.zeros(n_points)

    ax = np.zeros(n_points)
    ay = np.zeros(n_points)
    az = np.zeros(n_points)

    t[0] = 0
    x[0] = x_i
    y[0] = y_i
    z[0] = z_i

    vx[0] = vx_i
    vy[0] = vy_i
    vz[0] = vz_i

    r = np.zeros((1,3))
    r[0] = np.array([x[0], y[0], z[0]])

    extract(kwargs)
    #print(np.shape(Slmc))
    #print(x_lmc)
    #print(R_s_lmc)

    if (disk==0):
        ax[0] = -biff.gradient(r, S, T, G, Mass, R_s)[0][0]
        ay[0] = -biff.gradient(r, S, T, G, Mass, R_s)[0][1]
        az[0] = -biff.gradient(r, S, T, G, Mass, R_s)[0][2]
    if (disk==1):
        ax[0] = -biff.gradient(r, S, T, G, Mass, R_s)[0][0] + disk_bulge_a(x[0], y[0], z[0])[0]
        ay[0] = -biff.gradient(r, S, T, G, Mass, R_s)[0][1] + disk_bulge_a(x[0], y[0], z[0])[1]
        az[0] = -biff.gradient(r, S, T, G, Mass, R_s)[0][2] + disk_bulge_a(x[0], y[0], z[0])[2]


    if (LMC==1):
        r_lmc = np.zeros((1,3))
        r_lmc[0] = np.array([x[0]-x_lmc, y[0]-y_lmc, z[0]-z_lmc])
        ax_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][0]
        ay_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][1]
        az_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][2]
        ax[0] += ax_lmc
        ay[0] += ay_lmc
        az[0] += az_lmc

    # half step
    # Here I assume the host galaxy starts at position (0, 0, 0) and then its
    # initial v[1] is (0, 0, 0)
    t[1] = t[0] - h
    x[1] = x[0] - h * vx[0]
    y[1] = y[0] - h * vy[0]
    z[1] = z[0] - h * vz[0]

    vx[1] = vx[0] - h * ax[0]
    vy[1] = vy[0] - h * ay[0]
    vz[1] = vz[0] - h * az[0]

    r[0] = np.array([x[1], y[1], z[1]])

    if (disk==0):
        ax[1] = -biff.gradient(r, S, T, G, Mass, R_s)[0][0]
        ay[1] = -biff.gradient(r, S, T, G, Mass, R_s)[0][1]
        az[1] = -biff.gradient(r, S, T, G, Mass, R_s)[0][2]

    if (disk==1):
        ax[1] = -biff.gradient(r, S, T, G, Mass, R_s)[0][0] + disk_bulge_a(x[1], y[1], z[1])[0]
        ay[1] = -biff.gradient(r, S, T, G, Mass, R_s)[0][1] + disk_bulge_a(x[1], y[1], z[1])[1]
        az[1] = -biff.gradient(r, S, T, G, Mass, R_s)[0][2] + disk_bulge_a(x[1], y[1], z[1])[2]

    if (LMC==1):
        r_lmc[0] = np.array([x[1]-x_lmc, y[1]-y_lmc, z[1]-z_lmc])
        ax_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][0]
        ay_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][1]
        az_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][2]
        ax[1] += ax_lmc
        ay[1] += ay_lmc
        az[1] += az_lmc

    for i in range(2, len(x)):
        t[i] = t[i-1] - h
        x[i] = x[i-2] - 2 * h * vx[i-1]
        y[i] = y[i-2] - 2 * h * vy[i-1]
        z[i] = z[i-2] - 2 * h * vz[i-1]

        vx[i] = vx[i-2] - 2 * h * ax[i-1]
        vy[i] = vy[i-2] - 2 * h * ay[i-1]
        vz[i] = vz[i-2] - 2 * h * az[i-1]

        r = np.zeros((1,3))
        r[0] = np.array([x[i], y[i], z[i]])

        if (disk==0):
            ax[i] = -biff.gradient(r, S, T, G, Mass,R_s)[0][0]
            ay[i] = -biff.gradient(r, S, T, G, Mass,R_s)[0][1]
            az[i] = -biff.gradient(r, S, T, G, Mass,R_s)[0][2]
            print(ax[i])
        if (disk==1):
            ax[i] = -biff.gradient(r, S, T, G, Mass,R_s)[0][0] + disk_bulge_a(x[i], y[i], z[i])[0]
            ay[i] = -biff.gradient(r, S, T, G, Mass,R_s)[0][1] + disk_bulge_a(x[i], y[i], z[i])[1]
            az[i] = -biff.gradient(r, S, T, G, Mass,R_s)[0][2] + disk_bulge_a(x[i], y[i], z[i])[2]

        if (LMC==1):
            r_lmc[0] = np.array([x[i]-x_lmc, y[i]-y_lmc, z[i]-z_lmc])
            ax_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][0]
            ay_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][1]
            az_lmc = -biff.gradient(r_lmc, Slmc, Tlmc, G, Mass, R_s_lmc)[0][2]
            ax[i] += ax_lmc
            ay[i] += ay_lmc
            az[i] += az_lmc

    return t, x, y, z, vx, vy, vz




def integrate_biff_t(x_i, y_i, z_i, vx_i, vy_i, vz_i, time, S, T, G,\
                     Mass, R_s, dt, disk=0, **kwargs):
    """
    Function that computes
    """
    h = -dt
    n_points = int(time * 1/abs(h))

    #from kpc/gyrs to km/s
    #convtokms = 1 * units.kpc / units.Gyr
    #convtokms = convtokms.to(units.km/units.s)

    #kpc_s2 = 1 * units.kpc / units.s**2
    #convtokpc_gyr2 = kpc_s2.to(units.kpc/units.Gyr**2)

    t = np.zeros(n_points)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)

    vx = np.zeros(n_points)
    vy = np.zeros(n_points)
    vz = np.zeros(n_points)

    ax = np.zeros(n_points)
    ay = np.zeros(n_points)
    az = np.zeros(n_points)

    t[0] = 0
    x[0] = x_i
    y[0] = y_i
    z[0] = z_i

    vx[0] = vx_i
    vy[0] = vy_i
    vz[0] = vz_i

    r = np.zeros((1,3))
    r[0] = np.array([x[0], y[0], z[0]])

    extract(kwargs)

    if (disk==0):
        ax[0] = -biff.gradient(r, S[0], T[0], G, Mass, R_s)[0][0]
        ay[0] = -biff.gradient(r, S[0], T[0], G, Mass, R_s)[0][1]
        az[0] = -biff.gradient(r, S[0], T[0], G, Mass, R_s)[0][2]

    if (disk==1):
        ax[0] = -biff.gradient(r, S[0], T[0], G, Mass, R_s)[0][0]\
                + disk_bulge_a(x[0], y[0], z[0])[0]

        ay[0] = -biff.gradient(r, S[0], T[0], G, Mass, R_s)[0][1]\
                + disk_bulge_a(x[0], y[0], z[0])[1]
        az[0] = -biff.gradient(r, S[0], T[0], G, Mass, R_s)[0][2]\
                + disk_bulge_a(x[0], y[0], z[0])[2]
    # half step
    # Here I assume the host galaxy starts at position (0, 0, 0) and then its
    # initial v[1] is (0, 0, 0)
    if (LMC==1):
        r_lmc = np.zeros((1,3))
        r_lmc[0] = np.array([x[0]-x_lmc, y[0]-y_lmc, z[0]-z_lmc])
        ax_lmc = -biff.gradient(r_lmc, Slmc[0], Tlmc[0], G, Mass, \
                                R_s_lmc)[0][0]

        ay_lmc = -biff.gradient(r_lmc, Slmc[0], Tlmc[0], G, Mass, \
                                R_s_lmc)[0][1]

        az_lmc = -biff.gradient(r_lmc, Slmc[0], Tlmc[0], G, Mass, \
                                R_s_lmc)[0][2]
        ax[0] += ax_lmc
        ay[0] += ay_lmc
        az[0] += az_lmc

    t[1] = t[0] - h
    x[1] = x[0] - h * vx[0]
    y[1] = y[0] - h * vy[0]
    z[1] = z[0] - h * vz[0]

    vx[1] = vx[0] - h * ax[0]
    vy[1] = vy[0] - h * ay[0]
    vz[1] = vz[0] - h * az[0]

    r[0] = np.array([x[1], y[1], z[1]])

    if (disk==0):

        ax[1] = -biff.gradient(r, S[1], T[1], G, Mass, R_s)[0][0]
        ay[1] = -biff.gradient(r, S[1], T[1], G, Mass, R_s)[0][1]
        az[1] = -biff.gradient(r, S[1], T[1], G, Mass, R_s)[0][2]

    if (disk==1):

        ax[1] = -biff.gradient(r, S[1], T[1], G, Mass, R_s)[0][0]\
                + disk_bulge_a(x[1], y[1], z[1])[0]
        ay[1] = -biff.gradient(r, S[1], T[1], G, Mass, R_s)[0][1]\
                + disk_bulge_a(x[1], y[1],  z[1])[1]
        az[1] = -biff.gradient(r, S[1], T[1], G, Mass, R_s)[0][2]\
                + disk_bulge_a(x[1], y[1], z[1])[2]


    if (LMC==1):
        r_lmc = np.zeros((1,3))
        r_lmc[0] = np.array([x[1]-x_lmc, y[1]-y_lmc, z[1]-z_lmc])
        ax_lmc = -biff.gradient(r_lmc, Slmc[1], Tlmc[1], G, Mass, \
                                R_s_lmc)[0][0]

        ay_lmc = -biff.gradient(r_lmc, Slmc[1], Tlmc[1], G, Mass, \
                                R_s_lmc)[0][1]

        az_lmc = -biff.gradient(r_lmc, Slmc[1], Tlmc[1], G, Mass, \
                                R_s_lmc)[0][2]
        ax[1] += ax_lmc
        ay[1] += ay_lmc
        az[1] += az_lmc


    for i in range(2, n_points):
        t[i] = t[i-1] - h
        x[i] = x[i-2] - 2 * h * vx[i-1]
        y[i] = y[i-2] - 2 * h * vy[i-1]
        z[i] = z[i-2] - 2 * h * vz[i-1]

        vx[i] = vx[i-2] - 2 * h * ax[i-1]
        vy[i] = vy[i-2] - 2 * h * ay[i-1]
        vz[i] = vz[i-2] - 2 * h * az[i-1]

        r[0] = np.array([x[i], y[i], z[i]])

        if (disk==0):
            ax[i] = -biff.gradient(r, S[i], T[i], G, Mass, R_s)[0][0]
            ay[i] = -biff.gradient(r, S[i], T[i], G, Mass, R_s)[0][1]
            az[i] = -biff.gradient(r, S[i], T[i], G, Mass, R_s)[0][2]

        if (disk==1):
            ax[i] = -biff.gradient(r, S[i], T[i], G, Mass, R_s)[0][0]\
                     + disk_bulge_a(x[i], y[i], z[i])[0]

            ay[i] = -biff.gradient(r, S[i], T[i], G, Mass, R_s)[0][1]\
                    + disk_bulge_a(x[i], y[i], z[i])[1]

            az[i] = -biff.gradient(r, S[i], T[i], G, Mass, R_s)[0][2]\
                    + disk_bulge_a(x[i], y[i], z[i])[2]

        if (LMC==1):
            r_lmc = np.zeros((1,3))
            r_lmc[0] = np.array([x[1]-x_lmc, y[i]-y_lmc, z[i]-z_lmc])
            ax_lmc = -biff.gradient(r_lmc, Slmc[i], Tlmc[i], G, Mass, \
                                    R_s_lmc)[0][0]

            ay_lmc = -biff.gradient(r_lmc, Slmc[i], Tlmc[i], G, Mass, \
                                    R_s_lmc)[0][1]

            az_lmc = -biff.gradient(r_lmc, Slmc[i], Tlmc[i], G, Mass, \
                                    R_s_lmc)[0][2]
            ax[i] += ax_lmc
            ay[i] += ay_lmc
            az[i] += az_lmc

    return t, x, y, z, vx, vy, vz
