## TODO: interpolate COM
import numpy as np
from scipy import interpolate


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
    time_array = np.linspace(0, time, time/dt_nbody+1)
    time_array_new = np.linspace(0, time, time/dt_int+2)

    ## Coefficient Matrices size: [time, nmax+1, lmax+1, lmax+1]
    S_new = np.zeros((int(time/dt_int)+2, nmax+1, lmax+1, lmax+1))
    T_new = np.zeros((int(time/dt_int)+2, nmax+1, lmax+1, lmax+1))
    # Interpolating the coefficients.
    for i in range(nmax+1):
        for j in range(lmax+1):
            for k in range(lmax+1):
                if k<=j:
                    # put the constrain k<j ?·
                    print(len(time_array), len(S[:,i,j,k]))
                    f = interpolate.interp1d(time_array, S[:,i,j,k])
                    S_new[:,i,j,k] = f(time_array_new)
                    f2 = interpolate.interp1d(time_array, T[:,i,j,k])
                    T_new[:,i,j,k] = f2(time_array_new)

    return S_new, T_new
