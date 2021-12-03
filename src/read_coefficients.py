import numpy as np
import coefficients_smoothing


def load_scf_coefficients(coeff_files, nmax, lmax, mmax, pmass, sn):
    """
    Load coefficients.
    ## this is the s/n sample to take the mean values of the coefficients!
    min_sample: n_min = 0
    max_sample: nmax = 1
    # pmass : particle mass
    # sn
    """

    #ncoeff_sample = max_sample - min_sample
# smoothing routines -------------------
    data = np.loadtxt(coeff_files)
    S = data[:,0]
    SS = data[:,1]
    T = data[:,2]
    TT = data[:,3]
    ST = data[:,4]
    S = coefficients_smoothing.reshape_matrix(S, nmax, lmax, lmax)
    SS = coefficients_smoothing.reshape_matrix(SS, nmax, lmax, lmax)
    T = coefficients_smoothing.reshape_matrix(T, nmax, lmax, lmax)
    TT = coefficients_smoothing.reshape_matrix(TT, nmax, lmax, lmax)
    ST = coefficients_smoothing.reshape_matrix(ST, nmax, lmax, lmax)

    S_smooth, T_smooth, N_smooth = coefficients_smoothing.smooth_coeff_matrix(S, T, SS, TT, ST,\
                                                                              pmass,\
                                                                              nmax,\
                                                                              lmax,\
                                                                              mmax,\
                                                                              sn,\
                                                                              sn_out=0)
    return S_smooth, T_smooth


def read_coefficients_smooth(coeff_files, pmass,
                             tmax, nmax, lmax, sn=4):
    """
    Function that reads the coefficients.
    """

    S, T = load_scf_coefficients(coeff_files, nmax, lmax, nmax, pmass, sn)

    S_nlm = S.reshape(tmax, nmax+1, lmax+1, lmax+1)
    T_nlm = T.reshape(tmax, nmax+1, lmax+1, lmax+1)

    return np.ascontiguousarray(S_nlm), np.ascontiguousarray(T_nlm)

def read_coeff_files_smooth(coeff_files, snap1, snap2, nmax, lmax, backwards=0):
    pmass = 1.577212515257997438e-06
    sn=4
    t = snap2-snap1+1
    print("total time = {}".format(int(t)))
    S_nlm_all = np.zeros((t, nmax+1, lmax+1, lmax+1))
    T_nlm_all = np.zeros((t, nmax+1, lmax+1, lmax+1))

    for i in range(snap1, snap2+1):
        S_nlm_all[i], T_nlm_all[i] = read_coefficients_smooth(coeff_files+'{:0>3d}.txt'.format(i), pmass, 1, nmax, lmax, sn)

    if backwards==1:
        return S_nlm_all[::-1], T_nlm_all[::-1]
    elif backwards==0:
        return S_nlm_all, T_nlm_all

# Done smoothing


def read_coefficients(path, tmax, nmax, lmax):
    """
    Function that reads the coefficients.
    """

    ST = np.loadtxt(path)

    S = ST[:,0]
    T = ST[:,2]

    S_nlm = S.reshape(tmax, nmax+1, lmax+1, lmax+1)
    T_nlm = T.reshape(tmax, nmax+1, lmax+1, lmax+1)
    print('Coefficients loades with shape: tmax={}, nmax={}, lmax={}'.(tmax, nmax, lmax)

    return np.ascontiguousarray(S_nlm), np.ascontiguousarray(T_nlm)


def read_coeff_files(path, snap1, snap2, tmax, nmax, lmax):
    t = snap2-snap1+1
    S_nlm_all = np.zeros((t, nmax+1, lmax+1, lmax+1))
    T_nlm_all = np.zeros((t, nmax+1, lmax+1, lmax+1))
    for i in range(snap1, snap2+1):
        S_nlm_all[i] = read_coefficients(path+'{:0>3d}.txt'.format(i), tmax, nmax, lmax)[0]
        T_nlm_all[i] = read_coefficients(path+'{:0>3d}.txt'.format(i), tmax, nmax, lmax)[1]
    return S_nlm_all, T_nlm_all
