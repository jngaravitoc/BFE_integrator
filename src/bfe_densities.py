import numpy as np
import biff
import pygadgetreader
import sys
import orbit
import matplotlib.pyplot as plt

def BFE_density_profile(Snlm, Tnlm, nmax, lmax, true_M, true_r_s):
    """
    Computes the density profile of the BFE

    """

    r = np.logspace(-2,1,512)
    pos = np.zeros((len(r),3))
    pos[:,0] = r
    rho_bfe = biff.density(pos, Snlm, Tnlm, nmax, lmax, true_M, true_r_s)
    return r, rho_bfe

def N_body_densities(pos, M, rmax=50, nbins=20):
    """
    Computes density profile of the sim.

    """

    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    r_bins = np.logspace(-2, 1, nbins)
    for i in range(len(rho_bins)-1):
        index_c = np.where((r<r_bins[i+1]) & (r>r_bins[i]))[0]
        V = 4/3.*np.pi*(r_bins[i+1]**3 - r_bins[i]**3)
        rho_bins[i] = np.sum(M[index_c])/V
    return r_bins, rho_bins


if __name__ == "__main__":


    #coeff_path = sys.argv[1]
    coeff_path = './coefficients/ST_MWST_MW_beta0_LMC6_6.26M_snap_0.txt'
    #nmax = sys.argv[2]
    nmax = 10
    #lmax = sys.argv[3]
    lmax = 0
    #M_halo = sys.argv[4]
    #r_s = sys.argv[5]
    #halo_name = sys.argv[6]

    # Read coefficients
    S, T = orbit.read_coefficients(coeff_path, 1, nmax, lmax)
    print(S, T)
    # Compute densities
    r_BFE, rho_BFE = BFE_density_profile(S, T, nmax, lmax, M_halo, r_s)
    # read sim
    #pos = pygadgetreader.readsnap(path_snap, 'pos', 'dm')
    #vel = pygadgetreader.readsnap(path_snap, 'vel', 'dm')
    #pot = pygadgetreader.readsnap(path_snap, 'pot', 'dm')
    plt.plot(r_BFE, rho_BFE, c='k')
    plt.savefig('bfe_density.pdf', bbox_inches='tight')
    # COM
    # truncate halo

    # plot
    
