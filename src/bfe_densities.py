import numpy as np
import biff
import pygadgetreader
import sys
import orbit
import matplotlib.pyplot as plt
import soda

def dens_hernquist(a, r, M):
    a = a
    M = M
    rho = M / (2 * np.pi) * a / (r*(r+a)**3)
    return rho

def pot_hernquist(a, r, M, G):
    a = a
    r = r
    M = M
    phi = -G*M / (r+a)

    return phi

def re_center_halo(pos, r_cm):
    """
    Re-center a halo positions or velocities.
    """

    print('COM coordinates', r_cm)
    for i in range(3):
        pos[:,i] = pos[:,i] - r_cm[i]
    return pos 



def BFE_density_profile(Snlm, Tnlm,  true_M, true_r_s, rmax):
    """
    Computes the density profile of the BFE

    """

    r = np.logspace(-2, rmax, 512)
    pos = np.zeros((len(r), 3))
    pos[:,0] = r
    rho_bfe = biff.density(pos, Snlm, Tnlm, true_M, true_r_s)
    return r, rho_bfe

def BFE_potential_profile(Snlm, Tnlm, true_M, true_r_s, G, rmax):
    """
    Computes the density profile of the BFE

    """

    r = np.logspace(-2, rmax, 512)
    pos = np.zeros((len(r), 3))
    pos[:,0] = r
    pot_bfe = biff.potential(pos, Snlm, Tnlm, G, true_M, true_r_s)
    return r, pot_bfe



def N_body_densities(pos, M, rmax=50, nbins=20):
    """
    Computes density profile of the sim.

    """

    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    r_bins = np.linspace(0.01, rmax, nbins)
    rho_bins = np.zeros(len(r_bins))
    for i in range(len(rho_bins)-1):
        index_c = np.where((r<r_bins[i+1]) & (r>=r_bins[i]))[0]
        V = 4/3.*np.pi*(r_bins[i+1]**3 - r_bins[i]**3)
        rho_bins[i] = np.sum(M[index_c])/V
    dr = (r_bins[1]-r_bins[0])/2.
    return r_bins+dr, rho_bins

def N_body_potential(pos, pot, rmax=50, nbins=20): 
    """
    Computes density profile of the sim.

    """

    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    r_bins = np.linspace(0.01, rmax, nbins)
    pot_bins = np.zeros(len(r_bins))
    for i in range(len(pot_bins)-1):
        index_c = np.where((r<r_bins[i+1]) & (r>=r_bins[i]))[0]
        pot_bins[i] = np.mean(pot[index_c])
    dr = (r_bins[1]-r_bins[0])/2.
    return r_bins+dr, pot_bins

if __name__ == "__main__":


    #coeff_path = sys.argv[1]
    coeff_path = './coefficients/ST_MWST_MW_beta0_LMC6_6.26M_snap_0.txt'
    #nmax = sys.argv[2]
    nmax = 10
    #lmax = sys.argv[3]
    lmax = 0
    #_halo = sys.argv[4]
    #r_s = sys.argv[5]
    #halo_name = sys.argv[6]

    rmax = 2

    path_snap = './test_halo/LMC6_6.25M_vir_000'
    # Read coefficients
    S, T = orbit.read_coefficients(coeff_path, 1, nmax, lmax)
    print(S, T)
    # Compute densities
    #mass = pygadgetreader.readsnap(path_snap, 'mass', 'dm')

    mass = pygadgetreader.readsnap(path_snap, 'mass', 'dm')
    M_halo = np.sum(mass)
    print(M_halo)
    r_s = 25.158
    #r_s = 2.5158
    print(M_halo)
    G = 43007.1
    print(S[0])
    r_BFE, rho_BFE = BFE_density_profile(S[0], T[0], M_halo, r_s, rmax)
    r_BFE, pot_BFE = BFE_potential_profile(S[0], T[0], M_halo, r_s, G, rmax)
    # read sim
    #pos = pygadgetreader.readsnap(path_snap, 'pos', 'dm')
    #vel = pygadgetreader.readsnap(path_snap, 'vel', 'dm')
    #pot = pygadgetreader.readsnap(path_snap, 'pot', 'dm')

    pos = pygadgetreader.readsnap(path_snap, 'pos', 'dm')
    pot = pygadgetreader.readsnap(path_snap, 'pot', 'dm')
    rcm = [-0.01064566, -0.00043313,  0.00702804]
    pos_cm = re_center_halo(pos, rcm)


    r_NB, rho_NB = N_body_densities(pos, mass, rmax=100, nbins=50)
    r_NB, pot_NB = N_body_potential(pos, -pot, rmax=100, nbins=50)



    # tests
    pot_teo = pot_hernquist(r_s, r_NB, M_halo, G)
    rho_teo = dens_hernquist(r_s, r_NB, M_halo)

    plt.semilogy(r_BFE, -pot_BFE, c='k')
    plt.semilogy(r_NB, pot_NB, c='C9', ls='--')
    plt.semilogy(r_NB, -pot_teo, c='C1', ls=':')

    plt.savefig('bfe_density.pdf', bbox_inches='tight')
    # COM
    # truncate halo

    # plot
    
