import numpy as np
import biff
import pygadgetreader
import sys
import orbit
import matplotlib.pyplot as plt
import soda
import matplotlib.ticker as ticker

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



def BFE_density_profile(pos, Snlm, Tnlm,  true_M, true_r_s, rmax, nbins=20):
    """
    Computes the density profile of the BFE

    """

    #r = np.logspace(-2, rmax, 512)
    #pos = np.zeros((len(r), 3))
    #pos[:,0] = r
    rho_bfe = biff.density(np.ascontiguousarray(pos.astype(np.double)), Snlm, Tnlm, true_M, true_r_s)

    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    r_bins = np.linspace(0.01, rmax, nbins)
    rho_bins = np.zeros(len(r_bins))
    for i in range(len(rho_bins)-1):
        index_c = np.where((r<r_bins[i+1]) & (r>=r_bins[i]))[0]
        rho_bins[i] = np.mean(rho_bfe[index_c])
    dr = (r_bins[1]-r_bins[0])/2.
    return r_bins+dr, rho_bins

def BFE_potential_profile(pos, Snlm, Tnlm, true_M, true_r_s, G, rmax, nbins=20):
    """
    Computes the density profile of the BFE

    """

    pot_bfe = biff.potential(np.ascontiguousarray(pos.astype(np.double)), Snlm, Tnlm, G, true_M, true_r_s)
    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    r_bins = np.linspace(0.01, rmax, nbins)
    pot_bins = np.zeros(len(r_bins))
    for i in range(len(pot_bins)-1):
        index_c = np.where((r<r_bins[i+1]) & (r>=r_bins[i]))[0]
        pot_bins[i] = np.mean(pot_bfe[index_c])
    dr = (r_bins[1]-r_bins[0])/2.
    return r_bins+dr, pot_bins



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
    #coeff_path = './coefficients/ST_MWST_MW_beta0_LMC6_6.26M_snap_0.txt'
    coeff_path = './coefficients/ST_MWST_MW_beta0_40M_snap_0_n20_l2.txt'
    #oeff_path = './coefficients/ST_test.txt'
    #nmax = sys.argv[2]
    nmax = 20
    #lmax = sys.argv[3]
    lmax = 2
    #_halo = sys.argv[4]
    #r_s = sys.argv[5]
    #halo_name = sys.argv[6]

    rmax = 300

    #path_snap = './test_halo/LMC6_6.25M_vir_000'
    path_snap = './test_halo/MW2_40M_vir_000'
    # Read coefficients
    S, T = orbit.read_coefficients(coeff_path, 1, nmax, lmax)
    #print(S, T)
    # Compute densities

    mass = pygadgetreader.readsnap(path_snap, 'mass', 'dm')
    M_halo = np.sum(mass)
    print(M_halo)
    r_s = 40.82
    #r_s = 25.158
    #r_s = 2.5158
    G = 43007.1
    print(S[0])
    # read sim
    #pos = pygadgetreader.readsnap(path_snap, 'pos', 'dm')
    #vel = pygadgetreader.readsnap(path_snap, 'vel', 'dm')
    #pot = pygadgetreader.readsnap(path_snap, 'pot', 'dm')

    pos = pygadgetreader.readsnap(path_snap, 'pos', 'dm')
    pot = pygadgetreader.readsnap(path_snap, 'pot', 'dm')
    rcm = [-0.01064566, -0.00043313, 0.00702804]
    pos_cm = re_center_halo(pos, rcm)

    # BFE
    r_BFE, rho_BFE = BFE_density_profile(pos_cm, S[0], T[0], 1, r_s, rmax=300, nbins=100)
    r_BFE, pot_BFE = BFE_potential_profile(pos_cm, S[0], T[0], 1, r_s, G, rmax=300, nbins=100)

    # N-body
    r_NB, rho_NB = N_body_densities(pos, mass, rmax=300, nbins=100)
    r_NB, pot_NB = N_body_potential(pos, pot, rmax=300, nbins=100)

    # Analytic
    pot_teo = pot_hernquist(r_s, r_NB, M_halo, G)
    rho_teo = dens_hernquist(r_s, r_NB, M_halo)

    fig, ax = plt.subplots(2, 2)

    ax[0][0].semilogy(r_BFE, rho_BFE, c='k', label='BFE')
    ax[0][0].semilogy(r_NB, rho_NB, c='C9', ls='--', label='Gadget')
    ax[0][0].semilogy(r_NB, rho_teo, c='C1', ls=':', label='Analytic')
    ax[0][0].set_ylabel(r'$\rho$')
    ax[0][0].set_xticks(np.arange(0, 300, 50))

    ax[1][0].semilogy(r_BFE,-pot_BFE, c='k')
    ax[1][0].semilogy(r_NB, -pot_NB, c='C9', ls='--')
    ax[1][0].semilogy(r_NB, -pot_teo, c='C1', ls=':')
    ax[1][0].set_ylabel(r'$\phi$')
    ax[1][0].set_xlabel(r'$r [kpc]$')
    ax[1][0].set_xticks(np.arange(0, 300, 50))

    ax[0][1].plot(r_BFE, rho_BFE/rho_teo, c='k', label='BFE/Analytic')
    ax[0][1].plot(r_NB, rho_NB/rho_teo, c='C9', ls='--', label='Gadget/Analytic')
    ax[0][1].plot(r_NB, rho_BFE/rho_NB, c='C1', ls=':', label='BFE/Gadget')
    ax[0][1].set_ylabel(r'$\Delta \rho$')
    ax[0][1].set_xticks(np.arange(0, 300, 50))

    ax[1][1].plot(r_BFE, pot_BFE/pot_teo, c='k')
    ax[1][1].plot(r_NB, pot_NB/pot_teo, c='C9', ls='--')
    ax[1][1].plot(r_NB, pot_BFE/pot_NB, c='C1', ls=':')
    ax[1][1].set_ylabel(r'$\Delta \phi$')
    ax[1][1].set_xlabel(r'$r [kpc]$')
    ax[1][1].set_xticks(np.arange(0, 300, 50))

    ax[0][0].legend(fontsize=15)
    ax[0][1].legend(fontsize=15)

    plt.savefig('bfe_density_MW_40M.pdf', bbox_inches='tight')

    # COM
    # truncate halo

    # plot
