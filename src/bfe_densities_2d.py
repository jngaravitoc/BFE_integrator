import numpy as np
import biff
import pygadgetreader
import sys
import orbit
import matplotlib.pyplot as plt
import soda
import matplotlib.ticker as ticker
import sys
sys.path.insert(0, '../../MW_anisotropy/code/densities/')
sys.path.insert(0, '../../MW_anisotropy/code/kinematics/')
sys.path.insert(0, '../../MW_anisotropy/code/')
import density_tools


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



def BFE_density(Snlm, Tnlm,  true_M, true_r_s, xmin, xmax, ymin, ymax, nbinsx, nbinsy, z_plane):
    """
    Computes the density profile of the BFE
    """
    y_bins = np.linspace(xmin, xmax, nbinsx)
    z_bins = np.linspace(ymin, ymax, nbinsy)
    y, z= np.meshgrid(y_bins, z_bins)

    pos = np.array([np.ones(len(y.flatten()))*z_plane, y.flatten(), z.flatten()]).T

    rho_bfe = biff.density(np.ascontiguousarray(pos.astype(np.double)), Snlm, Tnlm, true_M, true_r_s)

    return rho_bfe

def BFE_potential(pos, Snlm, Tnlm, true_M, true_r_s, G):
    """
    Computes the density profile of the BFE

    """

    pot_bfe = biff.potential(np.ascontiguousarray(pos.astype(np.double)), Snlm, Tnlm, G, true_M, true_r_s)
    return pot_bfe



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
    #mass = pygadgetreader.readsnap(path_snap, 'mass', 'dm')

    mass = pygadgetreader.readsnap(path_snap, 'mass', 'dm')
    M_halo = np.sum(mass)
    print(M_halo)
    r_s = 40.82
    #r_s = 25.158
    #r_s = 2.5158
    G = 43007.1
    print(S[0])
    # read sim

    pos = pygadgetreader.readsnap(path_snap, 'pos', 'dm')
    pot = pygadgetreader.readsnap(path_snap, 'pot', 'dm')
    rcm = [-0.01064566, -0.00043313, 0.00702804]
    pos_cm = re_center_halo(pos, rcm)

    nbinsx = 100
    nbinsy = 100
    xmin = -300
    xmax = 300
    ymin = -300
    ymax = 300
    z_plane = 10
    nn = 1000

    # BFE
    rho_BFE = BFE_density(S[0], T[0], 1, r_s, xmin, xmax, ymin, ymax, nbinsx, nbinsy, z_plane)
    rho_BFE_mat = rho_BFE.reshape(nbinsx, nbinsy)

    # N-body
    rho_NB = density_tools.density_nn(pos, nbinsx+1, nbinsy+1, z_plane, nn, xmin, xmax, ymin, ymax)

    density_tools.density_peaks((rho_BFE_mat/rho_NB-1), xmin=-300,\
                                 xmax=300, ymin=-300, ymax=300, fsize=(6, 6))#, vmin=-1, vmax=1, levels=np.arange(-1, 1, 0.01))

    plt.savefig('bfe_2ddensity_MW40M_n20_l2.pdf', bbox_inches='tight')


    #density_tools.density_peaks(rho_BFE, xmin=-300, xmax=300, ymin=-300, ymax=300, figsize=(7, 6))
    #plt.savefig('bfe_2ddensity_test.pdf', bbox_inches='tight')

    # COM
    # truncate halo

    # plot
