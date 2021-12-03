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


def read_orbit(snap):
    orbit = np.loadtxt(snap)
    t = orbit[:,0]
    x = orbit[:,1]
    y = orbit[:,2]
    z = orbit[:,3]
    return t, x, y, z

def orbit_lmc(path):
    orbit = np.loadtxt(path)
    xmw = orbit[:,0]
    ymw = orbit[:,1]
    zmw = orbit[:,2]
    xlmc = orbit[:,6]
    ylmc = orbit[:,7]
    zlmc = orbit[:,8]
    x = xlmc - xmw
    y = ylmc - ymw
    z = zlmc - zmw
    return xmw, ymw, zmw, x, y, z

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



def BFE_density(Snlm, Tnlm,  true_M, true_r_s, xmin, xmax, ymin, ymax, nbinsx, nbinsy, z_plane, **kwargs):
    """
    Computes the density profile of the BFE
    """
    y_bins = np.linspace(xmin, xmax, nbinsx)
    z_bins = np.linspace(ymin, ymax, nbinsy)
    y, z= np.meshgrid(y_bins, z_bins)

    pos = np.array([np.ones(len(y.flatten()))*z_plane, y.flatten(), z.flatten()]).T

    rho_bfe = biff.density(np.ascontiguousarray(pos.astype(np.double)), Snlm, Tnlm, true_M, true_r_s)

    if 'S2' in kwargs.keys():
        S2 = kwargs['S2']
        T2 = kwargs['T2']
        M2 = kwargs['M2']
        r_s2 = kwargs['r_s2']
        ylmc = kwargs['ylmc']
        zlmc = kwargs['zlmc']
        print(ylmc, zlmc)
        y_bins2 = np.linspace(ylmc-xmin, ylmc-xmax, nbinsx)
        z_bins2 = np.linspace(zlmc-ymin, zlmc-ymax, nbinsy)
        y2, z2 = np.meshgrid(y_bins2, z_bins2)
        pos2 = np.array([np.ones(len(y2.flatten()))*z_plane, y2.flatten(), z2.flatten()]).T
        rho_bfe2 = biff.density(np.ascontiguousarray(pos2.astype(np.double)), S2, T2, M2, r_s2)


    return  rho_bfe, rho_bfe2

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
    #coeff_path = './coefficients/ST_MWST_MW_beta0_40M_snap_0_n20_l2.txt'
    coeff_path1 = './coefficients/ST_MWST_MWLMC6_beta0_100M_snap_114_n20_l2.txt'
    coeff_path2 = './coefficients/ST_LMCST_MWLMC6_beta0_100M_snap_114_n20_l2.txt'

    #oeff_path = './coefficients/ST_test.txt'
    #nmax = sys.argv[2]
    nmax = 20
    #lmax = sys.argv[3]
    lmax = 5
    #_halo = sys.argv[4]
    #r_s = sys.argv[5]
    #halo_name = sys.argv[6]

    rmax = 300

    r_s = 40.82
    #r_s = 25.158
    #r_s = 2.5158
    G = 43007.1

    # read sim
    #path_snap = './test_halo/LMC6_6.25M_vir_000'
    path_lmc_orbit = './orbits/MWLMC6_100M_b0_orbit_2.txt'
    xmw, ymw, zmw, xlmc, ylmc, zlmc = orbit_lmc(path_lmc_orbit)
    #path_snap = './test_halo/MW2_40M_vir_000'
    nbinsx = 100
    nbinsy = 100
    xmin = -250
    xmax = 250
    ymin = -250
    ymax = 250
    z_plane = 10
    nn = 10000

    # LMC orbit
    # Read coefficients
    static = read_orbit('./test_obit_LMC6')
    t_st = static[0]
    x_st = static[1]
    y_st = static[2]
    z_st = static[3]
    tevol = read_orbit('./test_obit_LMC6_tevol')

    t_t = tevol[0]
    x_t = tevol[1]
    y_t = tevol[2]
    z_t = tevol[3]

    for i in range(113, 114):
        coeff_path1 ='./coefficients/ST_MWST_MWLMC6_beta0_100M_snap_{:0>3d}_n20_l2.txt'.format(i)
        coeff_path2 = './coefficients/ST_LMCST_MWLMC6_beta0_100M_snap_{:0>3d}_n20_l2.txt'.format(i)
        S, T = orbit.read_coefficients(coeff_path1, 1, nmax, lmax)
        S2, T2 = orbit.read_coefficients(coeff_path2, 1, nmax, lmax)
        #print(S, T)
        # Compute densities
        #mass = pygadgetreader.readsnap(path_snap, 'mass', 'dm')

        #mass = pygadgetreader.readsnap(path_snap, 'mass', 'dm')
        #pos = pygadgetreader.readsnap(path_snap, 'pos', 'dm')
        #pot = pygadgetreader.readsnap(path_snap, 'pot', 'dm')

        #rcm = [xmw[114], ymw[114], zmw[114]]
        #pos_cm = re_center_halo(pos, rcm)

        # BFE
        rho_BFE, rho_BFE2 = BFE_density(S[0], T[0], 1, r_s, xmin, xmax, ymin, ymax,\
                                        nbinsx, nbinsy, z_plane, S2 = S2[0]/1E10, T2 = T2[0]/1E10,\
                                        M2 = 1, r_s2 = 25.158, ylmc=ylmc[i], zlmc=zlmc[i])
 
        rho_BFE_mat1 = rho_BFE.reshape(nbinsx, nbinsy)
        rho_BFE_mat2 = rho_BFE2.reshape(nbinsx, nbinsy)

        rho_BFE_mat = rho_BFE_mat1 + rho_BFE_mat2

        # N-body
        #rho_NB = density_tools.density_nn(pos, nbinsx+1, nbinsy+1, z_plane, nn, xmin, xmax, ymin, ymax)

        #density_tools.density_peaks((rho_BFE_mat/rho_NB-1)*100, xmin=-300,\
        #                             xmax=300, ymin=-300, ymax=300, fsize=(6, 6))#, vmin=-1, vmax=-0.9, levels=np.arange(-1, 1, 0.01))
        density_tools.density_peaks((np.log(rho_BFE_mat.T))/np.abs(np.max(np.log(rho_BFE_mat))),
                                     xmin=-250,\
                                     xmax=250, ymin=-250, ymax=250,\
                                     fsize=(6, 6), levels=np.arange(-2.4, -1, 0.01),\
                                     vmin=-2.4, vmax=-1,\
                                     lmc_orbit_cartesian=path_lmc_orbit)
        #plt.plot(y_t[:i], z_t[:i], c='k')
        #plt.plot(y_st[:i], z_st[:i], c='k', alpha=0.6, ls='--')

        #plt.scatter(y_t[i], z_t[i], c='k', marker='*', s=180, label='Time Evolving MW+LMC')
        #plt.scatter(y_st[i], z_st[i], c='k', alpha=0.6, label='Static MW')
        #plt.legend(loc='Upper left', fontsize=15)

        # Remove axis names
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        #-------------------------------- circles ------------------------
        theta = np.linspace(-np.pi, np.pi, 200)
        plt.plot(50*np.cos(theta), 50*np.sin(theta), c='k', ls='--', lw=0.4)
        #plt.plot(100*np.cos(theta), 100*np.sin(theta), c='k', ls='--', lw=0.4)
        plt.plot(150*np.cos(theta), 150*np.sin(theta), c='k', ls='--', lw=0.4)
        plt.plot(280*np.cos(theta), 280*np.sin(theta), c='k', ls='--', lw=0.4)

        #plt.xlabel(r'$\mathrm{y\, [kpc]}$')
        #plt.ylabel(r'$\mathrm{z\, [kpc]}$')
        plt.plot(np.linspace(-200, -100, 10), np.ones(10)*(-200), c='k', lw=2)
        plt.text(-200, -190, r'$\mathrm{100\, kpc}$', color='k', size=20)
        plt.text(-25, 52, r'$\mathrm{50\, kpc}$', color='k', size=12)
        plt.text(-25, 152, r'$\mathrm{150\, kpc}$', color='k', size=12)
        plt.text(150, 180, r'$\mathrm{R_{vir}:280\, kpc}$', color='k', size=10,rotation=315)

        plt.xticks(np.arange(-250, 251, 100))
        plt.yticks(np.arange(-250, 251, 100))
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.xlim(-250, 250)
        plt.ylim(-250, 250)
        plt.savefig('bfe_2ddensity_MWLMC6_snap_{:0>3d}.png'.format(i), bbox_inches='tight', pad_inches=0)
        plt.close()


