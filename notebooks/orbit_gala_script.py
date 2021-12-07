import numpy as np
import matplotlib.pyplot as plt
import linecache

import gala.potential as gp
import gala.integrate as gi
from gala.units import UnitSystem
import gala.dynamics as gd
import astropy.units as u



def get_center(coeff_path):
    line = linecache.getline(coeff_path, 5)
    line2 = line.split(":",1)[1].rstrip('\n')
    list_str = line2.replace('[', '').replace(']', '').split(' ')
    list_com = list(filter(None, list_str))

    xcom = float(list_com[0])
    ycom = float(list_com[1])
    zcom = float(list_com[2])


    line = linecache.getline(coeff_path, 6)
    line2 = line.split(":",1)[1].rstrip('\n')
    list_str = line2.replace('[', '').replace(']', '').split(' ')
    list_com = list(filter(None, list_str))

    vxcom = float(list_com[0])
    vycom = float(list_com[1])
    vzcom = float(list_com[2])

    return np.array([xcom, ycom, zcom]), np.array([vxcom, vycom, vzcom])


def analytic_orbits(pos_ics, vel_ics, dt, nsnaps):
    pos = np.zeros((len(pos_ics), nsnaps, 3))
    vel = np.zeros((len(vel_ics), nsnaps, 3))
    m2 = 1.85966E12
    halo_teo = gp.HernquistPotential(m=m2*u.Msun, c=rs_halo*u.kpc,
                                     units=[u.kpc, u.Gyr, u.Msun, u.radian])
    for i in range(len(pos_ics)):
        #print(pos_ics[i])
        #print(vel_ics[i])
        w0 = gd.PhaseSpacePosition(pos=pos_ics[i]*u.kpc, vel=vel_ics[i]*u.km/u.s)
        orbit_halo_teo = gp.Hamiltonian(halo_teo).integrate_orbit(w0, dt=dt*u.Gyr, n_steps=nsnaps-1)
        pos[i] = orbit_halo_teo.xyz.value.T
        vel[i] = orbit_halo_teo.v_xyz.value.T
        #print(vel[i][0])
    return pos, vel

def scf_orbits(S, T, pos_ics, vel_ics, dt, nsnaps, rcom=0, com_xj=1, com_vj=1):
    pos_all_t = np.zeros((len(pos_ics), nsnaps, 3))
    vel_all_t = np.zeros((len(pos_ics), nsnaps, 3))
    t = nsnaps*dt

    #m = 1.5713E12

    if rcom==1:
        print('using moving COM')
        halo_t = gp.scf.InterpolatedSCFPotential(m=1e10, r_s=rs_halo, Sjnlm=S, Tjnlm=T,
                                                 tj=np.arange(0, t, dt),
                                                 com_xj=com_xj, com_vj=com_vj,
                                                 units=[u.kpc, u.Gyr, u.Msun, u.radian])
    else:
        halo_t = gp.scf.InterpolatedSCFPotential(m=1e10, r_s=rs_halo, Sjnlm=S, Tjnlm=T,
                                                 tj=np.arange(0, t, dt),
                                                 com_xj=1, com_vj=1,
                                                 units=[u.kpc, u.Gyr, u.Msun, u.radian])

    if len(np.shape(S))==3:
        halo_t = gp.scf.SCFPotential(m=1e10, r_s=rs_halo, Snlm=S, Tnlm=T,
                                      units=[u.kpc, u.Gyr, u.Msun, u.radian])

    for i in range(len(pos_ics)):
        w0 = gd.PhaseSpacePosition(pos=pos_ics[i]*u.kpc,
                                   vel=vel_ics[i]*u.km/u.s)

        orbit_halo = gp.Hamiltonian(halo_t).integrate_orbit(w0, dt=dt*u.Gyr, n_steps=nsnaps-1)
        pos_all_t[i] = orbit_halo.xyz.value.T
        vel_all_t[i] = orbit_halo.v_xyz.value.T

    return pos_all_t, vel_all_t

def orbits_plots(time, qs, labels, ref_orbit=0):

    n_orbits = np.shape(qs)[0]
    cl = ["#00429d","#00ab7d","#2c2c2c","#ff005e","#810000"]
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs = ax.flatten()
    for i in range(n_orbits):
        print(i, np.shape(qs[i]))
        axs[0].plot(time, np.sqrt(np.sum(qs[i]**2, axis=1)),
                    label=labels[i], c=cl[i], lw=2)
        axs[1].plot(time, np.sqrt(np.sum((qs[i]-qs[ref_orbit])**2, axis=1)))
        axs[0].legend(ncol=5, fontsize=13)
    axs[0].set_xlim(-0.2, 5)
    axs[0].set_ylim(5, 40)

    axs[1].set_title('Residuals with respect to Gadget', fontsize=18)
    axs[1].set_xlabel('Time [Gyrs]')

    axs[0].set_ylabel(r'$r_{gal} \rm{[kpc]}$')
    axs[1].set_ylabel(r'$r_{gal} \rm{[kpc]}$')
    plt.show()

    return fig


if __name__ == "__main__":
    ########### Parameters defintion #########################
    dt = 0.02
    nsnaps_nbody = 237
    nmax = 20
    lmax = 20
    mmax = 20
    #rs_halo = 34.51
    rs_halo = 34.5
    n_orbits = 30 # in the N-body file
    t_orbit = np.arange(0, nsnaps_nbody*dt, dt)
    coeff_path_centers = '../../time-dependent-BFE/data/MW_halo/BFE_MW2_10M_halo_vir_host_snap'
    n_body_orbits = '../data/nbody_orbits/orbits_circular_MW2_10M_halo_vir_particle_'
    ## Loading coefficients
    S, T = np.loadtxt('../data/coefficients/MW2_10M_test_halo_sn_5.txt', unpack=True)
    S_scf = S.reshape((nsnaps_nbody, nmax+1, lmax+1, mmax+1))
    T_scf = T.reshape((nsnaps_nbody, nmax+1, lmax+1, mmax+1))
    print('Done loading {} coefficients with nmax={} and lmax={}'.format(nsnaps_nbody, nmax, lmax))
    # COM of the objects
    rcom_all = np.zeros((int(nsnaps_nbody), 3))
    vcom_all = np.zeros((int(nsnaps_nbody), 3))

    for i in range(int(nsnaps_nbody)):
        rcom_all[i], vcom_all[i] = get_center(coeff_path_centers+'_{:0>3d}.txt'.format(i))

    print('Done loading expansion center')
    # ICS:
    data_nbody_all = np.zeros((n_orbits, nsnaps_nbody, 6))
    for i in range(n_orbits):
        data_nbody_all[i] = np.loadtxt(n_body_orbits+'{:0>3d}.txt'.format(i))

    print('Done loading N-body orbits')
    r_init = data_nbody_all[0:n_orbits,0,0:3]
    v_init = data_nbody_all[0:n_orbits,0,3:6]
    # Nbody orbits
    xyz_all = data_nbody_all[0:n_orbits,:,0:3]
    vxyz_all = data_nbody_all[0:n_orbits,:,3:6]
    p_orbit = 9
    print('Integrating orbit with ICs: \n r={} \n v={} \n time-step={}'.format(r_init[9], v_init[9],dt))
    # Integrating orbits
    pos_teo, vel_teo = analytic_orbits(r_init[p_orbit:p_orbit+1], v_init[p_orbit:p_orbit+1], dt, nsnaps_nbody)
    pos_scf, vel_scf = scf_orbits(S_scf, T_scf, r_init[p_orbit:p_orbit+1], v_init[p_orbit:p_orbit+1], dt, nsnaps_nbody)
    pos_scf_rcom, vel_scf_rcom = scf_orbits(S_scf, T_scf, r_init[p_orbit:p_orbit+1], v_init[p_orbit:p_orbit+1], dt,
                                            nsnaps_nbody, rcom=1, com_xj=rcom_all)
    pos_scf_rvcom, vel_scf_rvcom = scf_orbits(S_scf, T_scf, r_init[p_orbit:p_orbit+1], v_init[p_orbit:p_orbit+1], dt,
                                             nsnaps_nbody, rcom=1, com_xj=rcom_all,
                                             com_vj=vcom_all)

    labels = ['Analytic', 'SCF', 'SCF rcom', 'SCF rvcom', 'Gadget']
    orbits_plots(t_orbit, np.array([pos_teo[0], pos_scf[0], pos_scf_rcom[0], pos_scf_rvcom[0], xyz_all[p_orbit]]), labels, 4)
