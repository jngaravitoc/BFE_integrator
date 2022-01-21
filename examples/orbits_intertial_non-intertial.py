import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy import units as u
import scipy.linalg as la

# Gala
from gala import potential as gp
import gala.dynamics as gd
from gala.potential.scf import compute_coeffs
from gala.units import galactic

# Local bfe code
sys.path.append('../src/')
from profiles import Hernquist
import bfe_fields as bfe
import leapfrog_bfe
from profiles import G



def run_gala_orbit(pos_sat, vel_sat, pos_tp, vel_tp):
    """
    Specify units in ICs
    """
    ## GALA set up

    # Design gala potential using Nbody and a test particle


    # Present-day position/velocity in inertial frame moving with instantaneous
    # Milky Way velocity:
    w0_mw = gd.PhaseSpacePosition(
        pos=[0, 0, 0]*u.kpc,
        vel=[0, 0, 0]*u.km/u.s,
    )
    pot_mw = gp.HernquistPotential(m=Mhost*u.Msun, c=1*u.kpc, units=galactic)

    # Values from Vasiliev et al. 2020
    w0_lmc = gd.PhaseSpacePosition(
        pos=pos_sat,
        vel=vel_sat,
    )

    w0_test = gd.PhaseSpacePosition(
        pos=pos_tp,
        vel=vel_tp,
    )


    pot_lmc = gp.HernquistPotential(m=Msat*u.Msun, c=1*u.kpc, units=galactic)

    w0 = gd.combine((w0_mw, w0_lmc, w0_test))
    nbody = gd.DirectNBody(w0, [pot_mw, pot_lmc, None])

    w1 = gd.combine((w0_lmc, w0_test))
    isolated = gd.DirectNBody(w1, [pot_lmc, None])




    # Full : Host - Satellite - test particle
    orbits = nbody.integrate_orbit(dt=dt*u.Gyr, t1=tmin*u.Gyr, t2=tmax*u.Gyr)


    # Orbit quantities full orbit
    pos_halo = np.array([orbits.xyz[:,:,0].to(u.kpc).value])[0]
    pos_sat = np.array([orbits.xyz[:,:,1].to(u.kpc).value])[0]
    pos_tp = np.array([orbits.xyz[:,:,2].to(u.kpc).value])[0]

    vel_halo = np.array([orbits.v_xyz[:,:,0].to(u.km/u.s).value])[0]
    vel_sat = np.array([orbits.v_xyz[:,:,1].to(u.km/u.s).value])[0]
    vel_tp = np.array([orbits.v_xyz[:,:,2].to(u.km/u.s).value])[0]


    r_halo = (np.sum(orbits.xyz[:,:,0]**2, axis=0))**0.5
    r_sat = (np.sum(orbits.xyz[:,:,1]**2, axis=0))**0.5
    r_tp = (np.sum(orbits.xyz[:,:,2]**2, axis=0))**0.5

    v_halo = (np.sum(orbits.v_xyz[:,:,0].to(u.km/u.s).value**2, axis=0))**0.5
    v_sat = (np.sum(orbits.v_xyz[:,:,1].to(u.km/u.s).value**2, axis=0))**0.5
    v_tp = (np.sum(orbits.v_xyz[:,:,2].to(u.km/u.s).value**2, axis=0))**0.5


    rtp_r2_sat = np.sum((np.array(orbits.xyz[:,:,2].to(u.kpc).value-orbits.xyz[:,:,1].to(u.kpc).value))**2, axis=0)**0.5
    vtp_r2_sat = np.sum((np.array(orbits.v_xyz[:,:,2].to(u.km/u.s).value-orbits.v_xyz[:,:,1].to(u.km/u.s).value))**2, axis=0)**0.5

    return [pos_halo, vel_halo], [pos_sat, vel_sat], [pos_tp, vel_tp]

def bfe_orbit(q, t, S, T, nmax, lmax, Mass, rs, com, acc):
    orbit_scft_acom = leapfrog_bfe.integrate_bfe_t(q=q,
                                                   time=t,
                                                   S=S, T=T , nmax=nmax, lmax=lmax,
                                                   G=G, Mass=Mass, rs=1, exp_com=com,
                                                   exp_acc = -acc,
                                                   disk=0, LMC=0)


    #t_scft_acom = orbit_scft_acom[0]
    x_scf = orbit_scft_acom[1]
    y_scf = orbit_scft_acom[2]
    z_scf = orbit_scft_acom[3]
    vx_scf = orbit_scft_acom[4]
    vy_scf = orbit_scft_acom[5]
    vz_scf = orbit_scft_acom[6]
    #r_scf = np.sqrt(x_scft_acom**2 + y_scft_acom**2 + z_scft_acom**2)
    #r_scf_sat = np.sqrt((x_scft_acom-sat_com[:-1,0])**2 + (y_scft_acom-sat_com[:-1,1])**2 + (z_scft_acom-sat_com[:-1,2])**2)
    return np.array([x_scf, y_scf, z_scf]), np.array([vx_scf, vy_scf, vz_scf])

# Functions to compute BFE
def density_func_sat(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2) # * u.kpc
    M = Msat # u.Msun
    rs = 1 #*u.kpc
    rho0 = M/(2*np.pi*rs**3)
    return rho0 / ((r/rs) *(1+r/rs)**3)

def density_func_host(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2) # * u.kpc
    M = Mhost # u.Msun
    rs = 1 #*u.kpc
    rho0 = M/(2*np.pi*rs**3)
    return rho0 / ((r/rs) *(1+r/rs)**3)

def compute_acceleration(vel):
    ax = np.diff(vel[0], axis=0)/dt*u.km/u.s/u.Gyr
    ay = np.diff(vel[1], axis=0)/dt*u.km/u.s/u.Gyr

    # Transform to kpc/Gyr^2
    ax_ru = np.zeros(nsteps)
    ay_ru = np.zeros(nsteps)

    ax_ru[:-1] = ax.to(u.kpc/u.Gyr**2).value
    ay_ru[:-1] = ay.to(u.kpc/u.Gyr**2).value

    return ax_ru, ay_ru

def plot_panel_1(time, q_host, q_sat, a_host, a_sat, P1_name):
    fig, ax = plt.subplots(2, 3, figsize=(20, 7), sharex=True)
    r_host = np.sqrt(np.sum(q_host[0]**2, axis=0))
    v_host = np.sqrt(np.sum(q_host[1]**2, axis=0))

    ax[0][0].plot(time, r_host)
    ax[0][1].plot(time, v_host)
    ax[0][2].plot(time, a_host[0], label='ax')
    ax[0][2].plot(time, a_host[1], label='ay')

    r_sat = np.sqrt(np.sum(q_sat[0]**2, axis=0))
    v_sat = np.sqrt(np.sum(q_sat[1]**2, axis=0))


    ax[1][0].plot(time, r_sat)
    ax[1][1].plot(time, v_sat)
    ax[1][2].plot(time, a_sat[0], label='ax')
    ax[1][2].plot(time, a_sat[1], label='ay')
    # Labels

    ax[0][0].set_title('Host position')
    ax[0][1].set_title('Host velocity')
    ax[0][2].set_title('Host acceleration')
    ax[0][2].legend(fontsize=15)

    ax[1][0].set_title('Satellite position')
    ax[1][1].set_title('Satellite velocity')
    ax[1][2].set_title('Satellite acceleration')
    ax[1][2].legend(fontsize=15)

    ax[0][0].set_ylabel(r'$\rm{Distance} [kpc]$')
    ax[0][1].set_ylabel(r'$\rm{Velocity} [km/s]$')
    ax[0][2].set_ylabel(r'$\rm{acc} [kpc/Gyr^2]$')

    ax[1][0].set_ylabel(r'$\rm{Distance} [kpc]$')
    ax[1][1].set_ylabel(r'$\rm{Velocity} [km/s]$')
    ax[1][2].set_ylabel(r'$\rm{acc} [kpc/Gyr^2]$')

    ax[1][0].set_xlabel(r'$\rm{Time} [Gyr]$')
    ax[1][1].set_xlabel(r'$\rm{Time} [Gyr]$')
    ax[1][2].set_xlabel(r'$\rm{Time} [Gyr]$')


    fig.savefig('P1_'+P1_name, bbox_inches='tight', dpi=200)
    return 0


def plot_panel_2(time, q_host, q_sat, q_tp, RF, P2_name):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    r_host = np.sqrt(np.sum(q_host[0]**2, axis=0))
    v_host = np.sqrt(np.sum(q_host[1]**2, axis=0))

    ax[0].plot(q_host[0][0], q_host[0][1], label='Host')
    ax[0].plot(q_sat[0][0], q_sat[0][1], label='Satellite')
    ax[0].plot(q_tp[0][0], q_tp[0][1], label='Test particle')

    ax[0].scatter(q_host[0][0][0], q_host[0][1][0])
    ax[0].scatter(q_sat[0][0][0], q_sat[0][1][0])
    ax[0].scatter(q_tp[0][0][0], q_tp[0][1][0])

    if RF == 'sat':
        ax[1].plot(q_tp[0][0]-q_sat[0][0], q_tp[0][1]-q_sat[0][1], c='C2')
        ax[1].scatter(q_tp[0][0][0]-q_sat[0][0][0], q_tp[0][1][0]-q_sat[0][1][0], c='C2')

        #r_tp = np.sqrt(np.sum((q_tp[0]-q_sat[0])**2, axis=0))
        #v_tp = np.sqrt(np.sum((q_tp[1]-q_sat[1])**2, axis=0))
        ax[1].set_title('Satellite RF')
        #ax[1][0].set_title('Satellite RF')
        #ax[1][1].set_title('Satellite RF')

    elif RF == 'host':
        ax[1].plot(q_tp[0][0]-q_host[0][0], q_tp[0][1]-q_host[0][1], c='C2')
        ax[1].scatter(q_tp[0][0][0]-q_host[0][0][0], q_tp[0][1][0]-q_host[0][1][0], c='C2')

        #r_tp = np.sqrt(np.sum((q_tp[0]-q_host[0])**2, axis=0))
        #v_tp = np.sqrt(np.sum((q_tp[1]-q_host[1])**2, axis=0))
        ax[1].set_title('Host RF')
        #ax[1][0].set_title('Host RF')
        #ax[1][1].set_title('Host RF')


    ax[0].legend(fontsize=15)

    ax[0].set_xlabel(r'$\rm{x} [kpc]$')
    ax[0].set_ylabel(r'$\rm{y} [kpc]$')

    ax[1].set_xlabel(r'$\rm{x} [kpc]$')
    ax[1].set_ylabel(r'$\rm{y} [kpc]$')




    fig.savefig('P2_'+P2_name, bbox_inches='tight', dpi=200)
    return 0

def plot_panel_3(time, q_host, q_sat, q_tp, q_scf, RF, com, P3_name):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    if RF == 'sat':
        r_tp = np.sqrt(np.sum((q_tp[0]-q_sat[0])**2, axis=0))
        r_tp_scf = np.sqrt(np.sum((q_scf[0]-com.T)**2, axis=0))

        ax.set_title('Satellite RF')

    elif RF == 'host':
        #ax[1].scatter(q_tp[0][0][0]-q_host[0][0][0], q_tp[0][1][0]-q_host[0][1][0], c='C2')
        r_tp = np.sqrt(np.sum((q_tp[0]-q_host[0])**2, axis=0))
        r_tp_scf = np.sqrt(np.sum((q_scf[0]-com.T)**2, axis=0))
        ax.set_title('Host RF')

    ax.plot(time, r_tp, label='Nbody')
    ax.plot(time, r_tp_scf, label='SCF + acceleration')
    #ax.plot(time, v_tp)

    ax.set_ylabel(r'$\rm{Distance} [kpc]$')
    ax.set_xlabel(r'$\rm{Time} [Gyr]$')
    ax.legend(fontsize=15)
    fig.savefig('P3_'+P3_name, bbox_inches='tight', dpi=200)

def main(pos_sat, vel_sat, pos_tp, vel_tp):
    print('Running Nbody orbit in Gala')
    q_host, q_sat, q_tp = run_gala_orbit(pos_sat*u.kpc, vel_sat*u.km/u.s
                                        , pos_tp*u.kpc, vel_tp*u.km/u.s)

    print('Computing SCF expansions')

    scf_coeffs_host = compute_coeffs(density_func_host, nmax=nmax, lmax=lmax, M=Mhost, r_s=1)
    S_host = scf_coeffs_host[0][0]
    T_host = scf_coeffs_host[1][0]

    scf_coeffs_sat = compute_coeffs(density_func_sat, nmax=nmax, lmax=lmax, M=Msat, r_s=1)
    S_sat = scf_coeffs_host[0][0]
    T_sat = scf_coeffs_host[1][0]

    # Computing time-dependent coefficients
    S_all_host = np.zeros((nsteps, nmax+1, lmax+1, lmax+1))
    T_all_host = S_all_host

    S_all_sat = np.zeros((nsteps, nmax+1, lmax+1, lmax+1))
    T_all_sat = S_all_sat

    for k in range(nsteps):
        S_all_host[k] = S_host
        S_all_sat[k] = S_sat

    # Compute halo acceleration and # COM
    print(np.shape(q_host[1][0]), "q_host[1]")
    ax_host, ay_host = compute_acceleration(q_host[1])
    ax_sat, ay_sat = compute_acceleration(q_sat[1])

    print('ploting orbits')
    #bfe_orbit()


    sat_com = np.array([q_sat[0][0]-q_sat[0][0][0], q_sat[0][1]-q_sat[0][1][0], np.zeros(nsteps)]).T
    host_com = np.array([q_host[0][0], q_host[0][1], np.zeros(nsteps)]).T

    plot_panel_1(t_orbit, q_host, q_sat, [ax_host, ay_host], [ax_sat, ay_sat], "test_panel1.png")
    plot_panel_2(t_orbit, q_host, q_sat, q_tp, RF, "test_panel2.png")

    if RF=='sat':
        q_tp_bfe = bfe_orbit(q_ics, t_orbit, S_all_sat, T_all_sat, nmax, lmax, Msat, 1, sat_com,
                             np.array([ax_sat, ay_sat, np.zeros_like(ax_sat)]).T)
        plot_panel_3(t_orbit, q_host, q_sat, q_tp, q_tp_bfe, RF, sat_com, "test_panel3.png")

    elif RF=='host':
        print(np.shape(t_orbit), "t_orbit")
        q_tp_bfe = bfe_orbit(q_ics, t_orbit, S_all_host, T_all_host, nmax, lmax, Mhost, 1, host_com,
                        np.array([ax_host, ay_host, np.zeros_like(ax_host)]).T)
        print(np.shape(q_tp_bfe[0]))
        plot_panel_3(t_orbit, q_host, q_sat, q_tp, q_tp_bfe, RF, host_com, "test_panel3.png")

        #plot_panel_2()
    return 0

if __name__ == "__main__":
    # Global variables
    figs_path = "../../time-dependent-scf/notebooks/figures/intertial_vs_non-intertial_ex/"
    Mhost = 1E12
    Msat = 5E11
    nmax = 10
    lmax = 1
    tmax = 8
    tmin = 0
    dt = 0.000001
    nsteps = int((tmax-tmin)/dt)
    print("nsteps", nsteps)

    pos_sat = [1200, 200, 0]
    vel_sat = [-300, 0, 0]

    pos_tp = [50, 0, 0]
    vel_tp = [0, -130, 0]

    v_factor = (1*u.km/u.s).to(u.kpc/u.Gyr).value
    q_ics = [50, 0, 0, 0*v_factor, -130*v_factor, 0]
    RF = 'host'

    t_orbit = np.linspace(tmin, tmax, nsteps)

    main(pos_sat, vel_sat, pos_tp, vel_tp)
