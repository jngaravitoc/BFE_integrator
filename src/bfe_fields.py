#!/usr/bin/env python3.8
"""
Script that computes various BFE fields. Here we follow moslty the notation
of Lowing+11.

Equations for the acceletation reference Hernquist and Ostriker 1992
https://ui.adsabs.harvard.edu/abs/1992ApJ...386..375H/abstract

Nico Garavito-Camargo
Center for Computational Astrophysics | Flatiron Institute

October/26/2021

TODO:
 - Fix acceleration computation


"""

import numpy as np
import sys
from scipy import special
import time
from astropy import units as u
from astropy import constants

class BFE_fields:
	def __init__(self, pos, S, T, rs, nmax, lmax, G, M):
		"""
    	Computes parallel BFE potential and density
    	Attributes:
		nlm_list     Creates arrays of indices n, l, m from 3d to 1d.
        bfe_pot      Core function that computed the BFE potential.
        potential    Computes the potential bfe_pot by receiving a task
                     with the arguments that are going to run in parallel.
        main         Runs potential in parallel using a pool to be defined
                         by the user.
        Parameters:
        -----------
        pos : numpy.ndarray with shape
        S : numpy.ndarray
        T : numpy.ndarray
        rs : float
            Hernquist halo scale length
        nmax : int
            nmax in the expansion
        lmax : int
            lmax in the expansion
        G : float
            Value of the gravitational constant
        M : float
            Total mass of the halo (M=1) if the masses
            of each particle where already used for computing the
			coefficients.
		"""
		self.pos = pos
		self.rs = rs*u.kpc
		self.nmax = nmax
		self.lmax = lmax
		self.r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
		self.theta = np.arccos(pos[:,2]/self.r) # cos(theta)
		self.phi = np.arctan2(pos[:,1], pos[:,0])
		self.s = self.r/self.rs.value
		self.G = constants.G
		self.M = M*u.Msun
		self.S = S
		self.T = T
		self.nparticles = int(len(self.s))

	def nlm_list(self):
		# TBD : This can be faster by defining the arrays of all the  n's, l's, and m's
		ncoeff = (self.nmax+1) * (self.lmax+1) * (self.lmax+1)
		n_list = np.zeros(ncoeff)
		l_list = np.zeros(ncoeff)
		m_list = np.zeros(ncoeff)
		S_list = np.zeros(ncoeff)
		T_list = np.zeros(ncoeff)
		i=0
		for n in range(self.nmax+1):
			for l in range(self.lmax+1):
				for m in range(l+1):
					n_list[i] = n
					l_list[i] = l
					m_list[i] = m
					S_list[i] = self.S[n][l][m]
					T_list[i] = self.T[n][l][m]
					i+=1
		return n_list, l_list, m_list, S_list, T_list

	def bfe_phi_nl(self, n, l, m, s):
		# Eq. 11 in Lowing
		# TBD: Check if I'm missing a sqrt(4*pi)
		phi_s =  s**l / (1+s)**(2*l+1)
		phi_nl = special.eval_gegenbauer(n, 2*l+1.5, (s-1)/(s+1))
		return phi_s * phi_nl * np.sqrt(4*np.pi)

	def bfe_potential(self):
		#Eq. 21 in Lowing
		nlist, llist, mlist, Slist, Tlist  = self.nlm_list()
		factor = ((2*llist + 1) * special.factorial(llist-mlist) / special.factorial(llist+mlist))**0.5
		phi_total = np.zeros(self.nparticles)
		for k in range(self.nparticles):
			Ylm = special.sph_harm(mlist, llist, 0, self.theta[k]).real # * factor
			phi_nl = self.bfe_phi_nl(nlist, llist, mlist, self.s[k])

			SpT = Slist * np.cos(mlist*self.phi[k]) + Tlist * np.sin(mlist*self.phi[k])
			phi_nlm = phi_nl * Ylm * SpT
			phi_total[k] = np.sum(phi_nlm)
		units =  self.G * self.M / self.rs
		return  -phi_total * units.to(u.kpc**2/u.Gyr**2)


	def bfe_density(self):
		# Eq 13 in Lowing+11
		nlist, llist, mlist, Slist, Tlist = self.nlm_list()
		Knl = 0.5*nlist*(nlist+4*llist+3) + (llist+1)*(2*llist+1)

		# Factor in the spherical harmonics that to compute as Ylm(theta) = factor*Plm(theta)
		rho_total = np.zeros(self.nparticles)

		# Loop over particle positions
		for k in range(self.nparticles):
			rho_nl = np.sqrt(4*np.pi) * Knl * self.s[k]**(llist) / (self.s[k]*(1+self.s[k])**(2*llist+3)) / (2*np.pi) # Eq. 10
			# Polynomials
			Ylm = special.sph_harm(mlist, llist, 0, self.theta[k]).real
			Xi_nl = special.eval_gegenbauer(nlist, 2*llist+1.5, (self.s[k]-1)/(self.s[k]+1))
			# Azymuthal component
			SpT = Slist * np.cos(mlist * self.phi[k]) + Tlist * np.sin(mlist * self.phi[k])
			# Sum over n,l,m
			rho_total[k] = np.sum(rho_nl * Xi_nl * Ylm * SpT)
		return rho_total * self.M/self.rs**3

	def d_phi_helper(self, n, l, m, s, theta):
		"""
		computes the derivatives of dphi/dr and dYlm/dtheta

		"""
		# Radial component
		# From Eqn. 3.29 in Hernquist and Ostriker 1992
		poly_ratio = special.eval_gegenbauer(n-1, 2*l+2.5, ((s-1)/(s+1))) /  special.eval_gegenbauer(n, 2*l+1.5, ((s-1)/(s+1)))
		dphi_dr_factor = (l/s - (2*l+1)/(1+s) + 4*(2*l+1.5)/(1+s)**2 * poly_ratio)


		# Theta component
		# From wolfram: https://functions.wolfram.com/Polynomials/SphericalHarmonicY/20/ShowAll.html
		# Both definitions are okay but this one is faster to compute numerically.

		l_zeros = np.where((l>=0))
		m_zeros = np.where((m<=l-1))

		dpot_dtheta = np.zeros_like(dphi_dr_factor)
		Plm_1 = np.zeros_like(dphi_dr_factor)

		Plm = special.lpmv(m[l_zeros], l[l_zeros], np.cos(theta))
		Plm_1[m_zeros] = special.lpmv(m[m_zeros], l[m_zeros]-1, np.cos(theta)) # check -1 in m-1

		gamma_factor = special.gamma(l[l_zeros]-m[l_zeros]+1)**0.5 / special.gamma(l[l_zeros]+m[l_zeros]+1)**0.5
		constant = ((2*l+1)/(4*np.pi))**0.5
		dpot_dtheta[l_zeros] = constant * gamma_factor * (l[l_zeros]*np.cos(theta) * Plm - (l[l_zeros]+m[l_zeros])*Plm_1)  / np.sin(theta)


		return dphi_dr_factor, dpot_dtheta

	def bfe_acceleration(self):
		# Computes bfe accelerations Eqn: 17-19 in Lowing+11
		ax = np.zeros(self.nparticles)
		ay = np.zeros(self.nparticles)
		az = np.zeros(self.nparticles)
		nlist, llist, mlist, Slist, Tlist = self.nlm_list()

		for k in range(self.nparticles):
			# Defining special functions and variables
			phi_nl_r = self.bfe_phi_nl(nlist, llist, mlist, self.s[k])

			# Computing the potential derivatives
			dpot_dr_factor, dpot_dtheta = self.d_phi_helper(nlist, llist, mlist, self.s[k], self.theta[k])

			Ylm = special.sph_harm(mlist, llist, 0, self.theta[k]).real

			SpT =  Slist*np.cos(mlist*self.phi[k]) + Tlist*np.sin(mlist*self.phi[k])
			TmS =  Tlist*np.cos(mlist*self.phi[k]) - Slist*np.sin(mlist*self.phi[k])


			# Computing the accelerations in spherical coordinates
			a_r = - np.sum(Ylm * dpot_dr_factor * phi_nl_r * SpT )
			a_theta = - 1/self.s[k] * np.sum(dpot_dtheta * phi_nl_r * SpT)
			a_phi = - 1/self.s[k] * np.sum(mlist * Ylm * phi_nl_r * TmS / np.sin(self.theta[k]))

			# Computing the acceleration in Cartessian coordinates
			# Eqns. 3.18-3.20 in Hernquist & Ostriker 92

			ax[k] = a_r * np.cos(self.phi[k]) * np.sin(self.theta[k]) + \
			 		a_theta * np.cos(self.phi[k]) * np.cos(self.theta[k]) - \
					a_phi * np.sin(self.phi[k])

			ay[k] = a_r * np.sin(self.phi[k]) * np.sin(self.theta[k]) + \
			 		a_theta * np.sin(self.phi[k]) * np.cos(self.theta[k]) + \
					a_phi * np.cos(self.phi[k])

			az[k] = a_r * np.cos(self.theta[k]) - a_theta * np.sin(self.theta[k])

		units = (self.G * self.M / self.rs**2).to(u.kpc/u.Gyr**2)

		return ax*units, ay*units, az*units



if __name__ == "__main__":
	import gala.dynamics as gd
	import gala.potential as gp

	#print("Start")
	#t1 = time.time()
	npoints = 1
	#coeff = np.loadtxt('/rsgrps/gbeslastudents/nicolas/MWLMC_sims/BFE/MW/MWLMC5/BFE_MWLMC5_b1snap_061.txt')
	S = np.ones((11, 11, 11))
	T = np.ones((11, 11, 11))

	S = np.random.rand(1331).reshape(11, 11, 11)*10

	pos = np.random.randint(-100, 100, (npoints, 3))

	halo = BFE_fields(pos, S, T, rs=14, nmax=10, lmax=10, G=1 ,M=50)
	a = halo.bfe_acceleration()
	rho = halo.bfe_density()
	pot = halo.bfe_potential()


	print("Gala \n")
	halo_gala = gp.SCFPotential(m=50, r_s=14, Snlm=S, Tnlm=T, units=['kpc', 'Msun', 'Gyr', 'degree'])
	a_gala = halo_gala.gradient(pos.T)
	rho_gala = halo_gala.density(pos.T)
	pot_gala = halo_gala.energy(pos.T)

	print("BFE acceleration:\n {} \n {} \n".format(a, a_gala))
	print("BFE density: ", rho, rho_gala)
	print("BFE potential: ", pot, pot_gala)
