import numpy as np
from scipy import special



def derivative_one(l, m, theta):
    Ylm = special.sph_harm(m, l, 0, theta).real # check -1 in m-1
    Ylm_1 = special.sph_harm(m+1, l, 0, theta).real # check -1 in m-1
    
    dpot_dtheta_1st = m*Ylm / np.tan(theta)
    dpot_dtheta_2nd = np.sqrt((l-m) * (l+m+1)) * Ylm_1
    dpot_dtheta = dpot_dtheta_1st + dpot_dtheta_2nd 
    return dpot_dtheta


def derivative_two(l, m, theta):
    Ylm = special.lpmv(m, l, np.cos(theta)) # check -1 in m-1
    Ylm_1 = special.lpmv(m, l-1, np.cos(theta)) # check -1 in m-1
    
    factor = special.gamma(l-m+1)**0.5 / special.gamma(l+m+1)**0.5
    constant = ((2*l+1)/(4*np.pi))**0.5
    return constant * factor * (l*np.cos(theta) * Ylm - (l+m)*Ylm_1) / np.sin(theta)



if __name__ == "__main__":
    l = np.ones(2)*5
    m = np.ones(2)*2

    theta = np.pi/4.
    dev1 = derivative_one(l, m, theta)
    dev2 = derivative_two(l, m, theta)
    print(dev1, dev2)
