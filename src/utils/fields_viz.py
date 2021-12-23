import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

# BFE fields code
import bfe_fields


def re_center_halo(pos, r_cm):
    """
    Re-center a halo positions or velocities.
    """

    print('COM coordinates', r_cm)
    for i in range(3):
        pos[:,i] = pos[:,i] - r_cm[i]
    return pos

def eval_2d_fields(coefficients,  bfe_params, grid_size, field, **kwargs):
    """
    Computes a BFE field in a grid

    Parameters:

    # TODO:
    - Allow for an off-cetner grid.
    - Generalize the plane of the grid to be in either x, y, or z

    """

    Snlm, Tnlm = coefficients
    M, rs, nmax, lmax, G = bfe_params
    xmin, xmax, ymin, ymax, nbinsx, nbinsy, z_plane = grid_size

    y_bins = np.linspace(xmin, xmax, nbinsx)
    z_bins = np.linspace(ymin, ymax, nbinsy)
    y, z = np.meshgrid(y_bins, z_bins)

    pos = np.array([np.ones(len(y.flatten()))*z_plane, y.flatten(), z.flatten()]).T

    halo_field = bfe_fields.BFE_fields(pos, Snlm, Tnlm, rs, nmax, lmax, G, M)

    if field == "density":
        twod_field = halo_field.bfe_density()
    elif field == "potential":
        twod_field = halo_field.bfe_potential()
    elif field == "acceleration":
        acc = halo_field.bfe_acceleration()
        twod_field = np.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2)
    else:
        print("Selected field not availble")
    #if 'S2' in kwargs.keys():
    #    S2 = kwargs['S2']
   # Reshape to get the right shape
   # TODO check why this shape -> flatten!
    return  twod_field.reshape((nbinsy, nbinsx))

def viz_all_fields(all_fields, grid_size, title_names=["Density", "Potential", "Acceleration"]):
    """
    Plots all the fields in 2d
    """
    k = len(all_fields)
    cmaps = ["viridis_r", "cividis_r", "inferno"]
    xmin, xmax, ymin, ymax = grid_size

    fig, ax = plt.subplots(1, k, figsize=(4*k, 4), sharey=True)

    for i in range(k):
        ax[i].imshow(all_fields[i].T, extent=[xmin, xmax, ymin, ymax], cmap=cmaps[i])
        ax[i].set_title(title_names[i], fontsize=18)
    return fig, ax
