import numpy as np
import pygadgetreader 
import sys

def get_particle_id(pos, vel, ids, r_lim, dr, v_lim, dv):
    r = np.sqrt(np.sum(pos**2, axis=1))
    v = np.sqrt(np.sum(vel**2, axis=1))
    index = np.where((r<r_lim+dr) & (r>r_lim-dr) & (v<v_lim+dv) & (v>v_lim-dv))
    print(len(index[0]))
    return ids[index]


def extract_orbit(snap, id_p, i):
    pos = pygadgetreader.readsnap(snap+'_{:0>3d}'.format(i), 'pos', 'dm')
    ids = pygadgetreader.readsnap(snap+'_{:0>3d}'.format(i), 'pid', 'dm')
    particles = np.in1d(ids, id_p)
    pos_orbit = pos[particles]
    return pos_orbit

def extract_all_orbits(snap, snap_i, snap_f, ids_p):
    N_snaps = snap_f - snap_i +1 
    N_part = len(ids_p)
    pos_orbits = np.zeros((N_snaps, N_part, 3))
    j=0
    for i in range(snap_i, snap_f+1):
        pos_orbits[j] = extract_orbit(snap, ids_p, i)
        j+=1
    return pos_orbits
    
def out_orbits(out_name, snap, snap_i, snap_f, ids_p):
    N_part = len(ids_p)
    all_pos = extract_all_orbits(snap, snap_i, snap_f, ids_p)
    for i in range(N_part+1):
        np.savetxt(out_name+"_particle_{:0>3d}.txt".format(i), all_pos[:,i,:])
        
if __name__ == "__main__":
    ids_file = argv[1]
    snaps_file = argv[2]
    out_path = argv[3]
    out_name = argv[4]
    snap_i = int(argv[5])
    snap_f = int(argv[6])
    ids_all = np.loadtxt(ids_file)
    out_orbits(out_path+out_name, snaps_file, snap_i, snap_f, ids_all)
