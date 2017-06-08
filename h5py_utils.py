import h5py

def open_r_mode(path):
    return h5py.File(path,"r")

def open_rplus_mode(path):
    return h5py.File(path,"r+")