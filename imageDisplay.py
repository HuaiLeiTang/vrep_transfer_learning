import matplotlib.pyplot as mlp
import numpy as np
import h5py
import time

def displayInfo(filename,pos):
    file=h5py.File(filename,"r")
    dset1=file["images"]
    img=dset1[pos]
    img.resize([64,64,3])
    img=img.astype(np.uint8)
    mlp.imshow(img,origin="lower")
    dset2=file["joint_pos"]
    dset3=file["joint_vel"]
    print "Joint pos: ",dset2[pos]
    print "Joint vel: ",dset3[pos]
    file.close()
    return


def loopDisplay(numberOfImages,filename):
    for i in range(numberOfImages):
        displayInfo(filename,i)
        time.sleep(0.5)
    return