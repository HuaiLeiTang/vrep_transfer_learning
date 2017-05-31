import matplotlib.pyplot as mlp
import numpy as np
import h5py
import cv2

def displayAfterGammaCorrection(img,gamma=1.0):
    img=img.astype('uint8')
    img=adjust_gamma(img,gamma)
    print "After gamma correct, gamma=", gamma
    displayFlattenedImage(img)
    return

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def displayFlattenedImage(img):
    img.resize([64,64,3])
    mlp.imshow(img,origin="lower")
    return

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
    print "Joint vel sum: ",np.sum(np.absolute(dset3[pos]))
    joint_vel = dset3[pos]/np.sum(np.absolute(dset3[pos]))
    print "Transformed joint vel: ", joint_vel
    print "Sum: ", np.sum(np.absolute(joint_vel))
    file.close()
    return





