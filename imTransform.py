import matplotlib.pyplot as mlp
import numpy as np
import h5py
import cv2

def averageBlur(img,kernel_width=5,kernel_height=5):
    return cv2.blur(img,(kernel_width,kernel_height))

def gammaCorrection(img,gamma=1.0):
    img=img.astype('uint8')
    img=adjust_gamma(img,gamma)
    return img

def displayAfterGammaCorrection(img,gamma=1.0):
    img=gammaCorrection(img,gamma)
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
    img=img.astype(np.uint8)
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

def flattenImage(img):
    return np.resize(img,img.size)


def addSPNoise(img, amount=0.01, sp_ratio=0.5):
    # assumes image pixel values are between 0 and 255
    # this version does not apply to all 3 channels, hence result will have different colors
    img = flattenImage(img)
    salt = np.ceil(amount * img.size * sp_ratio)
    s_indices = np.random.randint(img.size,size=int(salt))
    img[s_indices] = 255
    pepper = np.ceil(amount * img.size * (1.0 - sp_ratio))
    p_indices = np.random.randint(img.size,size=int(pepper))
    img[p_indices] = 0
    return img

def addSPNoiseForChannels(img, shape, amount=0.01, sp_ratio=0.5):
    # assumes shape is 3 dimensional(width*height*channel)
    img = flattenImage(img)
    resolution = shape[0]*shape[1]
    channels = shape[2]
    salt = np.ceil(amount * resolution * sp_ratio)
    s_indices = np.random.randint(resolution,size=int(salt))
    for i in range(shape[2]):
        img[s_indices*channels+i] = 255
    pepper = np.ceil(amount * resolution * (1.0 - sp_ratio))
    p_indices = np.random.randint(resolution,size=int(pepper))
    for i in range(shape[2]):
        img[p_indices*channels+i] = 0
    return img

## test case
# file = h5py.File("datasets/singleEpochNoOffset.hdf5","r")
# images=file["images"]
# sp_im=addSPNoiseForChannels(images[0],[64,64,3])