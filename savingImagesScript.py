import h5py
import numpy as np

filename = "filename"
f = h5py.File(filename,"r+") #file must exist for r+

numberOfInputs = 100 #set number of images to save
sizeOfImage = 128*128*3 #number of pixels
dset = f.create_dataset("images",(numberOfInputs,sizeOfImage),dtype='float')

for i in range(numberOfInputs):
    ##get image array from remoteAPI
    dset[i] = imageArray