import matplotlib.pyplot as mlp
import numpy as np
import h5py
import time
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

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

def displayInfoAndPredict(filename, pos, modelname):
    file = h5py.File(filename, "r")
    dset1 = file["images"]
    img = dset1[pos]
    img.resize([64, 64, 3])
    img = img.astype(np.uint8)
    mlp.imshow(img, origin="lower")
    dset2 = file["joint_pos"]
    dset3 = file["joint_vel"]
    model = load_model(modelname)
    img = np.resize(img, [1, 64, 64, 3])
    pred_joint_vel = model.predict(img,batch_size=1)[0]
    print "Joint pos: ", dset2[pos]
    print "Joint vel: ", dset3[pos]
    print "Predicted joint vel: ", pred_joint_vel
    file.close()
    return

def displayInfoAndInvertPredict(filename, pos, modelname):
    file = h5py.File(filename, "r")
    dset1 = file["images"]
    img = dset1[pos]
    img.resize([64, 64, 3])
    img = img.astype(np.uint8)
    mlp.imshow(img, origin="lower")
    dset2 = file["joint_pos"]
    dset3 = file["joint_vel"]
    model = load_model(modelname)
    img = np.resize(img, [1, 64, 64, 3])
    pred_joint_vel = model.predict(img,batch_size=1)[0]
    scaler = StandardScaler()
    scaler = scaler.fit(file["joint_vel"])
    pred_joint_vel = scaler.inverse_transform(pred_joint_vel)
    print "Joint pos: ", dset2[pos]
    print "Joint vel: ", dset3[pos]
    print "Predicted joint vel: ", pred_joint_vel
    file.close()
    return


def loopDisplay(numberOfImages,filename):
    for i in range(numberOfImages):
        displayInfo(filename,i)
        time.sleep(0.5)
    return