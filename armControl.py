# This is a setup to control the Mico arm in V-REP
# Obtain image from V-REP -> Pass into neural network -> Obtain joint velocities -> Apply to arm in V-REP

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import sys
import h5py
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
    print ('Connected to remote API server')

    # enable the synchronous mode on the client:
    vrep.simxSynchronous(clientID,True)

    # start the simulation:
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)

    # Load keras model
    #model = load_model("trained_models/model_100epochs50steps64res.h5")
    model = load_model("trained_models/model_singleEpochNoRandomOffsets.h5")

    # Open file to get the standardized range
    file = h5py.File("datasets/singleEpochNoOffset.hdf5","r")

    # Get joint handles
    jhList = [-1, -1, -1, -1, -1, -1]
    for i in range(6):
        err, jh = vrep.simxGetObjectHandle(clientID, "Mico_joint"+str(i+1), vrep.simx_opmode_blocking)
        print err
        jhList[i] = jh
    print "Joints handles: ", jhList
    jointpos = np.zeros(6)
    for i in range(6):
        err, jp = vrep.simxGetJointPosition(clientID, jhList[i], vrep.simx_opmode_streaming)
        jointpos[i] = jp
    print jointpos

    # Initialize vision sensor
    res, v1 = vrep.simxGetObjectHandle(clientID, "vs1", vrep.simx_opmode_oneshot_wait)
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_streaming)
    vrep.simxGetPingTime(clientID)
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
    print resolution



    # Get distance handle
    err, distanceHandle = vrep.simxGetDistanceHandle(clientID,"tipToCube",vrep.simx_opmode_blocking)
    err, distanceToCube = vrep.simxReadDistance(clientID,distanceHandle,vrep.simx_opmode_streaming)
    print "Initial distance to cube: ", distanceToCube

    #Step while IK movement has not begun
    returnCode, signalValue = vrep.simxGetIntegerSignal(clientID,"ikstart",vrep.simx_opmode_streaming)

    while (signalValue==0):
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
        returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)

    # Iterate over number of steps in training data generated
    numberOfInputs = 50
    for i in range(numberOfInputs):
        print "Step ", i
        # 1. Obtain image from vision sensor
        err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        img = np.resize(image,[1,64,64,3]) # resize into proper shape for input to neural network
        img = img.astype('float32')
        img = img/255

        # 2. Pass into neural network to get joint velocities
        jointvel = model.predict(img,batch_size=1)[0] #output is a 2D array of 1X6, access the first variable to get vector
        print "Joint velocities: ", jointvel

        ## Invert joint velocities
        scaler = StandardScaler()
        scaler = scaler.fit(file["joint_vel"])
        jointvel = scaler.inverse_transform(jointvel)
        print "Joint velocities after inverting: ", jointvel

        # 3. Apply joint velocities to arm in V-REP
        # using random joint velocities for testing now
        for j in range(6):
            err, jp = vrep.simxGetJointPosition(clientID, jhList[j], vrep.simx_opmode_buffer)
            jointpos[j] = jp
        # jointvel = (np.random.rand(6)-0.5)/10

        for j in range(6):
            err = vrep.simxSetJointPosition(clientID,jhList[j],jointpos[j]+jointvel[j],vrep.simx_opmode_oneshot)

        err, distanceToCube = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_buffer)
        print "Distance to cube: ", distanceToCube

        # trigger next step and wait for communication time
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    # final distance to cube used as a measurement of how well the neural network learnt
    err, distanceToCube = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_buffer)
    print "Final distance to cube: ", distanceToCube

    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)

    # delete model
    del model
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
