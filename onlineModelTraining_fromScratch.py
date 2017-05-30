# This is a setup to train the model by fitting on the image and path given by V-REP,
# BUT moves the arm according to the model's prediction
# At every step, the inverse kinematics path is recalculated. This will help the model correct its mistakes
# even when it goes off path into an unknown region.
# This is basically an online machine learning setting. One image and one label is given at each step.
# Since this is expected to be much slower, this training step should be seen as a finetuning step after
# the model is trained on a larger dataset of images.
# Sequence of events:
# Obtain image and path from V-REP -> Pass image into neural network -> Predict joint velocities
# -> Apply predicted velocities to arm in V-REP -> Fit model using image and path from V-REP -> Loop

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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
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

    # train a model from scratch using the online method
    inputShape = [64,64,3]
    num_joints = 6
    ## Model Start
    model = Sequential()
    # 2 conv + max pool layers
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1,
                     padding='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1,
                     padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 2 conv + max pool layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                     padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                     padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Fully connected
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_joints))

    # initiate adam optimizer
    adam = keras.optimizers.adam(lr=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mae'])  # displayed loss is mean squared error

    # Open file to get the standardized range
    #file = h5py.File("datasets/image100epochs50steps64res.hdf5")
    #file = h5py.File("datasets/singleEpochNoOffset.hdf5","r")

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

    # Iterate over number of steps to train for online model
    numberOfInputs = 5000
    for i in range(numberOfInputs):
        print "Step ", i
        #raw_input("Press Enter to continue...")
        # 1. Obtain image from vision sensor and next path position from Lua script
        err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        img = np.resize(image,[1,64,64,3]) # resize into proper shape for input to neural network
        img = img.astype('float32')
        img = img/255 # normalize input image

        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        err, retInts, nextPathPos, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, 'Mico',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'getNextPathPos', inputInts,
                                                                                     inputFloats, inputStrings,
                                                                                     inputBuffer,
                                                                                     vrep.simx_opmode_blocking)


        # 2. Pass into neural network to get joint velocities
        jointvel = model.predict(img,batch_size=1)[0] #output is a 2D array of 1X6, access the first variable to get vector
        #print "Predicted joint velocities: ", jointvel
        #print "Absolute sum: ", np.sum(np.absolute(jointvel))
        stepsize = 1
        jointvel *= stepsize

        ## Invert joint velocities
        # scaler = StandardScaler()
        # scaler = scaler.fit(file["joint_vel"])
        # jointvel = scaler.inverse_transform(jointvel)
        # print "Joint velocities after inverting: ", jointvel

        # 3. Apply joint velocities to arm in V-REP
        for j in range(6):
            err, jp = vrep.simxGetJointPosition(clientID, jhList[j], vrep.simx_opmode_buffer)
            jointpos[j] = jp
            err = vrep.simxSetJointPosition(clientID, jhList[j], jointpos[j] + jointvel[j], vrep.simx_opmode_oneshot)

        #err, distanceToCube = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_buffer)
        #print "Distance to cube: ", distanceToCube

        # 4. Fit model
        ik_jointvel = nextPathPos - jointpos
        ik_jointvel = ik_jointvel/np.sum(np.absolute(ik_jointvel))/10
        ik_jointvel = np.resize(ik_jointvel,(1,6))
        #print "IK Joint velocity: ", ik_jointvel
        #print "Sum: ", np.sum(np.absolute(ik_jointvel))
        model.fit(img, ik_jointvel,
                  batch_size=1,
                  epochs=1)

        # trigger next step and wait for communication time
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)

    # save updated model delete model and close h5py file
    model.save("trained_models/onlineModel_fromScratch.h5")
    del model
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
