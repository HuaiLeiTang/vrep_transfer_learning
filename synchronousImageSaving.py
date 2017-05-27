# This is a script to obtain images, joint positions, and joint velocities
# from V-REP and save them into a hdf5 file format

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

import h5py
import numpy as np


print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
    print ('Connected to remote API server')

    # enable the synchronous mode on the client:
    vrep.simxSynchronous(clientID,True)

    # start the simulation:
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)

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

    # Initialize data file
    filename = "datasets/100iterations100steps64res_unitJointVel_validation.hdf5"
    f = h5py.File(filename, "w")
    numberOfIterations = 10
    iterationCounter = 0
    numberOfSteps = 100  # number of steps per epoch
    totalDatapoints = numberOfIterations * numberOfSteps
    sizeOfImage = resolution[0] * resolution[1] * 3  # number of pixels multiplied by 3 channels (RGB)
    dset1 = f.create_dataset("images", (totalDatapoints, sizeOfImage), dtype="uint")
    dset2 = f.create_dataset("joint_pos", (totalDatapoints, 6), dtype="float")
    dset3 = f.create_dataset("joint_vel", (totalDatapoints, 6), dtype="float")

    #Step while IK movement has not begun
    returnCode, signalValue = vrep.simxGetIntegerSignal(clientID,"ikstart",vrep.simx_opmode_streaming)
    while (signalValue==0):
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
        returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)

    # Iterate over all inverse kinematic steps:
    for i in range(numberOfSteps*numberOfIterations):
        if (i % numberOfSteps == 0):
            returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)
            while (signalValue == 0):
                vrep.simxSynchronousTrigger(clientID)
                vrep.simxGetPingTime(clientID)
                returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)
            iterationCounter += 1
        print "Iteration ", iterationCounter
        print "Step ", i
        err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        img = np.array(image, dtype=np.uint8)
        dset1[i] = img
        jointpos = np.zeros(6)
        for j in range(6):
            err, jp = vrep.simxGetJointPosition(clientID, jhList[j], vrep.simx_opmode_buffer)
            jointpos[j] = jp
        dset2[i]=jointpos

        # trigger next step and wait for communication time
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)

    # calculate joint velocities excluding final image
    for k in range(numberOfIterations):
        for i in range(numberOfSteps-1):
            jointvel = dset2[k*numberOfSteps + i+1]-dset2[k*numberOfSteps + i]
            abs_sum = np.sum(np.absolute(jointvel))
            if abs_sum==0:
                dset3[k * numberOfSteps + i] = np.zeros(6)
            else:
                dset3[k * numberOfSteps + i] = jointvel/abs_sum/10

        # set final velocity to zeros
        dset3[(k+1)*numberOfSteps - 1] = np.zeros(6)

        # close datafile
    f.close()
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
