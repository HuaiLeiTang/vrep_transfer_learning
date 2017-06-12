# This is a setup to control the Mico arm in V-REP
# 1. Obtain image from V-REP
# 2. Pass into neural network
# 3. Obtain joint velocities
# 4. Apply to arm in V-REP

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

import numpy as np
from keras.models import load_model
import imTransform

print ('Program started')
# just in case, close all opened connections
vrep.simxFinish(-1)

# Connect to V-REP
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID != -1:
    print ('Connected to remote API server')

    # enable the synchronous mode on the client:
    vrep.simxSynchronous(clientID, True)

    # start the simulation:
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

    # Load model
    model = load_model("trained_models/a1_normal_online_mixed_16_64batchsize.h5")

    # Get joint handles
    jhList = [-1, -1, -1, -1, -1, -1]
    for i in range(6):
        err, jh = vrep.simxGetObjectHandle(clientID, "Mico_joint"+str(i+1), vrep.simx_opmode_blocking)
        jhList[i] = jh

    # Initialize joint position
    jointpos = np.zeros(6)
    for i in range(6):
        err, jp = vrep.simxGetJointPosition(clientID, jhList[i], vrep.simx_opmode_streaming)
        jointpos[i] = jp

    # Initialize vision sensor
    res, v1 = vrep.simxGetObjectHandle(clientID, "vs1", vrep.simx_opmode_oneshot_wait)
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_streaming)
    vrep.simxGetPingTime(clientID)
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)

    # Initialize distance handle
    err, distanceHandle = vrep.simxGetDistanceHandle(clientID,"tipToCube",vrep.simx_opmode_blocking)
    err, distanceToCube = vrep.simxReadDistance(clientID,distanceHandle,vrep.simx_opmode_streaming)

    # Step while IK movement has not begun
    returnCode, signalValue = vrep.simxGetIntegerSignal(clientID,"ikstart",vrep.simx_opmode_streaming)
    while (signalValue==0):
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
        returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)

    # Iterate over total steps desired
    current_episode = 0
    total_episodes = 20
    step_counter = 0
    while current_episode < total_episodes+1:
        # obtain current episode
        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        err, episode_table, _, _, _ = vrep.simxCallScriptFunction(clientID, 'Mico',
                                                                  vrep.sim_scripttype_childscript,
                                                                    'episodeCount', inputInts,
                                                                  inputFloats, inputStrings,
                                                                  inputBuffer, vrep.simx_opmode_blocking)
        if episode_table[0]>current_episode:
            step_counter = 0
            print "Episode: ", episode_table[0]
        current_episode = episode_table[0]
        step_counter += 1


        # 1. Obtain image from vision sensor
        err, resolution, img = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        img = np.array(img)
        img = imTransform.gammaCorrection(img,gamma=1.0) #applying gamma correction to the image
        img = np.resize(img,[1,64,64,3]) # resize into proper shape for input to neural network
        img = img.astype('float32')
        img = img/255 # normalize input image

        # 2. Pass into neural network to get joint velocities
        jointvel = model.predict(img,batch_size=1)[0] #output is a 2D array of 1X6, access the first variable to get vector
        stepsize = 1
        jointvel *= stepsize

        # 3. Apply joint velocities to arm in V-REP
        for j in range(6):
            err, jp = vrep.simxGetJointPosition(clientID, jhList[j], vrep.simx_opmode_buffer)
            jointpos[j] = jp
            err = vrep.simxSetJointPosition(clientID, jhList[j], jointpos[j] + jointvel[j], vrep.simx_opmode_oneshot)

        # Obtain distance to cube
        err, distanceToCube = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_buffer)

        # Print statements
        print "Step: ", step_counter
        print "Joint velocities: ", jointvel, " Abs sum: ", np.sum(np.absolute(jointvel))
        print "Distance to cube: ", distanceToCube

        # trigger next step and wait for communication time
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)



    # obtain performance metrics
    inputInts = []
    inputFloats = []
    inputStrings = []
    inputBuffer = bytearray()
    err, minDistStep, minDist, _, _ = vrep.simxCallScriptFunction(clientID, 'Mico',
                                                                  vrep.sim_scripttype_childscript,
                                                                  'performanceMetrics', inputInts,
                                                                  inputFloats, inputStrings,
                                                                  inputBuffer, vrep.simx_opmode_blocking)

    if res == vrep.simx_return_ok:
        #print "Min distance steps: ", minDistStep
        #print "Min distance: ", minDist
        print "Total episodes: ", len(minDist)
        print "Average min distance: ", np.mean(minDist)
    # other performance metrics such as success % can be defined (i.e. % reaching certain min threshold)

    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)

    # delete model
    del model

else:
    print ('Failed connecting to remote API server')
print ('Program ended')
