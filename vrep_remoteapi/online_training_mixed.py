import vrep
from vrep_utils import VrepConnection
import keras
from keras.models import load_model
import numpy as np
import h5py
import random

def train(load_model_path, save_model_path, number_training_steps,
          data_path, ratio=1, batch_size=2):
    # Initialize connection
    connection = VrepConnection()
    connection.synchronous_mode()
    connection.start()

    # Load data
    data_file = h5py.File(data_path,"r")
    x = data_file['images']
    y = data_file['joint_vel']
    datapoints = x.shape[0]
    x = np.resize(x, [datapoints, 64, 64, 3])
    x = x.astype('float32')
    x /= 255

    # Initialize ratios
    online_batchsize = int(np.floor(1.0 * batch_size/(ratio + 1)))
    data_images_batchsize = int(batch_size - online_batchsize)
    current_online_batch = 0
    x_batch = np.empty([batch_size, 64, 64, 3])
    y_batch = np.empty([batch_size, 6])
    print "Batch size: ", batch_size
    print "Online batch size: ", online_batchsize
    print "Dataset batch size: ", data_images_batchsize

    # Load keras model
    model = load_model(load_model_path)

    # Initialize adam optimizer
    adam = keras.optimizers.adam(lr=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mae'])

    # Use client id from connection
    clientID = connection.clientID

    # Get joint handles
    jhList = [-1, -1, -1, -1, -1, -1]
    for i in range(6):
        err, jh = vrep.simxGetObjectHandle(clientID, "Mico_joint" + str(i + 1), vrep.simx_opmode_blocking)
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
    err, distanceHandle = vrep.simxGetDistanceHandle(clientID, "tipToCube", vrep.simx_opmode_blocking)
    err, distanceToCube = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_streaming)

    # Step while IK movement has not begun
    returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)
    while (signalValue == 0):
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
        returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)

    # Iterate over number of steps to train model online
    path_empty_counter = 0
    step_counter = 0
    while step_counter < number_training_steps:
        # 1. Obtain image from vision sensor and next path position from Lua script
        err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        img = np.resize(image, [1, 64, 64, 3])  # resize into proper shape for input to neural network
        img = img.astype('float32')
        img = img / 255  # normalize input image

        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        err, _, next_pos, _, _ = vrep.simxCallScriptFunction(clientID, 'Mico',
                                                             vrep.sim_scripttype_childscript,
                                                             'getNextPathPos', inputInts,
                                                             inputFloats, inputStrings,
                                                             inputBuffer,
                                                             vrep.simx_opmode_blocking)

        # 2. Pass into neural network to get joint velocities
        jointvel = model.predict(img, batch_size=1)[0]  # output is a 2D array of 1X6, access the first variable to get vector
        stepsize = 1
        jointvel *= stepsize

        # 3. Apply joint velocities to arm in V-REP
        for j in range(6):
            err, jp = vrep.simxGetJointPosition(clientID, jhList[j], vrep.simx_opmode_buffer)
            jointpos[j] = jp
            err = vrep.simxSetJointPosition(clientID, jhList[j], jointpos[j] + jointvel[j], vrep.simx_opmode_oneshot)

        err, distanceToCube = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_buffer)

        # Check if next pos is valid before fitting
        if len(next_pos) != 6:
            path_empty_counter += 1
            continue

        ik_jointvel = next_pos - jointpos
        ik_jointvel = ik_jointvel / np.sum(np.absolute(ik_jointvel)) / 10
        ik_jointvel = np.resize(ik_jointvel, (1, 6))
        x_batch[current_online_batch] = img[0]
        y_batch[current_online_batch] = ik_jointvel[0]
        current_online_batch += 1

        if current_online_batch==online_batchsize:
            # Step counter
            print "Training step: ", step_counter
            step_counter += 1

            # Random sample from dataset
            indices = random.sample(range(int(datapoints)), int(data_images_batchsize))
            indices = np.sort(indices)
            indices = list(indices)
            x_batch[online_batchsize:] = x[indices]
            y_batch[online_batchsize:] = y[indices]

            # 4. Fit model
            model.fit(x_batch, y_batch,
                      batch_size=batch_size,
                      epochs=1)

            # Reset counter
            current_online_batch = 0

        # Print statements
        # print "Predicted joint velocities: ", jointvel, "Abs sum: ", np.sum(np.absolute(jointvel))
        # print "IK Joint velocity: ", ik_jointvel, "Abs sum: ", np.sum(np.absolute(ik_jointvel))
        # print "Distance to cube: ", distanceToCube

        # trigger next step and wait for communication time
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)

    # save updated model and delete model
    model.save(save_model_path)
    del model

    # Error check
    print "No of times path is empty:", path_empty_counter