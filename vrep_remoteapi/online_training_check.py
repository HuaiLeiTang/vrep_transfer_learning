import vrep
from vrep_utils import VrepConnection
import keras
from keras.models import load_model
import numpy as np
import h5py
import random
from image_utils.methods import h5py_to_array, display
import matplotlib.pyplot as plt
import math

def joint_difference(initial_pos, end_pos):
    joint_vel = np.empty(6)
    for i in range(6):
        difference = end_pos[i] - initial_pos[i]
        if difference > math.pi:
            joint_vel[i] = difference - 2 * math.pi
        elif difference < -math.pi:
            joint_vel[i] = difference + 2 * math.pi
        else:
            joint_vel[i] = difference
    return joint_vel


def train(load_model_path, save_model_path, number_training_steps,
          data_path, image_shape, ratio=1, batch_size=2,
          save_online_batch=False, save_online_batch_path=None):
    # Initialize connection
    connection = VrepConnection()
    connection.synchronous_mode()
    connection.start()

    # Load data
    data_file = h5py.File(data_path,"r")
    x = h5py_to_array(data_file['images'][:1000], image_shape)
    y = data_file['joint_vel'][:1000]
    datapoints = x.shape[0]

    # Initialize ratios
    online_batchsize = int(np.floor(1.0 * batch_size/(ratio + 1)))
    data_images_batchsize = int(batch_size - online_batchsize)
    current_online_batch = 0
    x_batch = np.empty((batch_size,) + image_shape)
    y_batch = np.empty((batch_size, 6))
    jointpos_array = np.empty((batch_size, 6))
    nextpos_array = np.empty((batch_size, 6))
    print "Batch size: ", batch_size
    print "Online batch size: ", online_batchsize
    print "Dataset batch size: ", data_images_batchsize

    # Load keras model
    model = load_model(load_model_path)

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
    err, distanceHandle = vrep.simxGetDistanceHandle(clientID, "tipToTarget", vrep.simx_opmode_blocking)
    err, distance_to_target = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_streaming)

    # Initialize online batch hdf5 file
    if save_online_batch:
        f = h5py.File(save_online_batch_path, "w")
        dset1 = f.create_dataset("images", (batch_size,) + image_shape, dtype=np.float32)
        dset2 = f.create_dataset("joint_pos", (batch_size, 6), dtype=np.float32)
        dset3 = f.create_dataset("next_pos", (batch_size, 6), dtype=np.float32)
        dset4 = f.create_dataset("distance", (batch_size, 1), dtype=np.float32)
        dset5 = f.create_dataset("joint_vel", (batch_size, 6), dtype=np.float32)

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
        img = np.resize(image, (1,) + image_shape)  # resize into proper shape for input to neural network
        img = img.astype(np.uint8)
        img = img.astype(np.float32)
        img /= 255

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

        err, distance_to_target = vrep.simxReadDistance(clientID, distanceHandle, vrep.simx_opmode_buffer)

        # Check if next pos is valid before fitting
        if len(next_pos) != 6:
            path_empty_counter += 1
            continue


        ik_jointvel = joint_difference(jointpos, next_pos)
        ik_jointvel = ik_jointvel / np.sum(np.absolute(ik_jointvel)) * 0.5 * distance_to_target
        ik_jointvel = np.resize(ik_jointvel, (1, 6))
        x_batch[current_online_batch] = img[0]
        y_batch[current_online_batch] = ik_jointvel[0]
        jointpos_array[current_online_batch] = jointpos
        nextpos_array[current_online_batch] = next_pos
        dset4[current_online_batch] = distance_to_target
        current_online_batch += 1


        if current_online_batch == online_batchsize:
            # Step counter
            print "Training step: ", step_counter
            step_counter += 1

            # Random sample from dataset
            if ratio > 0:
                indices = random.sample(range(int(datapoints)), int(data_images_batchsize))
                indices = sorted(indices)
                x_batch[online_batchsize:] = x[indices]
                y_batch[online_batchsize:] = y[indices]

                dset4[online_batchsize:] = data_file["distance"][indices]
                dset2[online_batchsize:] = data_file["joint_pos"][indices]

            # 4. Fit model
            model.fit(x_batch, y_batch,
                      batch_size=batch_size,
                      epochs=1)

            # Reset counter
            current_online_batch = 0

            # Save to online batch dataset
            dset1[:] = x_batch
            dset5[:] = y_batch
            dset2[:online_batchsize] = jointpos_array[:online_batchsize]
            dset3[:] = nextpos_array



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

    # Close file
    if save_online_batch:
        f.close()

    # Error check
    print "No of times path is empty:", path_empty_counter

if __name__ == "__main__":
    train(load_model_path="../models/a6_d1_v3.h5",
          save_model_path="../models/a6_d1_online.h5",
          number_training_steps=1,
          data_path="../datasets/d1.hdf5",
          image_shape=(128, 128, 3),
          batch_size=128,
          save_online_batch=True,
          save_online_batch_path="../datasets/online_batch.hdf5")
