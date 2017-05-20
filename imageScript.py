import vrep
import sys
import numpy as np
import matplotlib.pyplot as mlp


vrep.simxFinish(-1) # just in case, closes all opened connections

clientID = vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:
    print 'Connected to remote API server'
    res, v1 = vrep.simxGetObjectHandle(clientID, "vs1", vrep.simx_opmode_oneshot_wait)
    print 'Getting first image'
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_streaming)
    while (vrep.simxGetConnectionId(clientID) != -1):
        err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        print image
        if err == vrep.simx_return_ok:
            print "image OK"
            img = np.array(image,dtype=np.uint8)
            img.resize([resolution[1],resolution[0],3])
            mlp.imshow(img)

    vrep.simxGetPingTime(clientID)
    vrep.simxFinish(clientID)
else:
    print 'Connection not successful'
    sys.exit('Could not connect')



