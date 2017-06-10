import h5py
import numpy as np
import nn_architectures.architecture1 as nn

batch_size = 64
num_joints = 6
epochs = 2

train_path = "datasets/100iterations100steps64res_unitJointVel.hdf5"
test_path = "datasets/100iterations100steps64res_unitJointVel_validation.hdf5"
train_f = h5py.File(train_path, "r")
test_f = h5py.File(test_path, "r")
train_datapoints = train_f["images"].shape[0]
test_datapoints = test_f["images"].shape[0]

x_train = train_f["images"]
x_train = np.resize(x_train, [train_datapoints, 64, 64, 3])
y_train = train_f["joint_vel"]
x_test = test_f["images"]
x_test = np.resize(x_test, [test_datapoints, 64, 64, 3])
y_test = test_f["joint_vel"]

# Simple normalization of dividing all values by 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Initialize model with input shape and output size
model = nn.Model(x_train.shape[1:], num_joints)

model.compile()
model.fit(x_train=x_train, y_train=y_train,
          x_test=x_test, y_test=y_test,
          batch_size=batch_size, epochs=epochs)
model.plot()
model.save("trained_models/a1.h5", epochs)