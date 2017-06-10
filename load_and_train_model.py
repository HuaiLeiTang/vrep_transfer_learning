import keras_utils as ku
import h5py_utils as hu
import numpy as np
import matplotlib.pyplot as plt

# load train and test sets
trainset_name = "datasets/100iterations100steps64res_unitJointVel.hdf5"
testset_name = "datasets/100iterations100steps64res_unitJointVel_validation.hdf5"
train_f = hu.open_r_mode(trainset_name)
test_f = hu.open_r_mode(testset_name)
totalTrainDatapoints = train_f["images"].shape[0]
totalTestDatapoints = test_f["images"].shape[0]

x_train = train_f["images"]
x_train = np.resize(x_train, [totalTrainDatapoints, 64, 64, 3]) #saved data is flattened
y_train = train_f["joint_vel"]
x_test = test_f["images"]
x_test = np.resize(x_test, [totalTestDatapoints, 64, 64, 3])
y_test = test_f["joint_vel"]

# Simple normalization of dividing all values by 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# load your model
model_path = "trained_models/elu_noMaxPool_20epochs.h5"
model = ku.load(model_path)

# Some hyperparameters
batch_size = 64
epochs = 30

# fit your model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle="batch")

# save model and history
model_path = 'trained_models/elu_noMaxPool_21_50.h5'
model.save(model_path)
saved_model = hu.open_rplus_mode(model_path)
for key in history.history.keys():
    dset = saved_model.create_dataset("history/"+key,(epochs,),dtype='f')
    dset[...]=history.history[key]
saved_model.close()

## plot histories
# mean absolute error graph
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model MAE')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()