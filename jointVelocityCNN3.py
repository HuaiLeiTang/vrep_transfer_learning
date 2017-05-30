# Training a CNN with o 80x64x64x3 images from vrep
# Checking if it can overfit well. Test set will be used
# MODEL IS TRAINED WITH STANDARDIZED OUTPUT AND NORMALIZED INPUT
# THEREFORE, AT PREDICT TIME, INPUT NEEDS TO BE NORMALIZED
# AND OUTPUT HAS TO BE INVERTED

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

batch_size = 64
num_joints = 6
epochs = 50
data_augmentation = False

# load train and test sets
trainset_name = "datasets/100iterations100steps64res_unitJointVel.hdf5"
testset_name = "datasets/100iterations100steps64res_unitJointVel_validation.hdf5"
train_f=h5py.File(trainset_name,"r")
test_f=h5py.File(testset_name,"r")
totalTrainDatapoints = train_f["images"].shape[0]
totalTestDatapoints = test_f["images"].shape[0]

x_train=train_f["images"]
x_train=np.resize(x_train, [totalTrainDatapoints, 64, 64, 3]) #saved data is flattened
y_train=train_f["joint_vel"]
x_test=test_f["images"]
x_test=np.resize(x_test, [totalTestDatapoints, 64, 64, 3])
y_test=test_f["joint_vel"]

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
## Model Start
model = Sequential()
# 2 conv + max pool layers
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=1,
                 padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=1,
                 padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 2 conv + max pool layers
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
                 padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
                 padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Fully connected
model.add(Flatten())
model.add(Dense(512,kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_joints))

# initiate adam optimizer
adam = keras.optimizers.adam(lr=0.001)

# Let's train the model using Adam optimizer
model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['mae']) #displayed loss is mean squared error

# Simple normalization of dividing all values by 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle="batch")
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

# save model and history
model_path = 'trained_models/newModelTest.h5'
model.save(model_path)
saved_model = h5py.File(model_path,"r+")
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