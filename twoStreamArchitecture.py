# Two stream architecture adapted from 'Beyond Sharing Weight for Deep Domain Adaptation', Rozantsev et al.

from __future__ import print_function
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
import h5py
import numpy as np
import matplotlib.pyplot as plt
import fn
import imTransform

batch_size = 64
num_joints = 6
epochs = 1
data_augmentation = False

# load train and test sets
trainset_name = "datasets/100iterations100steps64res_unitJointVel.hdf5"
testset_name = "datasets/100iterations100steps64res_unitJointVel_validation.hdf5"
train_f=h5py.File(trainset_name,"r")
test_f=h5py.File(testset_name,"r")
totalTrainDatapoints = train_f["images"].shape[0]
totalTestDatapoints = test_f["images"].shape[0]

source_train=train_f["images"]
source_train=np.resize(source_train, [totalTrainDatapoints, 64, 64, 3]) #saved data is flattened
source_train_labels=train_f["joint_vel"]
source_test=test_f["images"]
source_test=np.resize(source_test, [totalTestDatapoints, 64, 64, 3])
source_test_labels=test_f["joint_vel"]

# Create target data by applying some distortion to source data
# Gamma correction is used in this case
target_train=train_f["images"]
target_train=imTransform.gammaCorrection(target_train[:target_train.shape[0]],gamma=1.3)
target_train=np.resize(target_train, [totalTrainDatapoints, 64, 64, 3])
target_train_labels=train_f["joint_vel"]

# Simple normalization of dividing all values by 255
source_train = source_train.astype('float32')
source_test = source_test.astype('float32')
target_train = target_train.astype('float32')
source_train /= 255
source_test /= 255
target_train /= 255

### Step 1 Start ###
## Step 1: Train a model on SOURCE data using source task loss only

inputs = Input(shape=source_train.shape[1:])
# 2 conv + maxpool + dropout layers
x = Conv2D(filters=16, kernel_size=(3,3), strides=1,
           padding='same')(inputs)
#x = Activation('relu')(x)
x = ELU()(x)
x = Conv2D(filters=16, kernel_size=(3,3), strides=1,
           padding='same')(x)
#x = Activation('relu')(x)
x = ELU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
# 2 conv + maxpool + dropout layers
x = Conv2D(filters=16, kernel_size=(3,3), strides=1,
           padding='same')(x)
#x = Activation('relu')(x)
x = ELU()(x)
x = Conv2D(filters=16, kernel_size=(3,3), strides=1,
           padding='same')(x)
#x = Activation('relu')(x)
x = ELU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
# flatten and one fully connected layer
x = Flatten()(x)
x = Dense(512,kernel_regularizer=keras.regularizers.l2(0.01))(x)
#x = Activation('relu')(x)
feature_layer = ELU(name='feature_layer')(x)
x = Dropout(0.5)(feature_layer)
predictions = Dense(num_joints,name='predictions')(x)

# initiate adam optimizer
adam = keras.optimizers.adam(lr=0.001)

source_model = Model(inputs=inputs, outputs=[predictions,feature_layer])
source_model.compile(loss={'predictions': 'mean_squared_error','feature_layer': fn.empty},
                     optimizer=adam,
                     metrics=['mae'],
                     loss_weights=[1,1])

placeholder_labels = np.zeros((10000,512))
history = source_model.fit(source_train, [source_train_labels, placeholder_labels],
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(source_test, [source_test_labels,placeholder_labels[:1000]]),
                           shuffle="batch")

# save model and history
model_path = 'trained_models/source_model.h5'
source_model.save(model_path)
saved_model = h5py.File(model_path,"r+")
for key in history.history.keys():
    dset = saved_model.create_dataset("history/"+key,(epochs,),dtype='f')
    dset[...]=history.history[key]
saved_model.close()

## plot histories
# # mean absolute error graph
# plt.plot(history.history['mean_absolute_error'])
# plt.plot(history.history['val_mean_absolute_error'])
# plt.title('Model MAE')
# plt.ylabel('mean absolute error')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()
# # loss graph
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()
### Step 1 End ###

### Step 2 Start ###
## Step 2: Load model trained in step 1 to be used as initial start point for training on target domain
## If labelled, use target task loss and MMD
## else if unlabelled, use MMD only
#target_model = load_model(model_path)
target_model = load_model('trained_models/source_model.h5',custom_objects={'empty': fn.empty})
layer_index = 15 #layer index has to be manually specified

# these two lines consumes quite a bit of memory
# and also it takes pretty long to compute them
#target_feature_layer = fn.get_activations(target_model, layer_index, target_train)
source_feature_layer = fn.get_activations(source_model, layer_index, source_train)

# only difference is the loss in the compile stage
# For labelled target data
target_model.compile(loss={'predictions': 'mean_squared_error',
                           'feature_layer':fn.rbf_mmd2_features(source_feature_layer)},
                     optimizer=adam,
                     metrics=['mse','mae'])
target_model.fit(target_train, target_train_labels,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(target_train, target_train_labels),
                 shuffle='batch')
# For unlabelled target data
# target_model.compile(loss=fn.rbfmmd2_features(target_feature_layer,source_feature_layer),
#                      optimizer=adam,
#                      metrics=['mse','mae'])
# target_model.fit(target_train, target_train_labels,
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  validation_data=(target_train, target_train_labels),
#                  shuffle='batch')

### Step 2 End ###