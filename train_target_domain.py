from __future__ import print_function
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
import h5py
import numpy as np
import matplotlib.pyplot as plt
from image_utils.methods import *
import random
from custom_loss_functions.empty import *
from custom_loss_functions.mmd import *
import keras.backend as K


def get_activations(model, layer, X_batch):
    activations_fn = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = activations_fn([X_batch, 0])
    return activations

batch_size = 128
num_joints = 6
data_augmentation = False

# load train and test sets
trainset_name = "datasets/d1.hdf5"
testset_name = "datasets/d1_test.hdf5"
image_shape = (128, 128, 3)
train_f = h5py.File(trainset_name,"r")
test_f = h5py.File(testset_name,"r")
source_train_datapoints = train_f["images"].shape[0]
test_datapoints = test_f["images"].shape[0]

# Initialize train images
train_images = train_f["images"]
train_images = h5py_to_array(train_images, image_shape)
source_train = train_images

# Initialize test images
test_images = test_f["images"]
test_images = h5py_to_array(test_images, image_shape)
source_test = test_images
target_test = test_images

# Initialize labels
source_train_labels = train_f["joint_vel"]
source_test_labels = test_f["joint_vel"]
target_test_labels = test_f["joint_vel"]

# Labelling ratio for target data
labelled_ratio = 1.0
target_train_ratio = 1.0 # ratio of target to source training data
target_train_datapoints = int(target_train_ratio * source_train_datapoints)
target_train_labelled_count = int(np.floor(labelled_ratio * target_train_datapoints))
target_train_unlabelled_count = target_train_datapoints - target_train_labelled_count

# Apply some transformation to create target domain
# Apply filter to target train images
yellow_filter = [1, 1, 0]
target_train = train_images[:target_train_datapoints]
target_train = tint_images(target_train, yellow_filter)
target_test = tint_images(target_test, yellow_filter)
target_train_labelled = target_train[:target_train_labelled_count]
target_train_unlabelled = target_train[target_train_labelled_count:]
target_train_labels = train_f["joint_vel"][:target_train_labelled_count]
target_train_unlabelled_labels = np.zeros((target_train_unlabelled_count, num_joints))

# Print statements
print("Total source train datapoints: ", source_train_datapoints)
print("Target to source ratio: ", target_train_ratio)
print("Total target train datapoints: ", target_train_datapoints)
print("Label ratio: ", labelled_ratio)
print("Total target train labelled datapoints: ", target_train_labelled_count)
print("Total target train unlabelled datapoints: ", target_train_unlabelled_count)


### Step 1 Start ###
## Step 1: Train a model on SOURCE data using source task loss only
train_source = True
model_path = 'models/a9_d1_source.h5'

if train_source:
    epochs = 100
    print("Training source model for %d epochs" % epochs)
    inputs = Input(shape=source_train.shape[1:], name='inputs')
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=2,
               padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=2,
               padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=2,
               padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=2,
               padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=2,
               padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)
    feature_layer = Activation('relu', name='feature_layer')(x)
    predictions = Dense(num_joints, name='predictions')(feature_layer)

    # initiate adam optimizer
    adam = keras.optimizers.adam(lr=0.001)

    source_model = Model(inputs=inputs, outputs=[predictions, feature_layer])
    source_model.compile(loss={'predictions': 'mean_squared_error',
                               'feature_layer': empty},
                         optimizer=adam,
                         loss_weights=[1, 1])

    placeholder_labels = np.zeros((source_train_datapoints, 256))
    history = source_model.fit(source_train, [source_train_labels, placeholder_labels],
                               batch_size=batch_size,
                               epochs=epochs,
                               validation_data=(source_test,
                                                [source_test_labels, placeholder_labels[:test_datapoints]]),
                               shuffle="batch",
                               verbose=1)

    # save model and history

    source_model.save(model_path)
    print("Saving model to " + model_path)
    saved_model = h5py.File(model_path, "r+")
    for key in history.history.keys():
        dset = saved_model.create_dataset("history/"+key, (epochs,), dtype='f')
        dset[...] = history.history[key]
    saved_model.close()
else:
    print("Loading source model")
    source_model = load_model(model_path, custom_objects={'empty': empty})

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

target_model = load_model(model_path, custom_objects={'empty': empty})
layer_index = 17 #layer index can actually be found by using the name

# sample random indices
sample_size = min(1024, source_train_datapoints)
sample_indices = sorted(random.sample(range(source_train_datapoints), sample_size))
source_feature_layer = get_activations(source_model, layer_index, source_train[sample_indices])

# Train on labelled target data
placeholder_labels = np.zeros((target_train_labelled_count, 256))
placeholder_labels2 = np.zeros((test_datapoints, 256))


# With mmd loss
print("Training model on target domain with mse and mmd loss")
target_model.compile(loss={'predictions': 'mean_squared_error',
                           'feature_layer': mmd2_rbf_X_quad(source_feature_layer)},
                     optimizer=target_model.optimizer)
# Without mmd loss
# target_model.compile(loss={'predictions': 'mean_squared_error',
#                            'feature_layer': empty},
#                      optimizer=target_model.optimizer)

epochs = 50
history = target_model.fit(target_train_labelled, [target_train_labels, placeholder_labels],
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(target_test,
                                            [target_test_labels, placeholder_labels2]),
                           shuffle='batch',
                           verbose=1)

# Train on unlabelled target data
task_placeholder_labels = np.zeros((1000,6))
test_placeholder_labels = np.zeros((1000,6))
target_model.compile(loss={'predictions': empty,
                           'feature_layer': mmd2_rbf_X_quad(source_feature_layer)},
                     optimizer=target_model.optimizer)
# target_model.fit(target_train_unlabelled, [task_placeholder_labels, placeholder_labels],
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  validation_data=(target_test,
#                                   [test_placeholder_labels, placeholder_labels2]),
#                  shuffle='batch')

target_model_path = "models/a9_d1_target_100_labelled_mse_mmd.h5"
target_model.save(target_model_path)
saved_model = h5py.File(target_model_path, "r+")
for key in history.history.keys():
    dset = saved_model.create_dataset("history/" + key, (epochs,), dtype='f')
    dset[...] = history.history[key]
saved_model.close()
### Step 2 End ###
