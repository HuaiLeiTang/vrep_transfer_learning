from vrep_remoteapi.online_training_mixed import train

load_model_path = "trained_models/a1_normal_online_mixed_16batchsize.h5"
save_model_path = "trained_models/a1_normal_online_mixed_16_64batchsize.h5"
data_path = "datasets/200iter100steps64res.hdf5"

batch_size = 64
number_training_steps = 1000

train(load_model_path, save_model_path, number_training_steps,
      data_path, ratio=1, batch_size=batch_size)