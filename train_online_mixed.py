from vrep_remoteapi.online_training_mixed import train

load_model_path = "trained_models/a6.h5"
save_model_path = "trained_models/a6_online2.h5"
data_path = "datasets/200iter100steps64res.hdf5"

batch_size = 64
number_training_steps = 500

train(load_model_path, save_model_path, number_training_steps,
      data_path, ratio=0, batch_size=batch_size)