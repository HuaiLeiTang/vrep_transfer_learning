from vrep_remoteapi.online_training import train

load_model_path = "trained_models/a1_normal30epochs.h5"
save_model_path = "trained_models/a1_normal_online2000steps"

train(load_model_path, save_model_path, 2000)