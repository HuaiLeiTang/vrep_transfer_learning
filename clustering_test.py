import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


file = h5py.File("datasets/200iter100steps64res.hdf5")
jointvel = file['joint_vel'][:]

np.set_printoptions(suppress=True)

model = TSNE(n_components=2, random_state=0)
output = model.fit_transform(jointvel[:1000])
plt.scatter(output[:,0],output[:,1],marker='+')
