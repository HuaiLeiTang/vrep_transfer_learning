import matplotlib.pyplot as plt
import numpy as np

N = 300 #num of data points
#data = np.random.random((N,2)) # uniform 2D data
data = np.random.normal(0,1,(N,2)) # 2D standard normal (Gaussian with mean 0, std dev 1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = data[:,0] # all rows, 0th col
y = data[:,1] # all rows, 1st col
ax.scatter(x,y,s=25,marker="+")

data2 = np.random.normal(2,1,(N,2))
ax.scatter(data2[:,0],data2[:,1],s=25,marker="o")

plt.ylim(-5,5)
plt.xlim(-5,5)

