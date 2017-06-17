import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


def h5py_to_array(dataset, img_shape):
    # img_shape as a tuple
    images = np.array(dataset)
    images = np.uint8(images)
    images = np.reshape(images, (images.shape[0],) + img_shape)
    return images

def convert_grayscale(images):
    # axis=-1 uses the last dimension
    images = images.mean(axis=-1)
    return images

def adjust_gamma(images, gamma):
    # gamma greater than 1 leads to darker images
    # gamma less than 1 leads to brigher images
    return exposure.adjust_gamma(images, gamma=gamma)

def display(img):
    # origin lower flips the image from top to bottom (mirror around x-axis)
    plt.imshow(img, origin='lower')

def display_grayscale(img):
    plt.imshow(img, origin='lower', cmap=plt.cm.gray)


# Test
if __name__ == '__main__':
    print 'Testing image_utils/methods.py'
    import h5py

    file = h5py.File("../datasets/200iter100steps64res.hdf5", "r")
    images = file['images']
    images = h5py_to_array(images, (64, 64, 3))
    display(images[0])