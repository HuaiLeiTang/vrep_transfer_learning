# All functions assume to be working with a float32 format image with range [0,1]
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.util import random_noise
from skimage.filters import gaussian, laplace, frangi, hessian

def h5py_to_array(dataset, img_shape):
    # img_shape as a tuple
    # assumes input is in the uint8 format
    # converts into a standard float32 in range[0,1]
    images = np.array(dataset)
    images = np.float32(images)/255
    images = np.reshape(images, (images.shape[0],) + img_shape)
    return images

def convert_grayscale(images):
    # axis=-1 uses the last dimension
    images = images.mean(axis=-1)
    return images

def adjust_gamma(images, gamma=1):
    # gamma greater than 1 leads to darker images
    # gamma less than 1 leads to brighter images
    return exposure.adjust_gamma(images, gamma=gamma)

def display(img):
    # origin lower flips the image from top to bottom (mirror around x-axis)
    return plt.imshow(img, origin='lower')

def display_grayscale(img):
    return plt.imshow(img, origin='lower', cmap=plt.cm.gray)


def tint_images(images, filter=[1, 1, 1]):
    return filter*images

## noise function return images with values in range [0,1]
def noise_gaussian(images, mean=0, var=0.01):
    # adds a gaussian variable with specified mean and var
    # to each pixel in image
    return random_noise(images, mode='gaussian',
                        mean=mean, var=var)

def noise_sp(images, amount=0.05, salt_vs_pepper=0.5):
    return random_noise(images, mode='s&p', amount=amount,
                        salt_vs_pepper=salt_vs_pepper)

def blur_gaussian(images, sigma=1):
    return gaussian(images, sigma=sigma)

def laplace_edges(images):
    return laplace(images)

def frangi_filter(image):
    return frangi(image)

def hessian_filter(image):
    return hessian(image)

def display_2images(image1, image2):
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    a = display(image1)
    b = fig.add_subplot(1,2,2)
    b = display(image2)
    return

# Test
if __name__ == '__main__':
    print 'Testing image_utils/methods.py'
    import h5py

    file = h5py.File("../datasets/200iter100steps64res.hdf5", "r")
    images = file['images']
    images = h5py_to_array(images[:5], (64, 64, 3))

    print 'Displaying image'
    display(images[0])
    plt.show()

    print 'Displaying gamma=2 corrected image'
    gamma_images = adjust_gamma(images,2)
    display(gamma_images[0])
    plt.show()

    print 'Displaying 2 images'
    display_2images(images[0], gamma_images[0])
    plt.show()

    print 'Displaying yellow tinted image'
    yellow_tinted_images = tint_images(images,[1,1,0])
    display(yellow_tinted_images[0])
    plt.show()

    # var can be reduced to limit the effect
    # default var=0.01
    print 'Displaying gaussian noise image'
    gaussian_noise_images = noise_gaussian(images)
    display(gaussian_noise_images[0])
    plt.show()

    # s&p doesn't work properly for triple color channels
    print 'Displaying salt and pepper noise image'
    sp_images = noise_sp(images)
    display(sp_images[0])
    plt.show()

    print 'Displaying gaussian blur image'
    gaussian_blur_images = blur_gaussian(images, sigma=0.5)
    display(gaussian_blur_images[0])
    plt.show()

    print 'Displaying laplace edges image'
    laplace_images = laplace_edges(images)
    display(laplace_images[0])
    plt.show()

    print 'Displaying grayscale image'
    grayscale_images = convert_grayscale(images)
    display_grayscale(grayscale_images[0])
    plt.show()

    # frangi and hessian filters can only be applied to one image at a time
    # might not be an efficient operation
    print 'Displaying frangi filtered image'
    frangi_image = frangi(grayscale_images[0])
    display_grayscale(frangi_image)
    plt.show()

    print 'Displaying hessian filtered image'
    hessian_image = hessian(grayscale_images[0])
    display_grayscale(hessian_image)
    plt.show()