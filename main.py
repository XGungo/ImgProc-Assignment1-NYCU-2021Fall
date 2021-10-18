import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numba import njit
import matplotlib.image as img
import cv2


def contrast_adjustment():
    pass


def color_correction():
    pass


def power_law(image, c=1., gamma=1.):
    return np.array((image / 255) ** gamma * c * 255, dtype='uint8')


def log(image):
    c = 255 / (np.log(1 + 255))
    log_transformed = c * np.log(1 + image)
    return np.array(log_transformed, dtype='uint8')


def minmax(image):
    return np.array((image - np.min(image)) / (np.max(image) - np.min(image)) * 255, dtype='uint8')


def hist_eq(image):
    hist, bin = np.histogram(image, 256, [0, 255], density=1)
    print(hist.shape)
    cdf = np.around(np.cumsum(hist) * 255).astype('uint8')
    return cdf[image]


def padding(image, kernel_shape, padding_mode='zero'):
    m, n = kernel_shape
    padding_image = np.zeros(np.array(image.shape) + np.array(((n - 1), (m - 1))))
    print(padding_image.shape)

    padding_image[(n - 1) // 2:-(n - 1) // 2, (m - 1) // 2: -(m - 1) // 2] = image
    if padding_mode == 'replicate':
        for i in range((n - 1) // 2):
            padding_image[i, (m - 1) // 2: -(m - 1) // 2] = image[0]
            padding_image[-(i + 1), (m - 1) // 2: -(m - 1) // 2] = image[-1]
        for j in range((m - 1) // 2):
            # top
            padding_image[(n - 1) // 2:-(n - 1) // 2, j] = image[:, 0]
            # top corner
            padding_image[:(n - 1) // 2, j] = image[0, 0]
            padding_image[-(n - 1) // 2:, j] = image[-1, 0]
            # bottom
            padding_image[(n - 1) // 2:-(n - 1) // 2, -(j + 1)] = image[:, -1]
            # bottom corner
            padding_image[:(n - 1) // 2, -(j + 1)] = image[0, -1]
            padding_image[-(n - 1) // 2:, -(j + 1)] = image[-1, -1]
    return padding_image


@njit
def convolution(image, kernel, padding_image):
    m, n = kernel.shape
    new_image = np.zeros(image.shape)
    img_m, img_n = image.shape
    for i in range(img_m):
        for j in range(img_n):
            new_image[i, j] = np.sum(kernel * padding_image[i:i + m, j:j + n])

    return new_image


def noise_smoothing(image, k_size=3, kernel_mode='average', padding_mode='zero'):
    kernel = np.ones((k_size, k_size)) / (k_size * k_size)
    padding_image = padding(image, (k_size, k_size), padding_mode)
    if kernel_mode == 'gaussian':
        x, y = np.mgrid[-(k_size - 1) // 2:(k_size - 1) // 2 + 1,
               -(k_size - 1) // 2:(k_size - 1) // 2 + 1]
        kernel = np.exp(-(x ** 2 + y ** 2))
        kernel = kernel / kernel.sum()

    return convolution(image, kernel, padding_image)


@njit
def midpoint_filter(image, m, n, padding_image):
    new_image = np.zeros(image.shape)
    img_m, img_n = image.shape
    for i in range(img_m):
        for j in range(img_n):
            new_image[i, j] = (np.max(padding_image[i:i + m, j:j + n]) + np.min(padding_image[i:i + m, j:j + n]))/2
    return new_image

@njit
def median_filter(image, m, n, padding_image):
    new_image = np.zeros(image.shape)
    img_m, img_n = image.shape
    for i in range(img_m):
        for j in range(img_n):
            new_image[i, j] = np.median(padding_image[i:i + m, j:j + n])
    return new_image

def filtering(image, k_size, mode='median', **kwargs):
    m = k_size
    n = k_size
    padding_image = padding(image, (m, n), **kwargs)
    if mode == 'median':
        return median_filter(image, m, n, padding_image)
    if mode == 'midpoint':
        return midpoint_filter(image, m, n, padding_image)
    if mode == 'laplacian':
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        return convolution(image, kernel, padding_image)
    if mode == 'laplacian-corner':
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        return convolution(image, kernel, padding_image)
    if mode == 'sharpening':
        print(123)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return convolution(image, kernel, padding_image)

if __name__ == '__main__':
    print('Hello World!')

    img = Image.open('sample3.jpeg').convert('L')
    # img = Image.open('sample.jpg')
    img.show()
    img = np.array(img)
    # new_img = Image.fromarray(log(img))
    # new_img.show()
    # for gamma in [.3, .5, .8, 1.2, 1.5, 2]:
    #     new_img = Image.fromarray(power_law(img, gamma=gamma))
    #     new_img.show()
    # new_img = Image.fromarray(minmax(img))
    # new_img.show()
    # plt.hist(img.ravel(), bins=range(256), cumulative=0, histtype='step', density=True)
    #
    # new_img = hist_eq(img)
    # plt.hist(new_img.ravel(), bins=range(256), cumulative=0, histtype='step', density=True)
    # new_img = Image.fromarray(new_img)
    # new_img.show()
    new_img = noise_smoothing(img, k_size=7, kernel_mode='average', padding_mode='replicate')
    Image.fromarray(new_img).show()
    new_img = filtering(img, k_size=3, mode='median', padding_mode='replicate')
    Image.fromarray(new_img).show()
    new_img = filtering(new_img, k_size=3, mode='laplacian', padding_mode='replicate')
    Image.fromarray(new_img).show()

