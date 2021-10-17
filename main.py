import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
    hist, bin = np.histogram(image, bins=range(256), density=1)
    cdf = np.around(np.cumsum(hist) * 255).astype('uint8')
    return cdf[image]


def convolution(image, kernel, padding='zero'):
    m, n = kernel.shape
    padding_image = np.zeros(np.array(image.shape) + np.array(((n - 1), (m - 1))))
    padding_image[(n - 1) // 2:-(n - 1) // 2, (m - 1) // 2: -(m - 1) // 2] = image
    if padding == 'replicate':
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

    new_image = np.zeros(image.shape)
    img_m, img_n = image.shape
    for i in range(img_n):
        for j in range(img_m):
            new_image[i, j] = np.sum(kernel * padding_image[i:i+n, j:j+m].T).astype('uint8')

    return new_image


def noise_smoothing(image, kernel_size=3, kernel_mode='average', **kwargs):

    if kernel_mode == 'average':
        kernel = np.ones((kernel_size, kernel_size))/(kernel_size * kernel_size)
        return convolution(image, kernel, **kwargs)
    if kernel_mode == 'gaussian':
        x, y = np.mgrid[-(kernel_size-1)//2:(kernel_size-1)//2+1, -(kernel_size-1)//2:(kernel_size-1)//2+1]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        print(gaussian_kernel)
        return 0


if __name__ == '__main__':
    print('Hello World!')

    img = Image.open('lena512.bmp')
    # img = Image.open('sample.jpg')
    # img.show()
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
    #
    # plt.show()
    new_img = noise_smoothing(img, kernel_size=7, kernel_mode='gaussian', padding='replicate')
    # new_img = Image.fromarray(new_img)
    # new_img.show()
