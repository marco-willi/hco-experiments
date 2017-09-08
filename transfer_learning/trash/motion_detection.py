# Test Motion Detection
import numpy as np
from keras.preprocessing.image import img_to_array,  load_img
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
from skimage.feature import register_translation
from scipy import ndimage as ndi
from scipy import ndimage
from skimage import feature
from skimage import draw
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter)
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, color
from skimage.filters import try_all_threshold
from skimage.filters import threshold_mean, threshold_isodata
from math import sqrt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# Parameters
image_path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\snapshot_wisconsin\\RACCOON\\'
image_id = '2008342'

image_path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\snapshot_wisconsin\\SQUIRRELSANDCHIPMUNKS\\'
image_id = '2008150'

image_path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\snapshot_wisconsin\\COYOTE\\'
image_id = '2008283'


image_path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\snapshot_wisconsin\\DEER\\'
image_id = '2007873'
image_id = '2007875'

num_images = 3
i = 0
grayscale = False
# load images
for i in range(0, num_images):
    f_path = image_path + image_id + '_' + str(i) + '.jpeg'
    img = load_img(f_path, grayscale=grayscale)
    img_arr = img_to_array(img) / 255
    #img_arr[:, :, (0)] = 0
    if i == 0:
        img_avg = img_arr
    else:
        img_avg = img_avg + img_arr
img_avg = img_avg / 3

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
# # load images
# for i in range(0, num_images):
#     f_path = image_path + image_id + '_' + str(i) + '.jpeg'
#     img = load_img(f_path, grayscale=grayscale)
#     img_arr = img_to_array(img)
#     # img_arr[:, :, (0)] = 0
#     if i == 0:
#         img_comp = img_arr
#     else:
#         img_diff = abs(img_comp - img_arr)
#
# image_max = ndi.maximum_filter(img_diff, size=10, mode='constant')
# coordinates = peak_local_max(img_diff, num_peaks=1)
# plt.imshow(img_diff)
# plt.gray()
# plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
# plt.show()


# load images and show denoised versions
for i in range(0, num_images):
    f_path = image_path + image_id + '_' + str(i) + '.jpeg'
    #img_arr = img_as_float(load_img(f_path, grayscale=grayscale))
    img = load_img(f_path, grayscale=grayscale)
    img_arr = img_to_array(img) / 255
    blurred = ndimage.gaussian_filter(img_arr, sigma=3)
    blu_avg = ndimage.gaussian_filter(img_avg, sigma=3)
    local_mean = ndimage.uniform_filter(img_arr, size=11)
    local_mean_avg = ndimage.uniform_filter(img_avg, size=11)
    im_med = ndimage.median_filter(img_arr, 3)
    im_med_avg = ndimage.median_filter(img_avg, 3)
    # for new, avg in zip([img_arr, blurred, local_mean, im_med],
    #     [img_avg, blu_avg, local_mean_avg, im_med_avg]):
    for new, avg in zip([img_arr, blurred, local_mean],
        [img_avg, blu_avg, local_mean_avg]):
        plt.imshow(np.squeeze(new))
        plt.gray()
        plt.show()
        plt.imshow(np.squeeze(avg))
        plt.gray()
        plt.show()
        diff = 1-abs(np.squeeze(new) - np.squeeze(avg))
        # (score, diff) = compare_ssim(
        #     np.squeeze(new),
        #     np.squeeze(avg), full=True, multichannel=bool(not grayscale))
        # plt.imshow(np.squeeze(diff))
        # plt.gray()
        # plt.show()
        # image_max = ndi.maximum_filter(diff, size=10, mode='constant')
        # coordinates = peak_local_max(diff, num_peaks=3)
        # plt.imshow(new)
        # plt.gray()
        # plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        # plt.show()
        thresh = threshold_isodata(diff)
        binary = diff > thresh
        binary_img = diff
        binary_img[binary] = 1
        binary_img[binary==False] = 0
        # calculate center
        x_sums = np.sum(binary_img, axis=0)
        x_min = np.argmin(x_sums, axis=0)
        y_sums = np.sum(binary_img, axis=1)
        y_min = np.argmin(y_sums, axis=0)
        center = (x_min[0], y_min[0])
        rr, cc = circle(c=center[0], r=center[1], radius=30, shape=binary_img.shape)
        binary_img[rr, cc, 0] = 1
        plt.imshow(np.squeeze(binary_img))
        plt.gray()
        plt.show()


# Plot a Histogram
n, bins, patches = plt.hist(binary_img.flatten(), 50, normed=1, facecolor='green', alpha=0.75)
# add a 'best fit' line
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([0, 1, 0, 0.5])
plt.grid(True)
plt.show()


diff.
diff[0:10,0:10]
diff.shape
# compare against average
img_arr[0:10,0:10,0]

for i in range(0, num_images):
    f_path = image_path + image_id + '_' + str(i) + '.jpeg'
    img = load_img(f_path, grayscale=grayscale)
    img_arr = img_to_array(img)
    img_arr[:, :, (0)] = 0
    (score, diff) = compare_ssim(np.squeeze(img_arr),
                                 np.squeeze(img_avg), full=True, multichannel=bool(not grayscale))
    plt.imshow(np.squeeze(img_arr))
    plt.gray()
    plt.show()
    plt.imshow(np.squeeze(diff))
    plt.gray()
    plt.show()

    # edge Detection
    # edges1 = roberts(np.squeeze(diff))
    image_max = ndi.maximum_filter(diff, size=10, mode='constant')
    coordinates = peak_local_max(diff, num_peaks=1)


    # edges2 = feature.canny(np.squeeze(diff), sigma=3)
    # plt.imshow(np.squeeze(edges1))
    # plt.gray()
    # plt.show()
    # plt.imshow(np.squeeze(edges2))
    # plt.gray()
    # plt.show()
    #
    # x_sums = np.sum(edges1, axis=0)
    # x_min = np.argmax(x_sums)
    # y_sums = np.sum(edges1, axis=1)
    # y_min = np.argmax(y_sums)
    # center = (x_min, y_min)
    #
    # rr, cc = circle(c=center[0], r=center[1], radius=30, shape=edges2.shape)
    # edges2[rr, cc] = 1
    # plt.imshow(edges2)
    # plt.gray()
    # plt.show()

    plt.imshow(img_arr)
    plt.gray()
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    plt.show()

    # img_arr[rr, cc, :] = 255
    # plt.imshow(img_arr)
    # plt.gray()
    # plt.show()


i=0
f_path = image_path + image_id + '_' + str(i) + '.jpeg'
img = load_img(f_path, grayscale=False)
img_arr = img_to_array(img)

i=1
f_path = image_path + image_id + '_' + str(i) + '.jpeg'
img2 = load_img(f_path, grayscale=False)
img2_arr = img_to_array(img2)


(score, diff) = compare_ssim(np.squeeze(img_arr),
                             np.squeeze(img2_arr), full=True, multichannel=True)

diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

plt.imshow(np.squeeze(img2_arr))
plt.gray()
plt.show()
plt.imshow(np.squeeze(img_arr))
plt.gray()
plt.show()
plt.imshow(np.squeeze(diff))
plt.gray()
plt.show()



shift, error, diffphase = register_translation(img_arr, img2_arr)
image_product = np.fft.fft2(img_arr) * np.fft.fft2(img_arr2).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")


def img_to_array(img, data_format=None):

    def load_img(path, grayscale=False, target_size=None):
