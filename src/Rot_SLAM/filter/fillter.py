import numpy as np
import cv2 as cv
from scipy.ndimage import convolve1d
from scipy.signal import firwin, welch
from skimage import io

image = cv.imread('/home/zzy/SLAM/my_slam_work/src/Rot_intensity_augm_LIVO/image/dense/200.png')

def artifact_filter(image, artifact_frequency, taps, epsilon, print_params=False):
  image = np.asarray(image, float)
  #First perform highpass vertically, then lowpass horizontally and remove filtered signal from original image
  return image - lowpass(highpass(image, artifact_frequency, taps, epsilon, print_params), artifact_frequency, taps, epsilon, print_params)


def highpass(image, distortion_freq, taps, epsilon, print_params=False):
  highpass_filter = firwin(taps, distortion_freq - epsilon,
               pass_zero='highpass', fs=1)
  if print_params:
    print("Highpass FIR Parameters:")
    print(highpass_filter)
  return convolve1d(image, highpass_filter, axis=0)

def lowpass(image, distortion_freq, taps, epsilon, print_params=False):
  lowpass_filter = firwin(taps, epsilon, pass_zero='lowpass', fs=1)
  if print_params:
    print("Lowpass FIR Parameters:")
    print(lowpass_filter)
  return convolve1d(image, lowpass_filter, axis=1)


artifact_frequency = 0.05 # Line artifacts are at every 4th line -> 0.25 frequency
epsilon = 0.04
taps = 33


image_filtered = artifact_filter(image, artifact_frequency, taps, epsilon, print_params=True)

cv.imwrite('/home/zzy/SLAM/my_slam_work/src/Rot_intensity_augm_LIVO/image/dense/200_filtered.png', image_filtered)