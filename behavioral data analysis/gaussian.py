import numpy as np
import matplotlib.pyplot as plt


GAUS_STD = 50  #gaussian standard deviation



def gaussian_2d(x, y, mean, std_dev):
    return np.exp(-((x - mean[0])**2 + (y - mean[1])**2) / (2 * std_dev**2)) / (2 * np.pi * std_dev**2)

def fixationGaussian(locations, map_size=(1920, 1080)):
    map_width, map_height = map_size
    x = np.arange(0, map_width)
    y = np.arange(0, map_height)
    X, Y = np.meshgrid(x, y)
    gaussian_map = np.zeros_like(X, dtype=np.float64)
    skipper = 0     # downsampling criterion
    for loc in locations:
        # gaussian_map += gaussian_2d(X, Y, loc, GAUS_STD)
        if skipper < 5:
            skipper+=1
            continue
        gaussian_map = np.maximum(gaussian_map, gaussian_2d(X, Y, loc, GAUS_STD))
        skipper = 0

    gaussian_map = gaussian_map / np.max(gaussian_map) 

    return gaussian_map
