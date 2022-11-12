import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage import transform
from skimage.io import imread, imshow
from skimage import data, io, filters

table = io.imread('pool001425.jpg')

def warp_to_table(img):

    points_of_interest =[[810, 295], 
                        [950, 620], 
                        [320, 620], 
                        [440, 295]]
    projection = [[400, 000],
                [400, 600],
                [000, 600],
                [000, 000]]
    points_of_interest = np.array(points_of_interest)
    projection = np.array(projection)

    tform = transform.estimate_transform('projective', points_of_interest, projection)
    warped_image = (transform.warp(img, tform.inverse, mode = 'symmetric'))[0:600,0:400]
    return warped_image

warp_to_table(table)
io.show()