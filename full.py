import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage import transform
from skimage.io import imread, imshow
from skimage import data, io, filters
import os
from PIL import Image, ImageDraw
import shutil
from random import randint

WIDTH = 400
HEIGHT = 600

def warp_to_table(frame):

    img = io.imread(frame)

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
    io.imsave(frame[0:-3] + "warped.jpg", warped_image)
    return warped_image

def colourReg(frame):
    img = Image.open(frame)
    for x in range(WIDTH):
        for y in range(HEIGHT):
            avgCol = img.getpixel(x,y)
            for xp in range(10):
                if (x+xp < WIDTH):
                    for yp in range(10):
                        if (y+yp < HEIGHT):
                            temp =img.getpixel(x+xp,y+yp)
                            avgCol = ((avgCol[0]+temp[0])/2,(avgCol[1]+temp[1])/2,(avgCol[1]+temp[1])/2)

RADIUS = 10

class ball:
    def __init__(self,x,y,colour):
        self.x = x
        self.y = y
        self.colour = colour

def create_frame(output,balls):
    img = Image.new('RGB', (WIDTH, HEIGHT), color = 'green')
    d = ImageDraw.Draw(img)
    for ball in balls:
        d.ellipse((ball.x-RADIUS,ball.y-RADIUS, ball.x+RADIUS, ball.y+RADIUS), fill=ball.colour, outline=(0, 0, 0))
    img.save("temp123123123213123/"+output)
def create_video():
    os.system("ffmpeg -r 24 -i temp123123123213123/output%06d.png -vcodec mpeg4 -y movie.mp4 -hide_banner")

for i in os.listdir("pics"):
    warp_to_table("pics/"+i)


'''
os.mkdir("temp123123123213123")
for i in range(1000):
    create_frame("output"+str(i).zfill(6)+".png",[ball(randint(10,WIDTH-10),randint(10,HEIGHT-10),(0,255,0))])
create_video()
shutil.rmtree('temp123123123213123')
'''