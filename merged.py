import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col
import cv2
from PIL import Image, ImageDraw

WIDTH = 400
HEIGHT = 600

currentBall = 17
pocketStatus = [[False,"NaN"],[False,"NaN"],[False,"NaN"],[False,"NaN"],[False,"NaN"],[False,"NaN"]]
pocketCoord = [(13,5),(384,6),(12,288),(393,287),(6,580),(394,576)]

def warp(frame, width, height):
    pts1 = np.float32([ [440, 295],[810, 295],[320, 620],[950, 620] ])
    pts2 = np.float32([ [0,0],[width,0],[0,height],[width,height] ])

    matrix = cv2.getPerspectiveTransform(pts1,pts2) 
    transformed = cv2.warpPerspective(frame, matrix, (width,height))

    return transformed

def colourReg(img):

    global currentBall, pocketStatus, pocketCoord
    img = Image.fromarray(img)
    ball = 0
    for i in range(len(pocketCoord)):
        if pocketStatus[i][0]:
            if img.getpixel(pocketCoord[i])[0]<150:
                currentBall -= 1
                pocketStatus[i] = [False,"NaN"]
                print("BALL")
        else:
            if img.getpixel(pocketCoord[i])[0]>150:
                pocketStatus[i] = [True,"Colour"]
                print("CLOSE")
    print(currentBall)

img = cv2.imread('pics/pool001407.jpg')
frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
warped = warp(frame, WIDTH, HEIGHT)
colourReg(warped)


