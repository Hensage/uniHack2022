import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col
import cv2
from PIL import Image, ImageDraw, ImageColor
import os
import shutil
from random import randint
import skimage.measure

WIDTH = 400
HEIGHT = 600
RADIUS = 12

width = 400
height = 600

last_frame = Image.new('RGB', (WIDTH, HEIGHT), color = 'green')

class ball:
    def __init__(self,x,y,colour):
        self.x = x
        self.y = y
        self.colour = colour


def draw_rectangles(ctrs, img):
    
    output = img.copy()
    
    for i in range(len(ctrs)):
    
        M = cv2.moments(ctrs[i]) # moments
        rot_rect = cv2.minAreaRect(ctrs[i])
        w = rot_rect[1][0] # width
        h = rot_rect[1][1] # height
        
        box = np.int64(cv2.boxPoints(rot_rect))
        cv2.drawContours(output,[box],0,(255,100,0),2) # draws box
        
    return output

def filter_ctrs(ctrs, min_s = 300, max_s = 2000, alpha = 5):  
    
    filtered_ctrs = [] # list for filtered contours
    
    for x in range(len(ctrs)): # for all contours
        
        rot_rect = cv2.minAreaRect(ctrs[x]) # area of rectangle around contour
        w = rot_rect[1][0] # width of rectangle
        h = rot_rect[1][1] # height
        area = cv2.contourArea(ctrs[x]) # contour area 

        
        if (h*alpha<w) or (w*alpha<h): # if the contour isnt the size of a snooker ball
            continue # do nothing
            
        if (area < min_s): # if the contour area is too big/small
            continue # do nothing 
        
        if (area > max_s):
            continue

        # if it failed previous statements then it is most likely a ball
        filtered_ctrs.append(ctrs[x]) # add contour to filtered cntrs list

        
    return filtered_ctrs # returns filtere contours

def find_ctrs_color(ctrs, input_img):

    K = np.ones((3,3),np.uint8) # filter
    output = input_img.copy() #np.zeros(input_img.shape,np.uint8) # empty img
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) # gray version
    mask = np.zeros(gray.shape,np.uint8) # empty mask

    for i in range(len(ctrs)): # for all contours
        
        # find center of contour
        M = cv2.moments(ctrs[i])
        cX = int(M['m10']/M['m00']) # X pos of contour center
        cY = int(M['m01']/M['m00']) # Y pos
    
        mask[...]=0 # reset the mask for every ball 
    
        cv2.drawContours(mask,ctrs,i,255,-1) # draws the mask of current contour (every ball is getting masked each iteration)

        mask =  cv2.erode(mask,K,iterations=3) # erode mask to filter green color around the balls contours
        
        output = cv2.circle(output, # img to draw on
                         (cX,cY), # position on img
                         20, # radius of circle - size of drawn snooker ball
                         cv2.mean(input_img,mask), # color mean of each contour-color of each ball (src_img=transformed img)
                         -1) # -1 to fill ball with color
    return output

def convert(frame, width, height, num):
    pts1 = np.float32([ [450, 305],[805, 305],[320, 615],[950, 615] ])
    pts2 = np.float32([ [0,0],[width,0],[0,height],[width,height] ])

    matrix = cv2.getPerspectiveTransform(pts1,pts2) 
    transformed = cv2.warpPerspective(frame, matrix, (width,height))

    transformed_blur = cv2.GaussianBlur(transformed,(0,0),2) # blur applied
    blur_RGB = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2RGB) # rgb version
    
    # hsv colors of the snooker table
    lower = np.array([60, 150, 110]) 
    upper = np.array([70, 400,400]) # HSV of snooker green: (60-70, 200-255, 150-240) 

    hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV) # convert to hsv
    mask = cv2.inRange(hsv, lower, upper) # table's mask

    # apply closing
    kernel = np.ones((5,5),np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # dilate->erode

    # invert mask to focus on objects on table
    _,mask_inv = cv2.threshold(mask_closing,5,255,cv2.THRESH_BINARY_INV) # mask inv

    masked_img = cv2.bitwise_and(transformed,transformed, mask=mask_inv) # masked image with inverted mask

    mask_img_bw = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    pooled = skimage.measure.block_reduce(mask_closing, (5,5), np.max)
    for i in range(1):
        pooled = skimage.measure.block_reduce(mask_closing, (5,5), np.max)

    # find contours and filter them
    ctrs, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # create contours in filtered img

    # draw contours before filter
    detected_objects = draw_rectangles(ctrs, mask_inv) # detected objects will be marked in boxes

    ctrs_filtered = filter_ctrs(ctrs) # filter unwanted contours (wrong size or shape)

    # draw contours after filter
    detected_objects_filtered = draw_rectangles(ctrs_filtered, transformed) # filtered detected objects will be marked in boxes

    # find average color inside contours:
    ctrs_color = find_ctrs_color(ctrs_filtered, transformed)
    ctrs_color = cv2.addWeighted(ctrs_color,0.5,transformed,0.5,0) # contours color image + transformed image
    
    balls = []

    #cv2.imwrite("balls/" + str(num) + ".jpg", detected_objects_filtered)

    for c in ctrs_filtered:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        balls.append(ball(cX, cY, "red"))
        # draw the contour and center of the shape on the image
        cv2.drawContours(detected_objects_filtered, [c], -1, (0, 255, 0), 2)
        cv2.circle(detected_objects_filtered, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(detected_objects_filtered, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.imshow(detected_objects)
    plt.title('blur')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(detected_objects_filtered)
    plt.title('table mask')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pooled) 
    plt.title('masked objects')
    plt.axis('off')
    plt.show()

    return balls

def convert_2(frame, width, height, num):
    global last_frame

    pts1 = np.float32([ [450, 305],[805, 305],[320, 615],[950, 615] ])
    pts2 = np.float32([ [0,0],[width,0],[0,height],[width,height] ])

    matrix = cv2.getPerspectiveTransform(pts1,pts2) 
    transformed = cv2.warpPerspective(frame, matrix, (width,height))

    transformed_blur = cv2.GaussianBlur(transformed,(0,0),2) # blur applied
    blur_RGB = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2RGB) # rgb version
    
    # hsv colors of the snooker table
    lower = np.array([60, 150, 110]) 
    upper = np.array([70, 400,400]) # HSV of snooker green: (60-70, 200-255, 150-240) 

    hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV) # convert to hsv
    mask = cv2.inRange(hsv, lower, upper) # table's mask

    # apply closing
    kernel = np.ones((5,5),np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # dilate->erode

    count = 0
    for row in mask_closing:
        count += sum(row)
    if count < 255*WIDTH*HEIGHT*0.85:
        pic = Image.fromarray(transformed)
        dst = Image.new('RGB', (pic.width + last_frame.width, pic.height))
        dst.paste(pic, (0, 0))
        dst.paste(last_frame, (last_frame.width, 0))
        return dst

     # invert mask to focus on objects on table
    _,mask_inv = cv2.threshold(mask_closing,5,255,cv2.THRESH_BINARY_INV) # mask inv

    masked_img = cv2.bitwise_and(transformed,transformed, mask=mask_inv) # masked image with inverted mask
    mask_RGB = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    ctrs, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # create contours in filtered img

    balls = []

    filtered_ctrs = [] # list for filtered contours
    oversized_ctrs = []
    
    for x in range(len(ctrs)): # for all contours
        
        rot_rect = cv2.minAreaRect(ctrs[x]) # area of rectangle around contour
        w = rot_rect[1][0] # width of rectangle
        h = rot_rect[1][1] # height
        area = cv2.contourArea(ctrs[x]) # contour area 

        
        if (h*3.4<w) or (w*3.4<h): # if the contour isnt the size of a snooker ball
            continue # do nothing
            
        if (area < 300 or area > 4000): # if the contour area is too big/small
            continue # do nothing 

        if (area > 2000):
            oversized_ctrs.append(ctrs[x])
            continue

        # if it failed previous statements then it is most likely a ball
        filtered_ctrs.append(ctrs[x]) # add contour to filtered cntrs list

    for c in filtered_ctrs:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        col = transformed[cY][cX]
        balls.append(ball(cX, cY, colour(col)))

    for c in oversized_ctrs:
        ball_size = 24
        x,y,w,h = cv2.boundingRect(c)
        mask_RGB = cv2.rectangle(mask_RGB, [x,y], [x+w, y+h], (255,0,0), 2)
        for i in range(x, x+w-ball_size, 2):
            for j in range(y, y+h-ball_size, 2):
                count = 0
                for k in range(ball_size):
                    count += sum(mask_closing[j+k][i:i+ball_size])

                if count < ball_size**2*255*0.05:
                    for l in range(ball_size):
                        mask_closing[j+l][i:i+ball_size] = [255] * ball_size
                    col = transformed[j+ball_size//2][i+ball_size//2]
                    balls.append(ball(i+ball_size/2, j+ball_size/2, colour(col)))
                    mask_RGB = cv2.rectangle(mask_RGB, [i,j], [i+ball_size, j+ball_size], (0,255,0), 2)

    # cv2.imshow("lalala", mask_RGB)
    # cv2.waitKey(0)

    pic = Image.fromarray(transformed)
    img = Image.new('RGB', (WIDTH, HEIGHT), color = 'green')
    d = ImageDraw.Draw(img)
    for b in balls:
        d.ellipse((b.x-RADIUS,b.y-RADIUS, b.x+RADIUS, b.y+RADIUS), fill=b.colour, outline=(0, 0, 0))

    dst = Image.new('RGB', (pic.width + img.width, pic.height))
    dst.paste(pic, (0, 0))
    dst.paste(img, (img.width, 0))
    last_frame = img
    #dst.show()
    return dst

def colour(col):
    return rgb2hex(int(col[0]), int(col[1]), int(col[2]))

def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
try:
    os.mkdir("temp")
except:
    print("exists")
start = 10000
for i in range(start, 12000, 1):
    print(i)
    img = cv2.imread("pics/pool0" + str(i) + ".jpg")
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    balls = convert_2(frame, width, height, i)
    balls.save("temp/" + str(i-start).zfill(6) + ".jpg")
    #create_frame("output"+str(i-1400).zfill(6)+".png", balls)
#reate_video()
os.system("~/audio-orchestrator-ffmpeg/bin/ffmpeg -r 24 -i temp/%06d.jpg -vcodec mpeg4 -y balls6.mp4 -hide_banner")

shutil.rmtree('temp')

# for i in range(10,11):
#     img = cv2.imread("pics/pool0034" + str(i) + ".jpg")
#     frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     res = convert(frame, width, height)




