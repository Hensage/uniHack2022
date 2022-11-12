from PIL import Image, ImageDraw

WIDTH = 400
HEIGHT = 600
currentBall = 17
pocketStatus = [[False,"NaN"],[False,"NaN"],[False,"NaN"],[False,"NaN"],[False,"NaN"],[False,"NaN"]]
pocketCoord = [(13,5),(384,6),(12,288),(393,287),(6,580),(394,576)]
def colourReg(frame):
    global currentBall
    img = Image.open(frame)
    ball = 0
    for i in range(len(pocketCoord)):
        print(img.getpixel(pocketCoord[i])[0])
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
colourReg("pics/pool001410.warped.jpg")