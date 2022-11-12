from PIL import Image

WIDTH = 400
HEIGHT = 600

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

                    