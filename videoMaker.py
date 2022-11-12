from PIL import Image, ImageDraw
import os
import shutil
from random import randint

WIDTH = 400
HEIGHT = 800
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
os.mkdir("temp123123123213123")
for i in range(1000):
    create_frame("output"+str(i).zfill(6)+".png",[ball(randint(10,WIDTH-10),randint(10,HEIGHT-10),(0,255,0))])
create_video()
shutil.rmtree('temp123123123213123')