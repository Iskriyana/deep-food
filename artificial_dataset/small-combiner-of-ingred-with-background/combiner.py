import numpy as np
from matplotlib.pyplot import imread
from PIL import Image
import random

apple = Image.open("./apple.png")
apple_w, apple_h = (200, 200)
apple = apple.resize((apple_w,apple_h))
background = Image.open("./background.jpg")
background_w, background_h = background.size

num_apples=3
for _ in range(num_apples):
    position_of_apple = (random.randint(0, background_w-apple_w), random.randint(0, background_h-apple_h))
    background.paste(apple, position_of_apple, apple)
background
