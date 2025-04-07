import PIL
from PIL import Image

img = Image.open('imgs/fox.jpg')
# print(PIL.__version__)
# img.rotate(90).show()
img = img.crop((100, 100, 400, 400))
img.show()