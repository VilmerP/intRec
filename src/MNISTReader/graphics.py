from PIL import Image

class Graphics(object):
    def __init__(self):
        self.width = 28
        self.height = 28
        self.mode = "L"


    def show_image(self, pixels, label):
        image = Image.new(self.mode, (self.width, self.height))
        image.putdata(pixels)
        image.save("src/MNISTReader/" + str(label) + ".tif")