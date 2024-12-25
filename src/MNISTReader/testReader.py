from reader import Reader
from graphics import Graphics

dataReader = Reader()
testImage = Graphics()
labels, images = dataReader.read_training_data()
testImage.show_image(images[0], labels[0])