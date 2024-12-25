import numpy as np
import struct
from array import array

class Reader(object):
    #Initialize class attributes
    def __init__(self):
        #Filepaths
        self.training_images = "./data/train-images.idx3-ubyte"
        self.training_labels = "./data/train-labels.idx1-ubyte"
        self.test_images = "./data/t10k-images.idx3-ubyte"
        self.test_labels = "./data/t10k-labels.idx1-ubyte"

    def read_test_data(self) -> tuple:
        return self.__read_data(True)
    
    def read_training_data(self) -> tuple:
        return self.__read_data(False)

    #Goes through file and converts to python lists
    def __read_data(self, is_test: bool) -> tuple:
        if (is_test):
            file_labels = open(self.test_labels, "rb")
            file_images = open(self.test_images, "rb")
        else:
            file_labels = open(self.training_labels, "rb")
            file_images = open(self.training_images, "rb")
        magic, size = struct.unpack(">II", file_labels.read(8))
        if magic != 2049:
            raise ValueError("Incorrect file (wrong magic number)")
        labels = []
        for _ in range(size):
            labels.append(struct.unpack("B", file_labels.read(1))[0])

        magic, size, cols, rows = struct.unpack(">IIII", file_images.read(16))
        if magic != 2051:
            raise ValueError("Incorrect file (wrong magic number)")
        images = []
        for i in range(size):
            image = []
            for _ in range(784):
                image.append(struct.unpack("B", file_images.read(1))[0])
            images.append(image)

        return labels, images