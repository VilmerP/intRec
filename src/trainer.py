from MNISTReader.reader import Reader
from nn.network import Network

# allows for training and testing the neural network
class Trainer(object):
    def __init__(self):
        self.dataReader = Reader()
        self.nn = Network()
        self.batch_size = 32

    # train the network with given amount of batches
    def train(self, batch_amount: int, randomize = True):
        labels, images = self.dataReader.read_training_data()

        self.nn.train(images, labels, self.batch_size, batch_amount, randomize)

    # tests the network with given amount of images
    def test(self, test_size: int):
        labels, images = self.dataReader.read_test_data()
        correct = 0
        incorrect = 0

        for i in range(test_size):
            value = self.nn.evaluate(images[i])
            if value == labels[i]:
                correct += 1
            else:
                incorrect += 1

        print(f"amount correct: {correct}, amount incorrect: {incorrect}")



trainer = Trainer()
# trainer.train(0, True)
trainer.test(10000)
# for i in range(100):
#     trainer.train(1875, False)
#     print(f"EPOCH #{i + 1} DONE")
#     trainer.test(1000)