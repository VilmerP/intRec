import numpy as np
import random as r
from nn.list_data import weightsHL, weightsOL, biasHL, biasOL

# The neural network. Can evaluate images as digits and train to improve accuracy
class Network(object):
    def __init__(self):
        self.learning_step = 0.1
        self.HL_activation = []
        self.output = []
        self.default_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.zlist_HL = []
        self.zlist_OL = []
        self.output_layer_size = 10
        self.hidden_layer_size = 16
        self.input_layer_size = 784
        self.weightsHL = weightsHL
        self.weightsOL = weightsOL
        self.biasHL = biasHL
        self.biasOL = biasOL

    # resets weights and biases to a normal distribution
    def randomize(self):
        self.weightsHL = np.random.normal(loc=0, scale=1, size=(self.hidden_layer_size, self.input_layer_size))
        self.weightsOL = np.random.normal(loc=0, scale=1, size=(self.output_layer_size, self.hidden_layer_size))
        self.biasHL = np.random.normal(loc=1, scale=1, size=(self.hidden_layer_size))
        self.biasOL = np.random.normal(loc=1, scale=1, size=(self.output_layer_size))


    def __zcalc(self, activation: list, weight: list, bias: list) -> list:
        result = []
        for i in range (len(bias)):
            result.append(np.dot(activation, weight[i]) + bias[i])
        return result
    

    # activation funcs
    def __sigmoidfunc(self, z: float) -> float:
        return 1/(1 + np.exp(-z)) 
    
    def __tanhfunc(self, z: float) -> float:
        return np.tanh(z)
    
    # activation derivatives
    def __delsig(self, z: float) -> float:
        return self.__sigmoidfunc(z)*(1-self.__sigmoidfunc(z))
    
    def __deltanh(self, z: float) -> float:
        return 1 - np.tanh(z)**2
    
    # derivative of cost function
    def __delcost(self, act: list, label: list) -> float:
        return 2*(act - label)
    
    # calculates layers according to input
    def forward_pass(self, inputs: int):
        self.output = []
        self.HL_activation = []
        self.zlist_HL = np.array(self.__zcalc(inputs, self.weightsHL, self.biasHL))
        for i in range(len(self.zlist_HL)):
            self.HL_activation.append(self.__tanhfunc(self.zlist_HL[i]))

        self.zlist_OL = np.array(self.__zcalc(self.HL_activation, self.weightsOL, self.biasOL))
        for i in range(len(self.zlist_OL)):
            self.output.append(self.__sigmoidfunc(self.zlist_OL[i]))
        self.output = np.array(self.output)

    # calculates gradient of parameters (weights and biases) 
    def __backprop(self, label: list, input: list):
        # output layer
        dAdZ_OL = self.__delsig(self.zlist_OL)
        dCdA_OL = self.__delcost(self.output, label)
        dCdB_OL = np.multiply(dAdZ_OL, dCdA_OL) # hadamard product
        dCdW_OL = []
        for i in range(len(self.output)):
            dCdW_OL.append([])
            for j in range(len(self.HL_activation)):
                dCdW_OL[i].append(dCdB_OL[i] * self.HL_activation[j])
        dCdW_OL = np.array(dCdW_OL)

        # hidden layer
        dAdZ_HL = self.__deltanh(self.zlist_HL)
        dCdA_HL = []
        # remember that the gradient operator can be a linear operator
        for i in range(len(self.HL_activation)):
            sum = 0
            for j in range(len(self.output)):
                sum += dCdA_OL[j] * dAdZ_OL[j] * self.weightsOL[j][i]
            dCdA_HL.append(sum)
        dCdA_HL = np.array(dCdA_HL)
        dCdB_HL = np.multiply(dCdA_HL, dAdZ_HL)
        dCdB_HL = np.multiply(dCdA_HL, dAdZ_HL)
        dCdW_HL = []
        for i in range(len(self.HL_activation)):
            node = []
            for j in range(len(input)):
                node.append(dCdA_HL[i] * dAdZ_HL[i] * input[j])
            dCdW_HL.append(np.array(node))


        return dCdW_HL, dCdB_HL, dCdW_OL, dCdB_OL

    # applies changes after a batch
    def learn(self, sum_dCdW_HL: int, sum_dCdB_HL: int, sum_dCdW_OL: int, sum_dCdB_OL: int, batch_size: int):
        sum_dCdW_HL /= batch_size
        sum_dCdW_OL /= batch_size
        sum_dCdB_HL /= batch_size
        sum_dCdB_OL /= batch_size

        self.weightsHL -= sum_dCdW_HL
        self.weightsOL -= sum_dCdW_OL
        self.biasHL -= sum_dCdB_HL
        self.biasOL -= sum_dCdB_OL

    # updades file list_data holding all weights and biases after running through all batches (one epoch)
    def update_values(self):
        with open("src/nn/list_data.py", "w") as file:
            file.write(f"weightsHL = {np.ndarray.tolist(self.weightsHL)}\n")
            file.write(f"weightsOL = {np.ndarray.tolist(self.weightsOL)}\n")
            file.write(f"biasHL = {np.ndarray.tolist(self.biasHL)}\n")
            file.write(f"biasOL = {np.ndarray.tolist(self.biasOL)}\n")

    # evaluates given image (input) as a digit
    def evaluate(self, input: int) -> int:
        self.forward_pass(input)
        return np.argmax(self.output)

    # trains the neural network
    def train(self, images: list, labels: list, batch_size: int, batch_amount: int, randomize=True):
        if batch_size * batch_amount > 60000:
            raise Exception("Can't request more than 60 000 images")
        if randomize:
            self.randomize()

        for i in range(batch_amount):

            sum_dCdW_HL = np.zeros((self.hidden_layer_size, self.input_layer_size))
            sum_dCdB_HL = np.zeros(self.hidden_layer_size)
            sum_dCdW_OL = np.zeros((self.output_layer_size, self.hidden_layer_size))
            sum_dCdB_OL = np.zeros(self.output_layer_size)

            # one batch per loop
            for j in range(batch_size):
                label = self.default_label.copy()
                label[labels[(i * batch_size) + j]] = 1
                input = np.array(images[(i * batch_size) + j])
                mean = np.mean(input)
                input = (input - mean) / 255

                self.forward_pass(input)
                
                dCdW_HL, dCdB_HL, dCdW_OL, dCdB_OL = self.__backprop(label, input)
                sum_dCdW_HL += dCdW_HL
                sum_dCdW_OL += dCdW_OL
                sum_dCdB_HL += dCdB_HL
                sum_dCdB_OL += dCdB_OL

            self.learn(sum_dCdW_HL, sum_dCdB_HL, sum_dCdW_OL, sum_dCdB_OL, batch_size)

        self.update_values()
