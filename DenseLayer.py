import numpy as np
from SyntheticData import spiral_data

np.random.seed(0)


#Generating synthetic data
X, y = spiral_data(100, 3)

class LayerDense:

    #initalizing the weights and baises
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
    
    def forward(self, inputs):
        #We need to pass the output from the previous nureon to the next nueron (Feed Forward)
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#Initializing layers
layer1 = LayerDense(2, 5) #Basically how many features per sample, like in the above exmaple we have 4 features so 4 inputs we have.
activation1 = ActivationReLU() 

#Passing the data into the 1st layer
layer1.forward(X)

#Applying ReLU (Rectified Linear Unit Activation Function)
#Relu - Rectified Linear Unit Activation Function is basically very simple activation function

activation1.forward(layer1.output)
print(activation1.output)
