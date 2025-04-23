import numpy as np


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)  
        self.bias = 0  
        self.learning_rate = learning_rate  

    
    def activation(self, x):
        return 1 if x >= 0 else 0

   
    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)

   
    def train(self, inputs, targets, epochs=10):
        for epoch in range(epochs):
            for x, target in zip(inputs, targets):
                output = self.forward(x)
                error = target - output  
                
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error
            print(f"Epoch {epoch+1}/{epochs}, Weights: {self.weights}, Bias: {self.bias}")


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


and_output = np.array([0, 0, 0, 1])  # AND Gate
or_output = np.array([0, 1, 1, 1])   # OR Gate
nor_output = np.array([1, 0, 0, 0])  # NOR Gate


and_perceptron = Perceptron(input_size=2)
or_perceptron = Perceptron(input_size=2)
nor_perceptron = Perceptron(input_size=2)


print("Training for AND Gate:")
and_perceptron.train(inputs, and_output)

print("\nTraining for OR Gate:")
or_perceptron.train(inputs, or_output)

print("\nTraining for NOR Gate:")
nor_perceptron.train(inputs, nor_output)


print("\nTesting AND Gate:")
for x in inputs:
    print(f"Input: {x}, Output: {and_perceptron.forward(x)}")

print("\nTesting OR Gate:")
for x in inputs:
    print(f"Input: {x}, Output: {or_perceptron.forward(x)}")

print("\nTesting NOR Gate:")
for x in inputs:
    print(f"Input: {x}, Output: {nor_perceptron.forward(x)}")

