# Importar las bibliotecas necesarias
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        # Inicialización de pesos y sesgos usando He initialization
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2. / layers[i])
            bias = np.zeros(layers[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)

    def feedforward(self, X):
        activation = X
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activation, weight) + bias
            activation = self.relu(z)
        return activation

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(X)):
                self.backpropagation(X[i], y[i], learning_rate)

    def backpropagation(self, x, y, learning_rate):
        activations = [x]
        zs = []

        # Feedforward
        activation = x
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activation, weight) + bias
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)

        # Backpropagation
        delta = self.cost_derivative(activations[-1], y) * self.relu_derivative(zs[-1])
        self.weights[-1] -= learning_rate * np.outer(activations[-2], delta)
        self.biases[-1] -= learning_rate * delta

        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.relu_derivative(z)
            delta = np.dot(delta, self.weights[-l + 1].T) * sp
            self.weights[-l] -= learning_rate * np.outer(activations[-l - 1], delta)
            self.biases[-l] -= learning_rate * delta
    
    # Definir la función para crear el modelo de la red neuronal
    @staticmethod
    def crear_modelo(input_dim):
        model = Sequential()
        model.add(Dense(16, input_dim=input_dim, activation='relu'))  # Primera capa oculta
        model.add(Dense(8, activation='relu'))  # Segunda capa oculta
        model.add(Dense(1))  # Capa de salida
        optimizer = Adam(learning_rate=0.01)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model
    
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)