import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score 

#preprocessing
encodings = {'a' : [1,0,0,0], 't': [0,1,0,0], 'c': [0,0,1,0], 'g': [0,0,0,1]}
headers = ["outcome", "id", "nucleotides"]

data = pd.read_csv('data.csv')
data.to_csv('data.csv', header = headers, index = False)
data = pd.read_csv('data.csv')

def encode(nucleotides):
    return np.array([encodings[nucleotide] for nucleotide in nucleotides])

X = data['nucleotides']
X = np.array([encode(nucleotide) for nucleotide in X]).reshape(len(X),-1)

y = data['outcome'].map({'+':1, '-': 0}).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = True, random_state = 42)
X_train = X_train.T
X_test = X_test.T

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(32, 228) * np.sqrt(1 / 228)
        self.W2 = np.random.randn(2,32) * np.sqrt(1 / 32)
        self.B1 = np.zeros((32,1))
        self.B2 = np.zeros((2,1))

    @staticmethod
    def RELU(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    @staticmethod
    def dropout(A, dropout_rate):
        mask = (np.random.rand(*A.shape) > dropout_rate) / (1.0 - dropout_rate)
        return A * mask

    def forward(self, x, dropout_rate=0.2):
        self.Z1 = self.W1.dot(x) + self.B1
        self.A1 = NeuralNetwork.RELU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.B2
        self.A2 = NeuralNetwork.softmax(self.Z2)
        return self.Z1, self.A1, self.Z2, self.A2

    @staticmethod
    def one_hot(Y):
        onehoty = np.zeros((2, Y.size))
        onehoty[Y, np.arange(Y.size)] = 1
        return onehoty

    @staticmethod
    def deriv_Relu(X):
        return X > 0

    def backward(self, X, Y):
        onehoty = NeuralNetwork.one_hot(Y.astype(int))
        m = Y.size
        dZ2 = self.A2 - onehoty
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        dB2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * NeuralNetwork.deriv_Relu(self.Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        dB1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, dB1, dW2, dB2

    def update_params(self, dW1, dB1, dW2, dB2, alpha):

        if not hasattr(self, 'vW1'):
            self.vW1 = np.zeros_like(self.W1)
            self.vW2 = np.zeros_like(self.W2)
            self.vB1 = np.zeros_like(self.B1)
            self.vB2 = np.zeros_like(self.B2)

        beta = 0.9
        self.vW1 = beta * self.vW1 + (1 - beta) * dW1
        self.vW2 = beta * self.vW2 + (1 - beta) * dW2
        self.vB1 = beta * self.vB1 + (1 - beta) * dB1
        self.vB2 = beta * self.vB2 + (1 - beta) * dB2

        self.W1 -= alpha * self.vW1
        self.W2 -= alpha * self.vW2
        self.B1 -= alpha * self.vB1
        self.B2 -= alpha * self.vB2

        return self.W1, self.W2, self.B1, self.B2

    @staticmethod
    def getPredictions(A2):
        return np.argmax(A2, axis=0)

    @staticmethod
    def get_accuracy(predictions, Y):
        return np.mean(predictions == Y)

    def train(self, X, y, epochs, alpha):
        for i in range(epochs):
            current_alpha = alpha
            Z1, A1, Z2, A2 = self.forward(X)
            dW1, dB1, dW2, dB2 = self.backward(X, y)
            self.W1, self.W2, self.B1, self.B2 = self.update_params(dW1, dB1, dW2, dB2, current_alpha)

    def test(self, X, y):
        tz1, ta1, tz2, ta2 = self.forward(X, dropout_rate = 0.0)
        print("Test Accuracy: ", NeuralNetwork.get_accuracy(NeuralNetwork.getPredictions(ta2), y))

class FederatedLearning:
    def __init__(self, num_clients, model, epochs, local_epochs, alpha):
        self.num_clients = num_clients
        self.model = NeuralNetwork()
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.alpha = alpha
        self.local_model = [NeuralNetwork() for client in range(self.num_clients)]

    def split_data(self, X, y):
        data = X.shape[1]//self.num_clients

        X_split = []
        y_split = []

        for i in range(self.num_clients):
            starting_index = i*data
            ending_index = starting_index + data
            X_split.append(X[:, starting_index:ending_index])
            y_split.append(y[starting_index:ending_index])
        return X_split, y_split

    def local_attributes(self):
        for model in self.local_model:
            model.W1 = np.copy(self.model.W1)
            model.W2 = np.copy(self.model.W2)
            model.B1 = np.copy(self.model.B1)
            model.B2 = np.copy(self.model.B2)

    def combine_local_weights(self):
        comb_W1 = np.zeros_like(self.model.W1)
        comb_W2 = np.zeros_like(self.model.W2)
        comb_B1 = np.zeros_like(self.model.B1)
        comb_B2 = np.zeros_like(self.model.B2)

        for model in self.local_model:
            comb_W1 += model.W1
            comb_W2 += model.W2
            comb_B1 += model.B1
            comb_B2 += model.B2
        num_clients = self.num_clients
        self.model.W1 = comb_W1 / num_clients
        self.model.W2 = comb_W2 / num_clients
        self.model.B1 = comb_B1 / num_clients
        self.model.B2 = comb_B2 / num_clients

    def train(self, X, y):
        accuracy = 0
        X_split, y_split = self.split_data(X, y)
        for epoch in range(self.epochs):
            self.local_attributes()
            
            for i, model in enumerate(self.local_model):
                model.train(X_split[i], y_split[i], self.local_epochs,  self.alpha)
            self.combine_local_weights()
            Z1, A1, Z2, A2 = self.model.forward(X)
            predictions = self.model.getPredictions(A2)
            accuracy = NeuralNetwork.get_accuracy(predictions, y)

        print("Accuracy: " + str(accuracy))

    def test(self, X, y):
        tz1, ta1, tz2, ta2 = self.model.forward(X, dropout_rate = 0.0)
        print("Test Accuracy: ", NeuralNetwork.get_accuracy(NeuralNetwork.getPredictions(ta2), y))
        print(f"Test F1: {f1_score(NeuralNetwork.getPredictions(ta2), y)}")


num_clients = 5
epochs = 10
local_epochs = 20
alpha = 0.01
federated_learning = FederatedLearning(num_clients, NeuralNetwork, epochs, local_epochs, alpha)
federated_learning.train(X_train, y_train)
federated_learning.test(X_test,y_test)
