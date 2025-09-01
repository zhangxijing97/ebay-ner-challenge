# neural_network.py
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        """
        layers: [(in_dim, 'ReLU'), (hidden, 'ReLU'), ..., (num_classes, 'Softmax')]
        """
        self.layers = layers
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        params = {}
        L = len(self.layers)
        for l in range(1, L):
            in_dim = self.layers[l-1][0]
            out_dim = self.layers[l][0]
            params[f'W{l}'] = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
            params[f'b{l}'] = np.zeros((1, out_dim))
        return params

    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, a): return (a > 0).astype(a.dtype)  # 用激活后a更省代码
    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def get_activation(self, name):
        return self.relu if name == 'ReLU' else self.softmax

    def forward(self, X):
        cache = {'A0': X}
        L = len(self.layers)
        for l in range(1, L):
            W, b = self.parameters[f'W{l}'], self.parameters[f'b{l}']
            Z = np.dot(cache[f'A{l-1}'], W) + b
            cache[f'Z{l}'] = Z
            act = self.get_activation(self.layers[l][1])
            cache[f'A{l}'] = act(Z)
        return cache

    def compute_loss(self, Y_onehot, Y_hat):
        # 交叉熵（多分类）
        eps = 1e-12
        return -np.mean(np.sum(Y_onehot * np.log(Y_hat + eps), axis=1))

    def train(self, X, Y_onehot, epochs, lr):
        L = len(self.layers); m = X.shape[0]
        for epoch in range(epochs):
            cache = self.forward(X)
            Y_hat = cache[f'A{L-1}']
            loss = self.compute_loss(Y_onehot, Y_hat)

            # 反向传播
            dA = Y_hat - Y_onehot  # Softmax+CE 的简洁梯度
            for l in reversed(range(1, L)):
                A_prev = cache[f'A{l-1}']
                if self.layers[l][1] == 'Softmax':
                    dZ = dA
                else:  # ReLU
                    dZ = dA * self.relu_derivative(cache[f'A{l}'])
                dW = (A_prev.T @ dZ) / m
                db = np.sum(dZ, axis=0, keepdims=True) / m
                if l > 1:
                    dA = dZ @ self.parameters[f'W{l}'].T
                self.parameters[f'W{l}'] -= lr * dW
                self.parameters[f'b{l}'] -= lr * db

            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
