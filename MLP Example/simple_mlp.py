import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

class SimpleMLP:
    def __init__(self, input_size = 2, hidden_size = [50, 50], output_size = 1):
        self.W1 = torch.randn(input_size, hidden_size[0], requires_grad=True)
        self.b1 = torch.randn(1, hidden_size[0], requires_grad=True)
        self.W2 = torch.randn(hidden_size[0], hidden_size[1], requires_grad=True)
        self.b2 = torch.randn(1, hidden_size[1], requires_grad=True)
        self.W3 = torch.randn(hidden_size[1], output_size, requires_grad=True)
        self.b3 = torch.randn(1, output_size, requires_grad=True)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = torch.relu(self.z1)
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = torch.relu(self.z2) 
        self.z3 = torch.matmul(self.a2, self.W3) + self.b3
        self.a3 = torch.sigmoid(self.z3)  # Final output activation
        return self.a3
    
    def backward(self, X, y, output, lr=0.01):
        m = X.shape[0]

        # Compute gradients for the output layer
        dz3 = output - y
        dW3 = torch.matmul(self.a2.T, dz3) / m
        db3 = torch.sum(dz3, axis=0) / m

        # Compute gradients for the second hidden layer
        da2 = torch.matmul(dz3, self.W3.T)
        dz2 = da2 * (self.a2 > 0)  # Derivative of ReLU
        dW2 = torch.matmul(self.a1.T, dz2) / m
        db2 = torch.sum(dz2, axis=0) / m

        # Compute gradients for the first hidden layer
        da1 = torch.matmul(dz2, self.W2.T)
        dz1 = da1 * (self.a1 > 0)  # Derivative of ReLU
        dW1 = torch.matmul(X.T, dz1) / m
        db1 = torch.sum(dz1, axis=0) / m

        # Update weights and biases using gradient descent
        with torch.no_grad():
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W3 -= lr * dW3
            self.b3 -= lr * db3

    def train(self, X, y, epochs=1000, lr=0.01):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            #Compute loss using (Mean Squared Error)
            loss = torch.mean((output - y) ** 2)
            losses.append(loss.item())
            #update weights
            self.backward(X, y, output, lr)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        return losses
    

input_size = 2
hidden_size = [50,50]
output_size = 1
model = SimpleMLP(input_size, hidden_size, output_size)

#Train  model and store the losses
losses = model.train(X_train, y_train, epochs=1000, lr=0.1)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()
