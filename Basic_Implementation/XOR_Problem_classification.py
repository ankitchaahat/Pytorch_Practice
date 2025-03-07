import torch
import torch.nn as nn
import torch.optim as optim


# XOR Input and Output Data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


class XOR_Net(nn.Module):
    def __init__(self):
        super(XOR_Net, self).__init__()
        self.hidden = nn.Linear(2, 4)  # Hidden layer with 4 neurons
        self.output = nn.Linear(4, 1)  # Output layer with 1 neuron
        self.activation = nn.Sigmoid() # Sigmoid activation function

    def forward(self, x):
        x = self.activation(self.hidden(x))  # Hidden layer with activation
        x = self.activation(self.output(x))  # Output layer with activation
        return x



model = XOR_Net()  
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent






epochs = 5000  # Number of iterations

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, Y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 500 == 0:  # Print loss every 500 epochs
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')



with torch.no_grad():
    predictions = model(X)
    predictions = (predictions > 0.5).float()  # Convert probabilities to 0 or 1
    print("Predictions:\n", predictions)




# Output 
[[0.], [1.], [1.], [0.]]

