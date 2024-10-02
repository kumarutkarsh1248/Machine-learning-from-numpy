import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Create a dataset
# Generating some random data
np.random.seed(0)
x = np.random.rand(100, 1) * 10  # 100 random points in the range [0, 10]
y = 2 * x + 1 + np.random.randn(100, 1)  # y = 2x + 1 + noise

# Convert numpy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Step 3: Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.linear(x)

# Create an instance of the model
model = LinearRegressionModel()

# Step 4: Set the Loss Function and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Step 5: Train the Model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Make Predictions
with torch.no_grad():
    predicted = model(x_tensor).numpy()

# Plot the results
plt.scatter(x, y, label='Original data')
plt.plot(x, predicted, color='red', label='Fitted line')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with PyTorch')
plt.show()
