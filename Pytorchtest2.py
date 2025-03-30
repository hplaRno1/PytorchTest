import torch
import torch.nn as nn

# Check if CUDA is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create two simple tensors
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=device)
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device=device)

print("Tensor x:")
print(x)
print("Tensor y:")
print(y)

# Perform basic operations: addition
sum_xy = x + y
print("Sum of x and y:")
print(sum_xy)

# Define a simple neural network with one fully connected layer
class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.fc = nn.Linear(2, 1)  # Input of size 2, output of size 1
    
    def forward(self, x):
        return self.fc(x)

# Instantiate the model and move it to the selected device
model = DummyNet().to(device)
print("DummyNet Model:")
print(model)

# Create dummy input data (batch of 2 samples, each with 2 features)
dummy_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
output = model(dummy_input)
print("Model output:")
print(output)

# Define a dummy target and compute the loss using Mean Squared Error
target = torch.tensor([[1.0], [0.0]], device=device)
criterion = nn.MSELoss()
loss = criterion(output, target)
print("Loss:")
print(loss.item())

# Perform a backward pass to compute gradients
loss.backward()
print("Backward pass completed.")

