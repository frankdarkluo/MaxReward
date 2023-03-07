import torch.nn as nn
import torch.optim as optim

import torch
from transformers import GPT2Model, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# Define input sequence
input_sequence = "This is an example sentence."

# Tokenize input sequence
inputs = tokenizer(input_sequence, return_tensors='pt')

# Generate embeddings for input sequence
outputs = model(**inputs)

# Extract last hidden state
last_hidden_state = outputs.last_hidden_state

print(last_hidden_state.shape)  # Output: torch.Size([1, 6, 768])


# Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(50257, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 50257)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define training data
X = last_hidden_state
y = torch.tensor([1, 2]).view(-1, 1)

# Define loss function and optimizer
criterion = nn.MSELoss()
net=Net()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# Train neural network
net = Net()
for epoch in range(1000):
    optimizer.zero_grad()
    output = net(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Test neural network
with torch.no_grad():
    test_output = net(X)
    print(test_output)
