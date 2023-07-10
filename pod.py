__all__ = ['Pod']

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

# Move the input data and model parameters to the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoD:
  def __init__(self, goal_size=5, inference_size=7):
    self.goal_size = goal_size
    self.window_size = inference_size
    self.goals = []
    
    # Define the model and loss function
    self.model = TinyCNN()
    self.model.to(device)
    
    self.pixel_locations = []
    for i in range(goal_size):
      for j in range(goal_size):
        self.pixel_locations.append((i+1, j+1))

  def add_goal(self, goal_array):
    goal0 = torch.tensor(goal_array, dtype=torch.float).float()
    self.goals.append(goal0)

    states, actions = noisify(goal0, self.pixel_locations, 5000)

    inputs = torch.stack(states).unsqueeze(1)
    inputs = inputs.to(device)
    self.inputs = inputs

    labels = torch.tensor(actions, dtype=torch.float).float().unsqueeze(1)  # Add extra dimension for single category output
    labels = labels.to(device)
    self.labels = labels
    return
  
  def _initialise(self):
    def initialize_weights(m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    self.model.apply(initialize_weights)

  def train(self):
    self._initialise()
    loss_function = nn.MSELoss()  # Binary Cross Entropy loss

    # Define the optimizer
    optimizer = optim.Adam(self.model.parameters(), lr=0.03)

    # Training loop
    num_epochs = 100
    for _ in range(num_epochs):
        # Forward pass
        outputs = self.model(self.inputs)
        loss = loss_function(outputs, self.labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save the trained model
    torch.save(self.model.state_dict(), 'tiny_cnn_model.pth')
    return
    
  def infer(self, level_array, mask_array=None, rounds=500):
    with torch.no_grad():
      current = torch.tensor(level_array, dtype=torch.float).float()
      mask = None
      height = current.shape[0]
      if mask_array: mask = torch.tensor(mask_array, dtype=torch.float).float()

      for _ in range(rounds):
          location = tuple(random.randint(1, height-2) for _ in range(2))
          state = sample_tensor(location[0], location[1], current)[None,None,:]
          if (mask is None) or (mask[location] != 0):
              current[location] = torch.round(self.model(state))

    return current


def noisify(goal, pixel_locations, rounds = 5000):
  padded_goal = pad(goal)
  steps = []
  states = []
  actions = []
  locations = []

  current = padded_goal.clone().detach()
  location_list = pixel_locations.copy()
  random.shuffle(location_list)
  
  for _ in range(rounds):
      noisy = torch.randint(2,size=padded_goal.size())
      while location_list:
          location = location_list.pop()
          current[location] = noisy[location]
          step = current.clone().detach()
          steps.append(step)
          states.append(sample_tensor(location[0], location[1], step))
          actions.append(padded_goal[location])
          locations.append(location)
  return states, actions

def sample_tensor(x, y,tensor,sample_size=3):
  # Calculate the coordinates for slicing the smaller tensor
  x_start = x - sample_size // 2
  x_end = x_start + sample_size
  y_start = y - sample_size // 2
  y_end = y_start + sample_size

  # Extract the smaller tensor from the larger tensor
  samples = tensor[x_start:x_end, y_start:y_end]
  return samples

def pad(tensor):
  padding = (1, 1)
  padded_tensor = tensor.clone().detach()
  padded_tensor = F.pad(padded_tensor, padding, mode='replicate')
  padded_tensor = torch.transpose(padded_tensor, 0, 1)
  padded_tensor = F.pad(padded_tensor, padding, mode='replicate')
  padded_tensor = torch.transpose(padded_tensor, 0, 1)
  return padded_tensor

# Define the CNN model
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
