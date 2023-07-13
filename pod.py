__all__ = ['Pod']

import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ai_framework import Learner, DataLoaders

# Move the input data and model parameters to the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StateActionDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = self.inputs[index]
        output_data = self.outputs[index]
        # Process the input and output data if needed
        return input_data, output_data
    
class PoD:
  def __init__(self, goal_size=5):
    self.goal_size = goal_size
    self.goals = []
    
    # Define the model and loss function
    self.model = TinyCNN()
    self.model.to(device)
    
    self.pixel_locations = get_pixel_locations(goal_size - 2)

  def add_goal(self, goal_array):
    goal0 = torch.tensor(goal_array, dtype=torch.float)
    self.goals.append(goal0)

    def generate_state_action_pairs(goal, rounds):
      pixel_locations = get_pixel_locations(goal.shape[0] - 2)

      states, actions = noisify(goal, pixel_locations, rounds)
      inputs = torch.stack(states).unsqueeze(1)
      inputs = inputs.to(device)

      labels = torch.tensor(actions, dtype=torch.float)[:, None]  # Add extra dimension for single category output
      labels = labels.to(device)
      return inputs, labels

    # Split your data into train and test subsets
    train_inputs, train_labels = generate_state_action_pairs(goal0, rounds=100)  # Training data
    valid_inputs, valid_outputs = generate_state_action_pairs(goal0, rounds=10)   # Validation data

    # Create separate instances of the dataset for train and test
    train_dataset = StateActionDataset(train_inputs, train_labels)
    valid_dataset = StateActionDataset(valid_inputs, valid_outputs)

    # Configure data loaders for train and test
    batch_size = 32
    shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    self.dls = DataLoaders(train_loader, valid_loader)
  
  def _initialise(self):
    def initialize_weights(m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    self.model.apply(initialize_weights)

  def train(self):
    learner = Learner(self.model, self.dls, lr=0.0015)
    self._initialise()
    learner.fit(10)

  def infer(self, level_array, mask_array=None, rounds=5000):
    with torch.no_grad():
      current = level_array.clone().detach() if torch.is_tensor(level_array) else torch.tensor(level_array, dtype=torch.float)
      current = current.to(device)

      mask = None
      height = current.shape[0]
      if mask_array: mask = torch.tensor(mask_array, dtype=torch.float)

      for _ in range(rounds):
          location = tuple(random.randint(1, height-2) for _ in range(2))
          state = sample_tensor(location[0], location[1], current)[None,None,:]
          if (mask is None) or (mask[location] != 0):
              current[location] = torch.round(self.model(state))

    return current.detach().cpu().numpy()

def noisify(goal, pixel_locations, rounds = 1,):
    padded_goal = goal #pad(goal)
    steps = []
    states = []
    actions = []
    locations = []

    current = padded_goal.clone().detach()
    
    for _i in range(rounds):
        location_list = pixel_locations.copy()
        random.shuffle(location_list)
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

def pad(tens):
  padding = (1, 1)
  padded_tensor = tens.clone().detach()
  padded_tensor = F.pad(padded_tensor, padding, mode='replicate')
  padded_tensor = torch.transpose(padded_tensor, 0, 1)
  padded_tensor = F.pad(padded_tensor, padding, mode='replicate')
  padded_tensor = torch.transpose(padded_tensor, 0, 1)
  return padded_tensor

# Define the CNN model
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def get_pixel_locations(size=5):
  # Record pixel locations
  pixel_locations = []
  for i in range(size):
      for j in range(size):
          pixel_locations.append((i+1, j+1))
  return pixel_locations