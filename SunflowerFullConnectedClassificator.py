import torch.nn as nn
import torch.nn.functional as F


class SunflowerFullConnectedClassificator(nn.Module):
    def __init__(self, input_size, output_size):
        super(SunflowerFullConnectedClassificator, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)