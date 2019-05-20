import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    # def __init__(self):
    #     super(Net, self).__init__()
        
    #     self.conv1 = nn.Conv2d(1, 32, 5)
    #     self.conv2 = nn.Conv2d(32, 64, 5)
    #     self.conv3 = nn.Conv2d(64, 128, 5)
    #     self.conv4 = nn.Conv2d(128, 256, 5)
        
    #     self.fc1 = nn.Linear(25600, 136)
        
    #     self.pool = nn.MaxPool2d(2, 2)
        
    #     self.dropout = nn.Dropout(p=0.4)
        
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = self.pool(F.relu(self.conv3(x)))
    #     x = self.pool(F.relu(self.conv4(x)))
    #     x = x.view(x.size(0), -1)
    #     x = self.dropout(x)
    #     x = self.fc1(x)
        
    #     return x

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(224720, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(224720)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x