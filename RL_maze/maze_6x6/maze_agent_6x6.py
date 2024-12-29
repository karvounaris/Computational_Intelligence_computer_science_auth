import torch
import torch.nn as nn

class DQNCNN(nn.Module):
    def __init__(self):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        conv_output_size = 32 * 6 * 6
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x

class DuelingDQNCNN(nn.Module):
    def __init__(self):
        super(DuelingDQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        conv_output_size = 32 * 6 * 6
        self.value_fc1 = nn.Linear(conv_output_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.value_fc2 = nn.Linear(128, 1)
        self.advantage_fc1 = nn.Linear(conv_output_size, 128)
        self.ln2 = nn.LayerNorm(128)
        self.advantage_fc2 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        val = self.relu(self.ln1(self.value_fc1(x)))
        val = self.value_fc2(val)
        adv = self.relu(self.ln2(self.advantage_fc1(x)))
        adv = self.advantage_fc2(adv)
        q_values = val + adv - adv.mean(1, keepdim=True)
        return q_values

class InceptionDQNCNN(nn.Module):
    def __init__(self):
        super(InceptionDQNCNN, self).__init__()
        self.branch1x1 = nn.Conv2d(1, 16, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(16)
        self.branch3x3 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn3x3 = nn.BatchNorm2d(16)
        self.branch5x5 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn5x5 = nn.BatchNorm2d(16)
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        b1x1 = self.relu(self.bn1x1(self.branch1x1(x)))
        b3x3 = self.relu(self.bn3x3(self.branch3x3(x)))
        b5x5 = self.relu(self.bn5x5(self.branch5x5(x)))
        b_pool = self.relu(self.bn1x1(nn.Conv2d(1, 16, kernel_size=1)(self.branch_pool(x))))
        x = torch.cat([b1x1, b3x3, b5x5, b_pool], dim=1)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x