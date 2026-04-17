import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)

        # Block 2
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)

        # Block 3
        self.conv4 = nn.Conv2d(16, 32, 3)
        self.conv5 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)

        # Block 4
        self.conv6 = nn.Conv2d(32, 64, 3)
        self.conv7 = nn.Conv2d(64, 64, 3)
        self.conv8 = nn.Conv2d(64, 64, 3)
        self.conv9 = nn.Conv2d(64, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)

        # Block 5
        self.conv10 = nn.Conv2d(64, 128, 5)
        self.bn5 = nn.BatchNorm2d(128)

        # Block 6
        self.conv11 = nn.Conv2d(128, 256, 5)
        self.bn6 = nn.BatchNorm2d(256)

        # Fully connected
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):

        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Block 2
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(self.conv3(x)))

        # Block 3
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn3(self.conv5(x)))

        # Block 4
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.bn4(self.conv9(x)))

        # Block 5
        x = F.relu(self.bn5(self.conv10(x)))

        # Block 6
        x = F.relu(self.bn6(self.conv11(x)))

        # Pool
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))

        return x