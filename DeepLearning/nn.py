import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((96, 96)),  # Resize the image to 96x96
    transforms.ToTensor(),  # Convert the image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 21 * 21, 120)  # Adjusted based on input size after convolution and pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 21 * 21)  # Adjusted based on input size after convolution and pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
