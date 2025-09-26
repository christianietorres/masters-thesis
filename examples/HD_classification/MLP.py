import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    """
    A class for creating 
    the Multilayer Perceptron (MLP) 
    neural network (NN) model for tabular data.

    Attributes:
        fc1 (torch.nn.Linear): First hidden layer (13 to 64).
        fc2 (torch.nn.Linear): second hidden layer (64 to 32).
        fc3 (torch.nn.Linear): Third hidden layer (32 to 16).
        fc4_b (torch.nn.Linear): Output layer for binary classification (16 to 1).
        fc4 (torch.nn.Linear): Output layer for multiclass classification (16 to 5).
        binary (bool): Whether to use binary classification output.
    """

    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.binary = True
        if self.binary == True:
            self.fc4_b = nn.Linear(16, 1)
        else:
            self.fc4 = nn.Linear(16, 5)
        
    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward propragation on 
        the Multilayer Perceptron (MLP) neural network (NN) model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 13).
        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, 1 or 5).
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.binary == True:
            x = self.fc4_b(x)
        else:    
            x = self.fc4(x)   
        return x