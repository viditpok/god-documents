import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(20 * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 15),
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        model_output = self.fc_layers(x)
        return model_output