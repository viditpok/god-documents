import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        """instantiates the CNN model

        HINT: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        super(CNN, self).__init__()

        self.feature_extractor = nn.Sequential(

            ...
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(

            ...
        )
        """
        super(CNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """runs the forward method for the CNN model

        Args:
            x (torch.Tensor): input tensor to the model

        Returns:
            torch.Tensor: output classification tensor of the model
        """

        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
