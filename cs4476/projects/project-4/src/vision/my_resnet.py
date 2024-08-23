import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """

        super().__init__()

        original_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        num_features = original_resnet.fc.in_features
        original_resnet.fc = nn.Linear(num_features, 15)

        self.conv_layers = nn.Sequential(*list(original_resnet.children())[:-2])

        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            original_resnet.fc,
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

        for param in self.conv_layers.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """

        x = x.repeat(1, 3, 1, 1)
        x = self.conv_layers(x)
        model_output = self.fc_layers(x)
        return model_output
