import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        trainset,
        testset,
        num_epochs=5,
        batch_size=16,
        init_lr=1e-3,
        device="cpu",
    ):
        self.model = model.to(device)
        self.trainset = trainset
        self.testset = testset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.device = device

        self.train_loss_per_epoch = []
        self.train_accuracy_per_epoch = []
        self.test_loss_per_epoch = []
        self.test_accuracy_per_epoch = []

    def train(self):
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adamax(self.model.parameters(), lr=self.init_lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0
            correct = 0
            total = 0
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")
                for idx, data in enumerate(tepoch):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total += len(labels)
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    running_loss += loss.item()
                    tepoch.set_postfix(
                        loss=running_loss / (idx + 1), accuracy=correct / total
                    )
            scheduler.step()
            self.train_loss_per_epoch.append(running_loss / len(trainloader))
            self.train_accuracy_per_epoch.append(correct / total)

            # validation
            self.model.eval()
            with torch.no_grad():
                test_loss = 0
                test_correct = 0
                test_total = 0
                for idx, data in enumerate(testloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, labels)
                    test_loss += loss.item()
                    test_total += len(labels)
                    test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                print(
                    f"Epoch {epoch + 1}: Validation Loss: {test_loss / len(testloader):.2f}, Validation Accuracy: {test_correct / test_total:.3f}"
                )
                self.test_loss_per_epoch.append(test_loss / len(testloader))
                self.test_accuracy_per_epoch.append(test_correct / test_total)

    def get_training_history(self):
        return (
            self.train_loss_per_epoch,
            self.train_accuracy_per_epoch,
            self.test_loss_per_epoch,
            self.test_accuracy_per_epoch,
        )

    def predict(self, testloader):
        self.model.eval()
        predict_probs = []
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                predict_probs.append(F.softmax(outputs, dim=1))
                predictions.append(outputs.argmax(dim=1))
                ground_truth.append(labels)

        return (
            torch.cat(predict_probs).cpu(),
            torch.cat(predictions).cpu(),
            torch.cat(ground_truth).cpu(),
        )
