import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet50

class NetResnet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(NetResnet50, self).__init__()
        self.cnn = nn.Sequential(*list(resnet50(pretrained=pretrained).children())[:-2])
        # self.cnn = resnet50(pretrained=True)

        self.fc = nn.Sequential(
            nn.Linear(2048 * 4 * 5, 126),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(126, 7),
            nn.ReLU(),
        )

    def forward(self, X):
        X = self.cnn(X)
        X = X.view(-1, 2048 * 4 * 5)
        return self.fc(X)

    def setCNNTrainable(self, trainable):
        if trainable:
            print("train CNN")
            for p in self.cnn.parameters():
                p.requires_grad = True
        else:
            print("do not train CNN")
            for p in self.cnn.parameters():
                p.requires_grad = False
