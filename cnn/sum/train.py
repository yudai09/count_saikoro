import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from SaikoroCombDataset import SaikoroCombDataset


class MyModel(torch.nn.Module):
    def __init__(self, nb_class):
        super(MyModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.Conv2d(64, 64, 5),
            nn.Conv2d(64, 128, 5))

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.Conv2d(64, 64, 5),
            nn.Conv2d(64, 128, 5))

        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 20 * 2, 120),
            nn.Linear(120, 84),
            nn.Linear(84, nb_class),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(y)

        x1 = x1.view(-1, 128 * 20 * 20)
        x2 = x2.view(-1, 128 * 20 * 20)

        x1x2 = torch.cat((x1, x2), 1)

        return self.fc(x1x2)


def main():
    # eye sum in range (2-12) but set nb_class 13.
    # it's trivial difference.
    model = MyModel(13)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    if torch.cuda.is_available():
        print('use cuda.')
        print('loading model.')
        model = model.cuda()
        print('loading criterion.')
        criterion = criterion.cuda()
        print('loading finished.')

    train_dataset = SaikoroCombDataset(train=True)
    test_dataset = SaikoroCombDataset(train=False)

    running_loss = 0
    batch_size = 32

    for epoch in range(100):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        running_loss = 0.0

        for batch_iter, batch_data in enumerate(train_dataloader):
            image1, image2, label = batch_data

            image1 = Variable(image1, requires_grad=True)
            image2 = Variable(image2, requires_grad=True)
            label = Variable(label, requires_grad=False)

            if torch.cuda.is_available():
                image1 = image1.cuda()
                image2 = image2.cuda()
                label = label.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image1, image2)

            # https://github.com/pytorch/pytorch/issues/3670
            # label should be 1d tensor.
            label = label.squeeze(1)

            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        print('[%d] test loss: %.3f' %
              (epoch + 1, running_loss / (len(train_dataset) / batch_size)))

        running_loss = 0
        for batch_iter, batch_data in enumerate(test_dataloader):
            image1, image2, label = batch_data

            image1 = Variable(image1, requires_grad=True)
            image2 = Variable(image2, requires_grad=True)
            label = Variable(label, requires_grad=False)

            if torch.cuda.is_available():
                image1 = image1.cuda()
                image2 = image2.cuda()
                label = label.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image1, image2)

            # https://github.com/pytorch/pytorch/issues/3670
            # label should be 1d tensor.
            label = label.squeeze(1)
            loss = criterion(outputs, label)
            # do not backward in test mode
            # loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        print('[%d] validation loss: %.3f' %
              (epoch + 1, running_loss / (len(test_dataset) / batch_size)))


if __name__ == '__main__':
    main()
