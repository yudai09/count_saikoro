# -*- using:utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet50
from tensorboardX import SummaryWriter
from dataset import SaikoroImageDataSet
import numpy as np
import os

import win_unicode_console
from memory_profiler import profile

from pretrainedmodels import inceptionv4


class NetResnet50(torch.nn.Module):
    def __init__(self):
        super(NetResnet50, self).__init__()
        self.cnn = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        # self.cnn = resnet50(pretrained=True)

        self.fc = nn.Sequential(
            nn.Linear(2048 * 4 * 5, 126),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(126, 6),
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


def main():
    # workaround a bug for windows
    # https://bugs.python.org/issue32245
    win_unicode_console.enable()

    dataset = SaikoroImageDataSet()

    train_dataset, test_dataset = dataset.train_test_split(train_split=0.8)

    model = NetResnet50()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()

    if torch.cuda.is_available():
        print('use cuda.')
        print('loading model.')
        model = model.cuda()
        print('loading criterion.')
        criterion = criterion.cuda()
        print('loading finished.')

    for trainable, epochs in [(False, 50), (True, 100)]:
    # for trainable, epochs in [(True, 200)]:
    # for trainable in [True]:
        model.setCNNTrainable(trainable)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-7)
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
        train_test(model, criterion, optimizer, train_dataset, test_dataset, epochs)


def train_test(model, criterion, optimizer, train_dataset, test_dataset, epochs, k=0):
    # model path to save weights
    model_path = 'models/model_{}.pth'.format(k)
    # tensorboard summary writer
    writer_train = SummaryWriter('./logs/train/{}'.format(k))
    writer_test = SummaryWriter('./logs/test/{}'.format(k))
    # calculate best test accuracy to save best model weights
    best_accuracy = 0.0

    for epoch in range(epochs):
        # pytorch bug: memory leaks when shuffle is True.
        # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        train_loss, train_accuracy = process(model, train_dataloader, optimizer, criterion, train=True)
        test_loss, test_accuracy = process(model, test_dataloader, optimizer, criterion, train=False)

        print('[%.2d epoch %.3d/%.3d] loss(train): %.5f, accuracy(train) %.5f, loss(val) %.5f, accuracy(val) %.5f' %
              (k, epoch + 1, epochs, train_loss, train_accuracy, test_loss, test_accuracy))

        writer_train.add_scalar('loss', train_loss, epoch)
        writer_test.add_scalar('loss', test_loss, epoch)
        writer_train.add_scalar('acc', train_accuracy, epoch)
        writer_test.add_scalar('acc', test_accuracy, epoch)

        if test_accuracy > best_accuracy:
            print("accuracy improved. %.3f -> %.3f" % (best_accuracy, test_accuracy))
            best_accuracy = test_accuracy
            if test_accuracy >= 0.6:
                print("save model to {}".format(model_path))
                model_save(model, model_path)
                writer_test.add_scalar('best_acc', test_accuracy, epoch)


def process(model, dataloader, optimizer, criterion, train=True):
    loss_list = []
    correct_list = torch.Tensor([])
    if torch.cuda.is_available():
        correct_list = correct_list.cuda()

    for batch_iter, batch_data in enumerate(dataloader):
        # print(str(batch_data).encode('cp932', 'ignore'))
        image = batch_data[0]
        label = batch_data[1]

        image = Variable(image, requires_grad=False)
        label = Variable(label, requires_grad=False)

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        if train:
            model.train()
        else:
            model.eval()

        # forward + backward + optimize
        outputs = model.forward(image)

        # https://github.com/pytorch/pytorch/issues/3670
        # label should be 1d tensor.
        outputs = outputs.squeeze(1)
        label = label.squeeze(1)

        # for windows
        label = label.long()

        loss = criterion(outputs, label)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # for classification
        # argmax
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == label.data).float()

        loss_list.append(loss.data[0])
        # print(predicted)
        # print(correct)
        # print(correct_list.size())
        correct_list = torch.cat([correct_list, correct], 0)

    loss_ave = np.mean(loss_list)
    accuracy_ave = correct_list.sum() / correct_list.size()[0]

    return loss_ave, accuracy_ave


def model_save(model, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    main()
