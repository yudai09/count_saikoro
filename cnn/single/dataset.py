# -*- coding:utf-8 -*-
from collections import defaultdict
import glob
import itertools
import numpy
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re


class SaikoroImageDataSet(Dataset):
    def __init__(self, X=None, y=None, train=True):
        self.augmentor = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=30),
                                             transforms.RandomVerticalFlip()])
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.train = train

        if X is not None and y is not None:
            self.X = X
            self.y = y
        else:
            files = "../images/*.jpg"
            files = glob.glob(files)
            self.X = files
            self.y = [int(file.split(".jpg")[0].split("_")[1]) for file in files]
            print(self.X)
            print(self.y)

    def train_test_split(self, train_split=0.8):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=1-train_split, random_state=51
        )
        return (SaikoroImageDataSet(X_train, y_train, train=True),
                SaikoroImageDataSet(X_test, y_test, train=False))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_path = self.X[idx]
        label = self.y[idx]
        image = self.load_image(image_path)
        return image, torch.from_numpy(numpy.expand_dims(label, axis=0))

    def load_image(self, image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = image.convert('RGB')
        if self.train:
            image = self.augmentor(image)
        image = self.loader(image).float()
        return image / 255


if __name__ == "__main__":
    # if os.name == 'nt':
    #     win_unicode_console.enable()

    SaikoroImageDataSet()
