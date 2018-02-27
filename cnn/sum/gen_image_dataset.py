# -*- coding:utf-8 -*-
import chainer
from chainer.datasets import ImageDataset, TupleDataset, LabeledImageDataset
import glob
import numpy as np

def gen_image_dataset():
    sumeyes_photos = []
    for photo in glob.glob("./images/*.jpeg"):
        print("{}".format(photo))
        # eyes = ['1','1'], ['1','2'], ..., ['6','6']
        eyes = photo[:-len('.jpeg')].split('_')[-2:]
        sum_eyes = int(eyes[0]) + int(eyes[1])
        sumeyes_photos.append((photo, sum_eyes))

    return LabeledImageDataset(sumeyes_photos)

if __name__ == '__main__':
    dataset = gen_image_dataset()
    print(dataset[1])
    train, test = chainer.datasets.get_cross_validation_datasets(gen_image_dataset(), 2)
