import argparse
import chainer
from chainer import links as L
from chainer import functions as F
from chainer import Variable
from chainer.datasets import ImageDataset, TransformDataset
import cv2
import numpy as np
import sys

from models.CNN import CNN

from gen_image_dataset import gen_image_dataset

def main(image_path):
    # image should be 32x32 and 3 channels
    model = L.Classifier(CNN())
    chainer.serializers.load_npz('./saikoro_model.npz', model)
    image = load_image(image_path)
    train, test = gen_image_dataset()

    # need np.expand_dims to input image to the classifier like mini-batch feeds.
    predict = F.softmax(model.predictor(np.expand_dims(image, axis=0))).data
    print(predict)
    print(np.argmax(predict))


def load_image(image_path):
    def transform(data):
        return data / 255

    return TransformDataset(ImageDataset([image_path]), transform)[0]

    # # below does not work property
    # image = cv2.imread(image_path)
    # # shape (32, 32, 3) -> (3, 32, 32)
    # image = image.transpose(2,0,1)
    # # 0-255 -> 0-1
    # image = image / 255
    # return np.expand_dims(image.astype(np.float32), axis=0)

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image', required=True,
                        help='path to saikoro image')
    args = parser.parse_args()
    main(args.image)
