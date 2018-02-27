# -*- coding:utf-8 -*-
import chainer
import glob
from collections import defaultdict
from chainer.datasets import LabeledImageDataset, TransformDataset

def gen_image_dataset():
    # There are 12 images for each eye.
    nr_train = 10
    nr_test = 2

    eye_photos = defaultdict(list)
    for photo in glob.glob("./images/*.jpeg"):
        eye = photo[:-len('.jpeg')].split('_')[-1:][0]
        eye_photos[eye].append(photo)

    train_photo_label = [] # [(path_to_image, label), ...]
    test_photo_label = []
    for eye in eye_photos.keys():
        # eye 1-6 -> 0-5 for computing
        photo_eye = [(photo, int(eye)-1) for photo  in eye_photos[eye]]
        train_photo_label += photo_eye[0:nr_train]
        test_photo_label += photo_eye[0:nr_test]

    def transform(data):
        x, label = data
        x = x / 255
        return x, label

    return TransformDataset(LabeledImageDataset(train_photo_label), transform), \
        TransformDataset(LabeledImageDataset(test_photo_label), transform)
