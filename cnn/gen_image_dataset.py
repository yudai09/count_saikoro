# -*- coding:utf-8 -*-
import chainer
import glob
from collections import defaultdict
from chainer.datasets import LabeledImageDataset

def gen_image_dataset():
    # There are 12 images for each eye combinations.
    nr_train = 10
    nr_test = 2

    eyes_photos = defaultdict(list)
    for photo in glob.glob("./images/*.jpeg"):
        print("{}".format(photo))
        # eyes = ['1','1'], ['1','2'], ..., ['6','6']
        eyes = photo[:-len('.jpeg')].split('_')[-2:]
        eyes = "{}_{}".format(eyes[0],eyes[1])
        eyes_photos[eyes].append(photo)

    train_photo_label = [] # [(path_to_image, label), ...]
    test_photo_label = []
    label_eyes = [] # [(label, eyes), ...]
    for label, eyes in enumerate(eyes_photos.keys()):
        print("label {}, eye {}, path {}".format(label, eyes, eyes_photos[eyes]))
        photo_label = [(photo, label) for photo  in eyes_photos[eyes]]
        train_photo_label += photo_label[0:nr_train]
        test_photo_label += photo_label[0:nr_test]
        label_eyes.append((label, eyes))

    print(train_photo_label)
    print(test_photo_label)
    print(label_eyes)

    return label_eyes, LabeledImageDataset(train_photo_label), LabeledImageDataset(test_photo_label)
