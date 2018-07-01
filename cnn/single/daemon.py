import argparse
import numpy as np
import sys

import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms, utils
from model import NetResnet50


# import picamera
from PIL import Image
# from PIL import ImageOps
import time

import numpy

# import serial

import cv2


def main(sleep=10):
    print("loading model..")
    model = load_model()
    model.eval()

    print("load model finish")
    cap = cv2.VideoCapture(0)
    loader = transforms.Compose([transforms.ToTensor()])
    # ser = serial.Serial('/dev/ttyACM0', 9600)

    while(True):
        # 読み捨て
        for i in range(10):
            ret, frame = cap.read()
        # capture_camera(camera, image_path)
        # Our operations on the frame come here

        # convert OpenCV image to PIL image
        image = Image.fromarray(frame)
        image = image.convert('L')
        image = image.convert('RGB')

        print(image.size)
        # original size is  480, 360
        image = image.crop((160, 120, 480, 360))
        image = image.resize((160, 120), resample=Image.BICUBIC)

        image.show()

        image_tensor = loader(image).float()
        image_tensor = image_tensor / 255

        image_tensor = Variable(image_tensor, requires_grad=False)
        image_tensor = image_tensor.unsqueeze(0)

        outputs = model.forward(image_tensor)
        print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)

        print("sleep {} seconds".format(sleep))
        time.sleep(int(sleep))


def load_model():
    model_path = '/output/model/model_{}.pth'.format(0)
    model = NetResnet50(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    return model


# def load_image(image:
#     # image = Image.open(image_name)
#     image = image.convert('RGB')
#     loader = transforms.Compose([transforms.ToTensor()])
#     image = loader(image).float()
#     return image / 255


if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sleep', required=False,
                        help='sleep seconds in the loop', default=3)
    args = parser.parse_args()
    main(args.sleep)
