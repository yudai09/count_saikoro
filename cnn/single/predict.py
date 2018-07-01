import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms, utils
from model import NetResnet50
from PIL import Image

# cap = cv2.VideoCapture(0)

def main(image_name):
    model = load_model()
    model.eval()
    image = load_image(image_name)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    # print(image)
    outputs = model.forward(image)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)


def load_model():
    model_path = '/output/model/model_{}.pth'.format(0)
    model = NetResnet50()
    model.load_state_dict(torch.load(model_path))
    return model


def load_image(image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    loader = transforms.Compose([transforms.ToTensor()])
    image = loader(image).float()
    return image / 255

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, help="echo the string you use here")
    args = parser.parse_args()
    main(args.image_path)
