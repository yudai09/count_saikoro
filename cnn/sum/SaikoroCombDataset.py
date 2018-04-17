import glob
import itertools
import numpy
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class SaikoroCombDataset(Dataset):
    """Saikoro combination dataset."""

    def __init__(self, image_dir='./images', train=True, train_split=0.8):
        self.train = train
        self.train_split = train_split
        self.image_names = glob.glob('{}/*.jpeg'.format(image_dir))
        self.loader = transforms.Compose([transforms.ToTensor()])

        perm = list(itertools.permutations(numpy.arange(len(self.image_names)),2))
        numpy.random.shuffle(perm)
        if train:
            self.perm = perm[:int(len(perm)*train_split)]
        else:
            self.perm = perm[int(len(perm)*train_split):]

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, idx):
        iter_1st, iter_2nd = self.perm[idx]

        image_name_1st = self.image_names[iter_1st]
        image_name_2nd = self.image_names[iter_2nd]

        eye_sum = int(self._eye(image_name_1st)) + int(self._eye(image_name_2nd))
        eye_sum = numpy.expand_dims(eye_sum, axis=0)

        eye_sum = torch.from_numpy(numpy.array(eye_sum))

        image_1st = self._load_image(image_name_1st) # / 255
        image_2nd = self._load_image(image_name_2nd) # / 255

        return image_1st, image_2nd, eye_sum

    def _eye(self, image_path):
        # retrieve eye from file path
        return image_path[:-len('.jpeg')].split('_')[-1:][0]

    def _load_image(self, image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = self.loader(image).float()
        return image
