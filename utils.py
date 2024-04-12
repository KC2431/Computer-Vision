import torch
from torch.utils.data import Dataset
from PIL import Image
import natsort
import os
import skimage.io as io
import pandas as pd


def apply_transforms_to_tensor(sample, transform):
    image, label = sample
    image = transform(image)
    return (image, label)


class CustomDataSet(Dataset):
    '''
    Dataset class for NIPS2017 images.
    '''
    def __init__(self, main_dir, transform, labels):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.labels = labels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.labels['ImageId'][idx]) + '.png'
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return (tensor_image, self.labels['TrueLabel'][idx] - 1)

