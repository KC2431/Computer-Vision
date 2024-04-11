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


class CustomDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform):
        self.file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.file.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.file.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

            
        return (image, y_label)
