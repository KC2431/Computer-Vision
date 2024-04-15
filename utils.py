import torch
from torch.utils.data import Dataset
from PIL import Image
import natsort
import os
import skimage.io as io
import pandas as pd

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:  
            torch.nn.init.constant_(m.bias, 0)

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

