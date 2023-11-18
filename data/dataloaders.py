import random
import os

from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms import Compose

from PIL import Image

current_dir = os.getcwd()
cache_dir = os.path.join(current_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

class DIV2KDataset(Dataset):
    def __init__(self, split:str="train", downscale_type:str="bicubic", scale:int=2, transforms:Compose=None, seed:int=1989, cache_dir:str=cache_dir):
        self.dataset = load_dataset("eugenesiow/Div2k", "{}_x{}".format(downscale_type, scale), split=split, cache_dir=cache_dir)
        self.scale = scale
        self.transforms = transforms
        self.seed = seed
    
    def __getitem__(self, index:int):
        lr_path = self.dataset[index]["lr"]
        hr_path = self.dataset[index]["hr"]
        lr_image = Image.open(lr_path)
        hr_image = Image.open(hr_path)
        
        # Random crop
        while True:
            i, j, h, w = transforms.RandomCrop.get_params(hr_image, output_size=(768, 768))
            if i % self.scale == 0 and j % self.scale == 0:
                break
        hr_image = transforms.functional.crop(hr_image, i, j, h, w)
        lr_image = transforms.functional.crop(lr_image, i//self.scale, j//self.scale, h//self.scale, w//self.scale)

        if self.transforms:
            torch.manual_seed(seed=self.seed)
            lr_image = self.transforms(lr_image)
            torch.manual_seed(seed=self.seed)
            hr_image = self.transforms(hr_image)
        
        # To tensor
        lr_image = transforms.functional.to_tensor(lr_image)
        hr_image = transforms.functional.to_tensor(hr_image)

        return lr_image, hr_image
    
    def __len__(self):
        return len(self.dataset)
    
class DIV2KDataLoader(DataLoader):
    def __init__(self, split:str="train", downscale_type:str="bicubic", scale:int=4, transforms:Compose=None, seed:int=1989, cache_dir:str=cache_dir, batch_size:int=1, shuffle:bool=True, num_crops:int=1):
        sub_datasets = [DIV2KDataset(split=split, downscale_type=downscale_type, scale=scale, transforms=transforms, seed=seed+int(random.random()*10), cache_dir=cache_dir) for _ in range(num_crops)]
        self.dataset = ConcatDataset(sub_datasets)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle)
