from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class Tr1toTr2(Dataset):
    def __init__(self, root_Tr2, root_Tr1, transform=None):
        self.root_Tr2 = root_Tr2
        self.root_Tr1 = root_Tr1
        self.transform = transform

        self.Tr2_images = os.listdir(root_Tr2)
        self.Tr1_images = os.listdir(root_Tr1)
        self.length_dataset = max(len(self.Tr2_images), len(self.Tr1_images)) # 1000, 1500
        self.Tr2_len = len(self.Tr2_images)
        self.Tr1_len = len(self.Tr1_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        Tr2_img = self.Tr2_images[index % self.Tr2_len]
        Tr1_img = self.Tr1_images[index % self.Tr1_len]

        Tr2_path = os.path.join(self.root_Tr2, Tr2_img)
        Tr1_path = os.path.join(self.root_Tr1, Tr1_img)

        Tr2_img = np.array(Image.open(Tr2_path).convert("RGB"))
        Tr1_img = np.array(Image.open(Tr1_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=Tr2_img, image0=Tr1_img)
            Tr2_img = augmentations["image"]
            Tr1_img = augmentations["image0"]

        return Tr2_img, Tr1_img
