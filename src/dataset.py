import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset

# local files
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config


class CassavaDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.transform = self.__get_augmentation(mode)
        self.df = df
        self.mode = mode

    def __getitem__(self, idx):
        uuid = self.df.loc[idx, "image_id"]
        image = cv2.imread(f"{config.data_path}/{uuid}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.__data_augmentation(self.transform, image)

        if self.mode == "test":
            return image
        else:
            label = self.df.loc[idx, "label"]
            return image, label

    def __len__(self):
        return self.df.shape[0]

    def __get_augmentation(self, mode="train"):
        if mode == "train":
            transform = [
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomBrightnessContrast(),
                albu.CLAHE(),
                albu.Resize(config.image_height, config.image_width),
                albu.Normalize(mean=config.mean, std=config.std),
                ToTensorV2(),
            ]
        else:
            transform = [
                albu.Resize(config.image_height, config.image_width),
                albu.Normalize(mean=config.mean, std=config.std),
                ToTensorV2(),
            ]
        return albu.Compose(transform)

    def __data_augmentation(self, transform, image):
        augmented = transform(image=image)
        return augmented['image']