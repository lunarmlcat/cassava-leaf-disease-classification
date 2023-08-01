import random
import os
import numpy as np
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm


# local files
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config
from src.dataset import CassavaDataset


#Pytorchで再現性を保つ
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)
    
    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    fh = FileHandler(log_folder)
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger


def get_model(model_name, pretrained=True, num_classes=1000):
    model = timm.create_model(model_name, pretrained=pretrained)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    model.to(config.device)
    
    return model


def get_dataloaders_dict(train_data, val_data):
    train_dataset = CassavaDataset(train_data, mode="train")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    val_dataset = CassavaDataset(val_data, mode="val")
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)

    return {"train": train_dataloader, "val": val_dataloader}