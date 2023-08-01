import torch_optimizer
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# local files
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config
from src.utils import *
from src.loss import ClassificationFocalLoss
from src.trainer import Trainer


if __name__ == '__main__':
    if not os.path.exists(config.weight_path):
        os.mkdir(config.weight_path)
    
    seed_torch(config.seed)
    logger = setup_logger(config.log_file_name)
    train_df = pd.read_csv(f"train.csv")
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed).split(np.arange(train_df.shape[0]), train_df.label.values)

    for train_index, val_index in folds:
        _train = train_df.loc[train_index,:].reset_index(drop=True)
        _valid = train_df.loc[val_index,:].reset_index(drop=True)

        model = get_model(config.model_name, pretrained=True, num_classes=config.num_classes)
        dataloaders_dict = get_dataloaders_dict(_train, _valid)
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=config.lr)
        criterion = ClassificationFocalLoss(n_classes=config.num_classes)

        trainer = Trainer(model, dataloaders_dict, optimizer, criterion, logger=logger)
        trainer.fit()