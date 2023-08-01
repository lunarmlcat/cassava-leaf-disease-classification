import torch
from tqdm.auto import tqdm as tqdm

from sklearn.metrics import accuracy_score, classification_report

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config as config


class Trainer:
    def __init__(self, model, dataloaders_dict, optimizer, criterion, scheduler=None, logger=None):
        # 初期値設定
        self.best_loss = 10**5
        self.best_score = 0.0
        self.counter = 0 # early_stop
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        self.n_fold = int()

        # setter
        self.model = model
        self.dataloaders_dict = dataloaders_dict
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        if logger is None:
            raise ValueError("logger is NoneType.")
        else:
            self.logger = logger

    # train
    def fit(self):
        freezed = True if config.freeze else False

        for epoch in range(config.epochs):
            if self.counter > 1:
                self.logger.debug("early stopping..")
                break

            self.logger.info(f"Epoch {epoch+1} / {config.epochs}")

            # 4epochから解除 config.freeze = Trueでのみ適用
            if freezed is True and epoch+1 > 3:
                self._unfreeze_model()
                freezed = False

            for phase in ["train", "val"]:
                if phase == "train":
                    epoch_loss = self._train(self.dataloaders_dict[phase])
                    self.logger.info(f'fold - {self.n_fold}:: phase: {phase}, loss: {epoch_loss:.4f}, -- learning_rate: {self.optimizer.param_groups[0]["lr"]}')
                    self._save_model()
                else:
                    epoch_loss, epoch_score = self._valid(self.dataloaders_dict[phase])
                    self.logger.info(f'fold - {self.n_fold}:: phase: {phase}, loss: {epoch_loss:.4f}, acc: {epoch_score:.4f}, -- learning_rate: {self.optimizer.param_groups[0]["lr"]}')
                    self._update_score(epoch_loss, epoch_score)
                    
            self._make_checkpoint()

        return self.best_score


    def set_n_fold(self, fold):
        self.n_fold = fold


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(config.device)


    def _train(self, train_loader):
        self.model.train()
        epoch_loss = 0.0

        # for idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            batch_size = inputs.shape[0]

            if config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()

                    if (idx + 1) % config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                loss.backward()

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if torch.isnan(loss):
                raise ValueError("contains the loss value of nan")

            epoch_loss += loss.item() * batch_size
            
        epoch_loss = epoch_loss / len(train_loader.dataset)

        return epoch_loss


    def _valid(self, valid_loader):
        self.model.eval()
        epoch_loss, epoch_score = 0.0, 0.0
        y_pred, y_true = [], []

        with torch.no_grad():
            # for _, (inputs, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            for _, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                batch_size = inputs.shape[0]

                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                if torch.isnan(loss):
                    raise ValueError("contains the loss value of nan")
                _, preds = torch.max(outputs.data, 1)

                epoch_loss += loss.item() * batch_size

                y_pred.extend(preds.to("cpu").numpy())
                y_true.extend(labels.to("cpu").numpy())

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = epoch_loss / len(valid_loader.dataset)
        epoch_score = accuracy_score(y_true, y_pred)
        self.logger.debug(f"\n{classification_report(y_true=y_true, y_pred=y_pred)}")

        return epoch_loss, epoch_score


    def _save_model(self):
        torch.save(self.model.state_dict(), f"{config.weight_path}/{config.model_name}.pth")


    def _update_score(self, epoch_loss, epoch_score):
        if self.best_score <= epoch_score:
            self.best_score = epoch_score
            self.best_loss = epoch_loss
            self.logger.debug(f"update best score: {self.best_score:.4f}")
            torch.save(self.model.state_dict(), f"{config.weight_path}/{config.model_name}.pth")
            self.counter = 0
        
        elif self.best_loss >= epoch_loss:
            self.best_loss = epoch_loss
            self.counter = 0

        else:
            self.logger.debug("There is no update of the best score")
            self.counter += 1


    def _unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True


    def _make_checkpoint(self):
        torch.save({
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
        }, f'{config.checkpoint_path}/{config.model_name}_checkpoint.pth.tar')

