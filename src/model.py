import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.preprocessing import LabelEncoder
from src.data_preprocessing import ASLDataset, get_transformations
from config import Config as CFG


class ASLClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.le = LabelEncoder()
        self.le.fit(CFG.label_names)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=25600, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=29)

        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.train_accuracy = Accuracy(task="multiclass", num_classes=29)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=29)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv4(x)))

        x = self.pool(x)
        # x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y.squeeze())
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # y_pred = torch.argmax(logits)
        loss = F.cross_entropy(logits, y.squeeze())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y.squeeze())
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        return optimizer


if __name__ == '__main__':
    train_transform, val_transform, test_transform = get_transformations()
    train_dataset = ASLDataset(directory='../data/split_data/train',
                               transform=train_transform)
    val_dataset = ASLDataset(directory="../data/split_data/val",
                             transform=val_transform)
    test_dataset = ASLDataset(directory="../data/split_data/test",
                              transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=15)

    logger = TensorBoardLogger("tb_logs", name="asl_classifier")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='../checkpoints/',
        filename='asl-avg-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{train_loss:.2f}-{train_acc:.2f}',
        save_top_k=2,
        mode='max',
    )

    trainer = Trainer(max_epochs=100,
                      logger=logger,
                      callbacks=checkpoint_callback,
                      deterministic=True,
                      log_every_n_steps=1)

    model = ASLClassifier()
    trainer.fit(model, train_loader, val_loader)
