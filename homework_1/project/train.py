import os
import torch
from torch import nn, optim
import torchmetrics
import torchvision.models as models
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import optuna
import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login()

class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        self.train_dataset = FashionMNIST(root="./data", train=True, download=True, transform=self.transform)
        self.val_dataset = FashionMNIST(root="./data", train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class LitModel(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    
    wandb_logger = WandbLogger(
        project="fashion-mnist",
        name=f"run-lr={lr:.6f}",
        log_model=True
    )

    model = LitModel(lr=lr)
    data_module = FashionMNISTDataModule(batch_size=32)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="best_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )

    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
