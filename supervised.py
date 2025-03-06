import os
import torch
import wandb
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from argparse import ArgumentParser
from torchvision import transforms
from torchvision.datasets import MNIST
from torchsummary import summary
from torchmetrics import Accuracy, F1Score
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from training.utils.utils import log_msg
from training.models.tiny_mnist_backbone import TinyMNISTBackbone
from dataset.tiny_kfold_mnist import KFoldDataModule
from training.utils.model_checkpoint import ModelCheckpoint


class MNISTModel(LightningModule):
    def __init__(self, backbone, linear_head, extended_metrics, num_classes, num_epochs, suffix):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.extended_metrics = extended_metrics
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.suffix = suffix
        self.test_logits = []
        self.test_y = []

        self.train_acc = Accuracy(task="multiclass", num_classes=10, average="micro", top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=10, compute_on_step=False, average="micro", top_k=1)
        self.test_acc = Accuracy(task="multiclass", num_classes=10, compute_on_step=False, average="micro", top_k=1)
        if self.extended_metrics:
            self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro", top_k=1)
            self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, compute_on_step=False, average="macro", top_k=1)
            self.test_f1 = F1Score(task="multiclass", num_classes=self.num_classes, compute_on_step=False, average="macro", top_k=1)

        summary(backbone.cuda(), (1, 28, 28))

    def on_train_epoch_start(self):
        self.backbone.train()
        self.linear_head.train()

    def on_validation_start(self):
        self.backbone.eval()
        self.linear_head.eval()
    
    def on_test_start(self):
        self.backbone.eval()
        self.linear_head.eval()
    
    def step(self, batch):
        x, y = batch
        z = self.backbone(x)
        logits = self.linear_head(z)
        loss = nn.functional.cross_entropy(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        if self.extended_metrics:
            self.train_f1(logits, y)
            self.log(f"train_f1{self.suffix}", self.train_f1)

        logits = logits.softmax(-1)
        self.train_acc(logits, y)

        self.log(f"train_loss{self.suffix}", loss, prog_bar=True)
        self.log(f"train_acc{self.suffix}", self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        if self.extended_metrics:
            self.val_f1(logits, y)
            self.log(f"val_f1{self.suffix}", self.val_f1)
        
        logits = logits.softmax(-1)
        self.val_acc(logits, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"val_loss{self.suffix}", loss, prog_bar=True, sync_dist=True)
        self.log(f"val_acc{self.suffix}", self.val_acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        if self.extended_metrics:
            self.test_f1(logits, y)
            self.log("test_f1", self.test_f1)
        
        logits = logits.softmax(-1)
        self.test_acc(logits.softmax(-1), y)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc)

        self.test_logits.append(np.argmax(logits.cpu().numpy(), axis=-1).tolist())
        self.test_y.append(y.cpu().numpy().tolist())
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.1,
            nesterov=False,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.num_epochs, eta_min=0.0
        )

        return [optimizer], [scheduler]
    
    def test_confusion_matrix(self, run_name, suffix=""):
        self.test_y = list(itertools.chain.from_iterable(self.test_y))
        self.test_logits = list(itertools.chain.from_iterable(self.test_logits))
        cm = confusion_matrix(self.test_y, self.test_logits)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        plt.savefig(f"results/supervised/{run_name}/test_confusion_matrix{suffix}.png")
        cm  = wandb.plot.confusion_matrix(y_true=self.test_y, preds=self.test_logits, class_names=["0","1","2","3","4","5","6","7","8","9"])
        return cm


def supervised_learning(config, args):
    args.__dict__.update(config)
    seed_everything(1234)
    
    if args.fast_dev_run == 0:
        wandb_logger = WandbLogger(project='supervised')
    log_msg(args)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    dataset = train_dataset + test_dataset
    
    dm = KFoldDataModule(dataset=dataset, k=args.k_folds, batch_size=args.batch_size, val_split=0.1, train_majority=args.train_majority)

    test_accuracies = []
    test_f1_values = []
    test_losses = []

    for fold_index in range(dm.k):
        backbone = TinyMNISTBackbone()
        linear_head = nn.Linear(32, 10, bias=True)
        model = MNISTModel(backbone, linear_head, args.extended_metrics, 10, args.num_epochs, f"_fold_{fold_index}")

        dm.update_fold_index(fold_index=fold_index)
        dm.setup()

        trainer_args = {
            "devices": 1,
            "precision": 32,
            "max_epochs": args.num_epochs,
            "accelerator": "gpu",
            "sync_batchnorm": False,
            "fast_dev_run": args.fast_dev_run,
        }

        if args.fast_dev_run == 0:
            trainer_args["logger"] = wandb_logger
            lr_monitor = LearningRateMonitor(logging_interval="step")
            if args.k_folds > 1: 
                model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/supervised/{wandb.run.name}/fold_{fold_index}")
            else: 
                model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/supervised/{wandb.run.name}")
            trainer_args["callbacks"] = [model_checkpoint, lr_monitor]
        
        trainer = Trainer(**trainer_args)
        trainer.fit(model, dm)
        test_results = trainer.test(model, datamodule=dm)
        test_accuracies.append(test_results[0]["test_acc"])

        if args.extended_metrics:
            test_f1_values.append(test_results[0]["test_f1"])
            test_losses.append(test_results[0]["test_loss"])
            if args.fast_dev_run == 0:
                cm = model.test_confusion_matrix(wandb.run.name, suffix=f"_fold_{fold_index}")
                wandb.log({"confusion_matrix":cm})

    if args.fast_dev_run == 0:
        wandb.log({"test_acc_avg":np.mean(test_accuracies)})
        if args.extended_metrics: 
            wandb.log({"test_f1_avg":np.mean(test_f1_values)})
            wandb.log({"test_loss_avg":np.mean(test_losses)})
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", default=25, type=int, help="number of total epochs to run")
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size")
    parser.add_argument("--data_dir", type=str, help="path to dataset", default="./dataset")
    parser.add_argument("--extended_metrics", default=True, action='store_false')
    parser.add_argument("--train_majority", default=False, action='store_true')
    parser.add_argument("--k_folds", default=5, type=int, help="number k-folds")
    parser.add_argument("--fast_dev_run", default=0, type=int)
    args = parser.parse_args()
    supervised_learning({}, args)