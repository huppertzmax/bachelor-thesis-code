import wandb
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.datasets import MNIST

from training.models.tiny_mnist_backbone import TinyMNISTBackbone
from training.utils.model_checkpoint import ModelCheckpoint
from training.utils.utils import log_msg
from training.utils.finetuner import SSLFineTuner
from training.utils.ckpt_loading import update_ckpt_dict
from dataset.tiny_kfold_mnist import KFoldDataModule


def tiny_kfold_linear_evaluation(config, args):
    args.__dict__.update(config)
    seed_everything(1234)
    if args.fast_dev_run == 0:
        name = args.run_name + f"_{args.k_folds}_folds" if args.run_name else None
        wandb_logger = WandbLogger(project='SSL', tags=['evaluation', 'tiny'], name=name) 
    log_msg(args)
    
    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        test_dataset = MNIST(root=args.data_dir, train=False, download=True, transform=transform)
        dataset = train_dataset + test_dataset
    else:
        raise NotImplementedError("other datasets have not been implemented till now")
    
    dm = KFoldDataModule(dataset=dataset, k=args.k_folds, batch_size=args.batch_size, val_split=0.1, train_majority=args.train_majority)

    test_accuracies = []
    test_f1_values = []
    test_losses = []

    for fold_index in range(dm.k):
        backbone = TinyMNISTBackbone()
        backbone.load_state_dict(update_ckpt_dict(args.ckpt_path))
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        dm.update_fold_index(fold_index=fold_index)
        dm.setup()

        tuner = SSLFineTuner( 
            backbone,
            in_features=args.in_features,
            num_classes=args.num_classes,
            epochs=args.num_epochs,
            hidden_dim=None,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            scheduler_type=args.scheduler_type,
            gamma=args.gamma,
            final_lr=args.final_lr,
            extended_metrics=args.extended_metrics,
            suffix=f"_fold_{fold_index}"
        )

        if args.fast_dev_run == 0:
            lr_monitor = LearningRateMonitor(logging_interval="step")
            if args.k_folds > 1: 
                model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/eval/{wandb.run.name}/fold_{fold_index}")
            else: 
                model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/eval/{wandb.run.name}")

        trainer_args = {
            "devices": args.gpus,
            "precision": 32 if args.fp32 else 16,
            "max_epochs": args.num_epochs,
            "accelerator": args.accelerator,
            "sync_batchnorm": True if args.gpus > 1 else False,
            "fast_dev_run": args.fast_dev_run,
        }

        if args.fast_dev_run == 0:
            trainer_args["logger"] = wandb_logger
            trainer_args["callbacks"] = [model_checkpoint, lr_monitor]
        
        trainer = Trainer(**trainer_args)

        trainer.fit(tuner, dm)
        test_results = trainer.test(datamodule=dm)
        test_accuracies.append(test_results[0]["test_acc"])

        if args.extended_metrics:
            test_f1_values.append(test_results[0]["test_f1"])
            test_losses.append(test_results[0]["test_loss"])
            if args.fast_dev_run == 0:
                cm = tuner.test_confusion_matrix(wandb.run.name, suffix=f"_fold_{fold_index}")
                wandb.log({"confusion_matrix":cm})
    
    if args.fast_dev_run == 0:
        wandb.log({"test_acc_avg":np.mean(test_accuracies)})
        if args.extended_metrics: 
            wandb.log({"test_f1_avg":np.mean(test_f1_values)})
            wandb.log({"test_loss_avg":np.mean(test_losses)})
        wandb.finish()