import wandb

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import MNISTDataModule
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor

from training.models.tiny_mnist_backbone import TinyMNISTBackbone
from training.utils.model_checkpoint import ModelCheckpoint
from training.utils.utils import log_msg
from training.utils.finetuner import SSLFineTuner
from training.utils.ckpt_loading import update_ckpt_dict


def tiny_linear_evaluation(config, args):
    args.__dict__.update(config)
    seed_everything(1234)
    
    if args.fast_dev_run == 0:
        name = args.run_name if args.run_name else None
        wandb_logger = WandbLogger(project='SSL', tags=['evaluation', 'tiny'], name=name)
    log_msg(args)

    if args.dataset == "mnist":
        dm = MNISTDataModule(data_dir=args.data_dir, normalize=False, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    else:
        raise NotImplementedError("other datasets have not been implemented till now")
    
    dm.train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dm.val_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dm.test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    backbone = TinyMNISTBackbone()
    backbone.load_state_dict(update_ckpt_dict(args.ckpt_path))
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    tuner = SSLFineTuner( 
        backbone,
        in_features=args.in_features,
        num_classes=dm.num_classes,
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
    )

    if args.fast_dev_run == 0:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/eval/{wandb.run.name}/")

    trainer_args = {
        "devices": args.gpus,
        "precision": 32 if args.fp32 else 16,
        "max_epochs": args.num_epochs,
        "accelerator": "gpu",
        "sync_batchnorm": True if args.gpus > 1 else False,
        "fast_dev_run": args.fast_dev_run,
    }

    if args.fast_dev_run == 0:
        trainer_args["logger"] = wandb_logger
        trainer_args["callbacks"] = [model_checkpoint, lr_monitor]
    
    trainer = Trainer(**trainer_args)

    trainer.fit(tuner, dm)
    trainer.test(datamodule=dm)

    if args.fast_dev_run == 0:
        if args.extended_metrics:
            cm = tuner.test_confusion_matrix(wandb.run.name)
            wandb.log({"confusion_matrix":cm})
        wandb.finish()