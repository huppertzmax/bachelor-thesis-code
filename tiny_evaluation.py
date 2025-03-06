from argparse import ArgumentParser
from training.tiny_linear_evaluation import tiny_linear_evaluation
from training.tiny_kfold_linear_evaluation import tiny_kfold_linear_evaluation
from training.utils.utils import log_msg
import warnings

if __name__ == "__main__":
    parser = ArgumentParser()

    warnings.filterwarnings("ignore", message="The feature .* is currently marked under review.")
    # data
    parser.add_argument("--dataset", type=str, help="cifar10, mnist", default="mnist")
    parser.add_argument("--data_dir", type=str, help="path to dataset", default="./dataset")
    
    # model 
    parser.add_argument("--arch", default="custom architecture", type=str, help="convnet architecture")
    parser.add_argument("--ckpt_path", type=str, help="path to ckpt")
    parser.add_argument("--feat_dim", default=16, type=int, help="number of feat dim(256 for product loss, 128 for others)")
    parser.add_argument("--in_features", type=int, default=32, help="2048 for resnet50 and 512 for resnet18") 
    
    # optimization 
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--nesterov", type=bool, default=False)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--final_lr", type=float, default=0.0)
    
    # training
    parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu")
    parser.add_argument("--num_epochs", default=25, type=int, help="number of epochs")
    parser.add_argument("--num_samples", default=100, type=int, help="number of samples")
    parser.add_argument("--k_folds", default=5, type=int, help="number k-folds")
    parser.add_argument("--extended_metrics", default=True, action='store_false')
    parser.add_argument("--train_majority", default=False, action='store_true')
    parser.add_argument("--num_classes", default=10, type=int, help="number of classes in dataset")

    # system
    parser.add_argument("--gpus", default=1, type=int, help="number of GPUs")
    parser.add_argument("--accelerator", default="gpu", help="gpu or cpu")
    parser.add_argument("--fp32", action="store_true") 
    parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
    parser.add_argument("--fast_dev_run", default=0, type=int)

    args = parser.parse_args()


    log_msg(f"Tiny linear evaluation {args.arch} with checkpoint {args.ckpt_path} on {args.dataset} starting ...")
    if args.k_folds > 1:
        print(f"Linear evaluation performed using {args.k_folds} folds\n")
        tiny_kfold_linear_evaluation({}, args)
    else: 
        tiny_linear_evaluation({}, args)