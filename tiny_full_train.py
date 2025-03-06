import torch
from argparse import ArgumentParser

from training.tiny_pretraining import tiny_pretraining
from training.tiny_linear_evaluation import tiny_linear_evaluation
from training.tiny_kfold_linear_evaluation import tiny_kfold_linear_evaluation
from embeddings import calculate_chunk_embedding
from training.utils.utils import log_msg

if __name__ == "__main__":

    parser = ArgumentParser()

    #data 
    parser.add_argument("--dataset", type=str, default="mnist", help="mnist, cifar10")
    parser.add_argument("--data_dir", type=str, default="./dataset/mnist_subset/", help="path to download data")
    parser.add_argument("--train_subset_name", type=str, default="mnist_train_subset_1024_per_class_aug_200_chunk_", help="name of train subset") 
    parser.add_argument("--val_subset_name", type=str, default="mnist_val_subset_1024_per_class_aug_200.pt", help="name of val subset") 

    #model
    parser.add_argument("--arch", default="custom architecture", type=str, help="convnet architecture")
    parser.add_argument("--feat_dim", default=16, type=int, help="feature dimension used as output of the projection head")
    parser.add_argument("--hidden_mlp", default=32, type=int, help="hidden layer dimension in projection head")

    # optimization 
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
    parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
    parser.add_argument("--use_lr_scheduler", default=False, help="use lr scheduler")
    parser.add_argument("--loss_type", default="spectral_contrastive", type=str, help="nt_xent, origin, sum, product, spectral_contrastive, spectral, rq_min or lg_min")
    parser.add_argument("--penalty_constrained", default=False, action="store_true")
    parser.add_argument("--constrained_rqmin", default=False, action="store_true")
    parser.add_argument("--orthogonal_constrained", default=True, action="store_false")
    parser.add_argument("--centering_constrained", default=True, action="store_false")
    parser.add_argument("--num_samples_reg_term", type=int, default=25, help="Only used when loss is lg_min")
    parser.add_argument("--sigma", type=float, default=0.75, help="Only used when loss is lg_min")

    # training
    parser.add_argument("--num_samples", default=1024, type=int, help="number of samples")
    parser.add_argument("--max_epochs", default=1, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=1, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="batch size per gpu")

    # system 
    parser.add_argument("--fast_dev_run", default=0, type=int)
    parser.add_argument("--accelerator", default="gpu", help="gpu or cpu")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
    parser.add_argument("--fp32", default=True, action="store_true") 
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--chunked_data", default=True) 
    parser.add_argument("--pre_augmented", default=True) 

    # linear eval 
    parser.add_argument("--in_features", type=int, default=32) 
    parser.add_argument("--num_epochs", default=25, type=int, help="number of epochs in linear evaluation")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--nesterov", type=bool, default=False)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--extended_metrics", default=True, action='store_false')
    parser.add_argument("--train_majority", default=False, action='store_true')
    parser.add_argument("--k_folds", default=5, type=int, help="number k-folds")
    parser.add_argument("--num_classes", default=10, type=int, help="number of classes in dataset")


    args = parser.parse_args()

    log_msg(f"Tiny pretraining {args.arch} on {args.dataset} starting ...")
    run_name = tiny_pretraining({}, args)

    args.ckpt_path = f"results/pretraining/{run_name}/last.ckpt"
    args.run_name = run_name
    args.data_dir = "./dataset"
    args.batch_size = args.eval_batch_size

    print("\n\n\n")
    log_msg(f"Tiny linear evaluation {args.arch} with checkpoint {args.ckpt_path} on {args.dataset} starting ...")
    if args.k_folds > 1:
        print(f"Linear evaluation performed using {args.k_folds} folds\n")
        tiny_kfold_linear_evaluation({}, args)
    else: 
        tiny_linear_evaluation({}, args)

    num_augmentations = 200 
    data_path = f"./dataset/mnist_subset/chunks/mnist_train_subset_{args.num_samples}_per_class_aug_{num_augmentations}"
    storage_path = "./results/embeddings/" + run_name + "/chunks"
    print("\n\n\n")
    log_msg(f"Calculation of embeddings using backbone of checkpoint {args.ckpt_path} starting ...")    
    calculate_chunk_embedding(
        num_augmentations=num_augmentations,
        num_samples_per_class=args.num_samples,
        ckpt_path=args.ckpt_path,
        data_path=data_path,
        storage_path=storage_path,
        num_chunks=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n\n\n")
    log_msg("Completed full run through: pretraining, linear evaluation and embedding calculation")
    print("\n\n\n")