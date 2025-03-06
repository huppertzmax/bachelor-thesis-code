from argparse import ArgumentParser
from training.tiny_pretraining import tiny_pretraining
from training.utils.utils import log_msg
import warnings

if __name__ == "__main__":

    parser = ArgumentParser()
    warnings.filterwarnings("ignore", message="The feature .* is currently marked under review.")

    #data 
    parser.add_argument("--dataset", type=str, default="mnist", help="mnist, cifar10")
    parser.add_argument("--data_dir", type=str, default="./dataset/mnist_subset/", help="path to download data")
    parser.add_argument("--train_subset_name", type=str, default="mnist_train_subset_1024_per_class_aug_200_chunk_", help="name of train subset")
    parser.add_argument("--val_subset_name", type=str, default="mnist_val_subset_1024_per_class_aug_200.pt", help="name of val subset")

    #model
    parser.add_argument("--arch", default="custom architecture", type=str, help="convnet architecture")
    parser.add_argument("--feat_dim", default=16, type=int, help="feature dimension")
    parser.add_argument("--hidden_mlp", default=32, type=int, help="hidden layer dimension in projection head")

    # optimization 
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
    parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
    parser.add_argument("--use_lr_scheduler", default=False, help="use lr scheduler")
    parser.add_argument("--penalty_constrained", default=False, action="store_true")
    parser.add_argument("--constrained_rqmin", default=False, action="store_true")
    parser.add_argument("--orthogonal_constrained", default=True, action="store_false")
    parser.add_argument("--centering_constrained", default=True, action="store_false")
    parser.add_argument("--num_samples_reg_term", type=int, default=25, help="Only used when loss is lg_min")
    parser.add_argument("--sigma", type=float, default=0.75, help="Only used when loss is lg_min")

    # loss
    parser.add_argument("--norm_p", default=2., type=float, help="norm p, -1 for inf")
    parser.add_argument("--loss_type", default="spectral_contrastive", type=str, help="nt_xent, origin, sum, product, spectral_contrastive, spectral, rq_min or kernel_infonce")
    parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
    parser.add_argument("--projection_mu", default=1., type=float, help="projection mu")
    
    # transformation
    parser.add_argument("--gaussian_blur", default=False, action="store_true", help="add gaussian blur")
    parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")

    # training
    parser.add_argument("--num_samples", default=1024, type=int, help="number of samples")
    parser.add_argument("--max_epochs", default=50, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=200, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")

    # system 
    parser.add_argument("--fast_dev_run", default=0, type=int)
    parser.add_argument("--accelerator", default="gpu", help="gpu or cpu")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
    parser.add_argument("--fp32", default=True, action="store_true") 
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--chunked_data", default=True) 
    parser.add_argument("--pre_augmented", default=True) 

    args = parser.parse_args()

    log_msg(f"Tiny pretraining {args.arch} on {args.dataset} starting ...")
    tiny_pretraining({}, args)