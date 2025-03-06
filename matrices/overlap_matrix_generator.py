import time
import os 
import numpy as np
from argparse import ArgumentParser
from scipy.sparse import random, save_npz, csr_matrix

from similarity_graph_generator import calculate_diagonal_matrices, calculate_normalized_matrix, create_aug_group_sparse_matrix, create_pair_block_sparse_matrix

parser = ArgumentParser()
parser.add_argument("--sup_block_size", type=int, default=204800, help="Size of the super block")
parser.add_argument("--n", type=int, default=2048000, help="Value of n (sup_block_size * 10)")
parser.add_argument("--n_blocks", type=int, default=10240, help="Number of blocks")
parser.add_argument("--num_augs", type=int, default=200, help="Number of augmentations")
parser.add_argument("--density", type=float, default=0.00001, help="Density value")
parser.add_argument("--target_sum_random", type=float, default=0.5, help="Target sum for random")
parser.add_argument("--target_sum_ssl", type=float, default=0.5, help="Target sum for SSL")
parser.add_argument("--ssl_construction", type=str, default="", help="if aug_group or pair creates mixture of random supervised and ssl")

args = parser.parse_args()
print(f"Args:  {args}\n")
n_formatted = formatted = f"{args.n:,}".replace(",", "_")

def create_random_supervised_matrix(n, sup_block_size, density, target_sum_random):
    target_sum = target_sum_random / 10
    sparse_matrix = csr_matrix((n, n))

    for i in range(10):
        print(f"################# Start iteration {i} #################")
        start_time = time.time()
        rng = np.random.default_rng(seed=i) 
        sparse_matrix_block = random(sup_block_size, sup_block_size, density=density, format='csr', random_state=rng)
        end_time = time.time()
        print(f"Runtime for matrix generation: {end_time - start_time:.6f} seconds")
        print(f"Matrix sum values: {sparse_matrix_block.sum()}\n")

        num_nonzero = sparse_matrix_block.nnz
        print("Number of non-zero elements: ", num_nonzero)
        random_values = rng.random(num_nonzero)  
        random_values = (random_values / random_values.sum()) * target_sum  
        sparse_matrix_block.data = random_values
        print(f"Matrix sum values: {sparse_matrix_block.sum()}\n")

        rows, cols = sparse_matrix_block.nonzero()
        values = sparse_matrix_block.data

        rows = rows + i * sup_block_size
        cols = cols + i * sup_block_size

        sparse_matrix_block = csr_matrix((values, (rows, cols)), shape=(n, n))

        sparse_matrix = sparse_matrix + sparse_matrix_block
    print(f"Matrix sum values: {sparse_matrix.sum()}")
    print(f"Matrix non-zero values: {sparse_matrix.nnz}\n")
    return sparse_matrix

def saved_sparse_matrix(sparse_matrix, n_formatted, density, affix="supervised_random_block"):
    os.makedirs(f"./generated/sparse_matrix_{n_formatted}/{affix}", exist_ok=True)
    save_npz(f"./generated/sparse_matrix_{n_formatted}/{affix}/matrix_random_density_{density}.npz", sparse_matrix)
    print(f"Matrix saved under: generated/sparse_matrix_{n_formatted}/{affix}/matrix_random_density_{density}.npz")

def calculate_diag_and_normed(sparse_matrix, num_samples, n_formatted, density, affix="supervised_random"):
    diagonal_inv_sqrt_matrix = calculate_diagonal_matrices(sparse_matrix, affix, num_samples, n_formatted, density)
    calculate_normalized_matrix(sparse_matrix, diagonal_inv_sqrt_matrix, affix, num_samples, n_formatted, density)

def calculate_mix_ssl_and_random_supervised(n, n_blocks, num_augs, density, target_sum_random, target_sum_ssl, ssl_construction="aug_group"):
    if ssl_construction == "aug_group":
        print("Creating aug_group matrix")
        aug_block_value = target_sum_ssl / (n_blocks * num_augs * num_augs)
        ssl_matrix = create_aug_group_sparse_matrix(n_blocks, num_augs, aug_block_value)
    elif ssl_construction == "pair":
        print("Creating pair matrix")
        pair_value = target_sum_ssl / n
        ssl_matrix = create_pair_block_sparse_matrix(n//2, pair_value)
    print(f"SSL Matrix sum values: {ssl_matrix.sum()}")
    print(f"SSL Matrix non-zero values: {ssl_matrix.nnz}\n")

    random_matrix = create_random_supervised_matrix(n, n//10, density, target_sum_random)
    sparse_matrix = ssl_matrix + random_matrix
    print(f"Mix matrix sum values: {sparse_matrix.sum()}")
    print(f"Mix matrix non-zero values: {sparse_matrix.nnz}\n")
    return sparse_matrix

if args.ssl_construction == "aug_group":
    matrix = calculate_mix_ssl_and_random_supervised(n=args.n, n_blocks=args.n_blocks, num_augs=args.num_augs, density=args.density, target_sum_random=args.target_sum_random, target_sum_ssl=args.target_sum_ssl, ssl_construction=args.ssl_construction)
    affix="mix_aug_group_r-sup"
elif args.ssl_construction == "pair":
    matrix = calculate_mix_ssl_and_random_supervised(n=args.n, n_blocks=args.n_blocks, num_augs=args.num_augs, density=args.density, target_sum_random=args.target_sum_random, target_sum_ssl=args.target_sum_ssl, ssl_construction=args.ssl_construction)
    affix="mix_pair_r-sup"
else: 
    matrix = create_random_supervised_matrix(args.n, args.sup_block_size, args.density, args.target_sum_random)
    affix="r-sup"
saved_sparse_matrix(matrix, n_formatted, density=args.density, affix=f"{affix}_block")
calculate_diag_and_normed(matrix, args.n_blocks//10 ,n_formatted, args.density, affix=affix)