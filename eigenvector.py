import time
import numpy as np
from argparse import ArgumentParser
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh

def load_matrix(path):
    start_time = time.time()
    sparse_matrix = load_npz(path)
    end_time = time.time()
    print(sparse_matrix.nnz)
    print(f"Matrix loaded from: {path} in: {end_time - start_time:.6f} seconds\n")
    return sparse_matrix.tocsr()

def calculate_k_eigenvectors(matrix, k, storage_path, which):
    start_time = time.time()
    eigenvalues, eigenvectors = eigsh(matrix, k=k, which=which) 
    end_time = time.time()
    print(f"Runtime for {k} first eigenvectors calculation: {end_time - start_time:.6f} seconds\n")
    print(eigenvalues)
    print(f"Eigenvalues shape: {eigenvalues.shape}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")
    np.save(storage_path + f"eigenvalues_k_{k}_{which}.npy", eigenvalues)
    np.save(storage_path + f"eigenvectors_k_{k}_{which}.npy", eigenvectors)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_k", type=int, default=32)
    parser.add_argument("--base_path", type=str, default="matrices/generated/sparse_matrix_2_048_000/aug_group_block/") 
    parser.add_argument("--matrix_name", type=str, default="normalized_matrix_n_1024.npz")
    args = parser.parse_args()
    print(f"Args:  {args}\n")

    matrix = load_matrix(args.base_path + args.matrix_name)
    calculate_k_eigenvectors(matrix, args.num_k, args.base_path, 'LM')