import os
import json
import time
from argparse import ArgumentParser
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# Implementation inspired by https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment

parser = ArgumentParser()

parser.add_argument("--matrix1_name", type=str)
parser.add_argument("--matrix2_name", type=str)
parser.add_argument("--matrix1_path", type=str)
parser.add_argument("--matrix2_path", type=str)
parser.add_argument("--path_suffix", type=str)
parser.add_argument("--gamma", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--kernel", type=str, default="Linear")


args = parser.parse_args()
print(args)
    
matrix1_name = args.matrix1_name
matrix2_name = args.matrix2_name
matrix1_path=  args.matrix1_path
matrix2_path = args.matrix2_path
matrix1 = np.load(matrix1_path) 
matrix2 = np.load(matrix2_path)

def centering(matrix):
    n = matrix.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, matrix), H)

def batch_kernel_HSIC(matrix1, matrix2, gamma, batch_size, kernel):
    n = matrix1.shape[0]
    hsic_total = 0
    
    for i in range(0, n, batch_size):
        X_batch = matrix1[i:i + batch_size]
        Y_batch = matrix2[i:i + batch_size]
        
        K_X_batch = rbf_kernel(X_batch, gamma=gamma) if kernel == "RBF" else linear_kernel(X_batch)
        K_Y_batch = rbf_kernel(Y_batch, gamma=gamma) if kernel == "RBF" else linear_kernel(Y_batch)
        
        K_X_batch_centered = centering(K_X_batch)
        K_Y_batch_centered = centering(K_Y_batch)
        
        hsic_total += np.sum(K_X_batch_centered * K_Y_batch_centered)
    
    return hsic_total

def kernel_CKA_mini_batches(matrix1, matrix2, gamma=None, batch_size=512, kernel="RBF"):
    hsic = batch_kernel_HSIC(matrix1, matrix2, gamma, batch_size, kernel)
    var1 = np.sqrt(batch_kernel_HSIC(matrix1, matrix1, gamma, batch_size, kernel))
    var2 = np.sqrt(batch_kernel_HSIC(matrix2, matrix2, gamma, batch_size, kernel))

    return hsic / (var1 * var2)

start_time = time.time()
cka_result = kernel_CKA_mini_batches(matrix1, matrix2, gamma=args.gamma, batch_size=args.batch_size, kernel=args.kernel)
end_time = time.time()
print(f"Runtime for CKA matrix: {end_time - start_time:.6f} seconds\n")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/cka", f"{timestamp}_cka-matrices_{args.path_suffix}")
os.makedirs(folder_name, exist_ok=True)

results = {
    "timestamp": timestamp,
    "matrix1_name": matrix1_name,
    "matrix1_path": matrix1_path,
    "matrix2_name": matrix2_name,
    "matrix2_path": matrix2_path,
    "cka": cka_result,
    "kernel": args.kernel,
    "calculation": "mini-batches", 
    "batch_size": args.batch_size, 
    "gamma": args.gamma
}

with open(folder_name + "/cka_results.json", 'w') as f:
    json.dump(results, f, indent=4)

print(f"Stored successfully under {folder_name}")