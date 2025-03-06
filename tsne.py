import time
import os
import json
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from numpy.random import default_rng
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

def load_matrix(matrix_path, num_parts):
    matrix = np.load(matrix_path)
    part_size = matrix.shape[0] // num_parts
    print("Shape of loaded matrix: ", matrix.shape)
    print("Part size: ", part_size)
    return matrix, part_size

def generate_subset_matrix(matrix, part_size, num_parts, num_samples):
    indices = []
    rng = default_rng(123)
    subset = rng.choice(part_size, size=num_samples, replace=False)
    for i in range(num_parts):
        indices.extend([x+i*part_size for x in subset])

    labels = np.array([i for i in range(num_parts) for _ in range(num_samples)])
    matrix = matrix[indices]
    print("Shape of subset matrix", matrix.shape)
    print("Shape of subset labels", labels.shape)
    return matrix, labels

def run_tsne(matrix, iterations, metric, folder_name):
    print("\nTSNE starting ...")
    start_time = time.time()
    tsne = TSNE(n_iter=iterations, metric=metric)
    tsne_results = tsne.fit_transform(matrix)
    np.save(f"{folder_name}/tsne_results.npy", tsne_results)
    end_time = time.time()
    print(f"Runtime for tsne: {end_time - start_time:.6f} seconds")
    print(f"Saved results under: {folder_name}/tsne_results.py\n")
    return tsne_results

def visualize_tsne(tsne_results, labels, matrix_name, folder_name):
    fig = plt.figure( figsize=(10,8) )
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=15, alpha=0.5)
    plt.colorbar(scatter, label='Digit Label')
    plt.title(f"t-SNE - {matrix_name}")
    plt.show()
    plt.savefig(f"{folder_name}/tsne.png")

def store_config(timestamp, matrix_name, matrix_path, num_parts, num_samples, part_size, tsne_iter, tsne_metric, folder_name):
    results = {
        "timestamp": timestamp,
        "matrix_name": matrix_name,
        "matrix_path": matrix_path,
        "num_parts": num_parts,
        "num_samples": num_samples,
        "part_size": part_size,
        "tsne_iter": tsne_iter,
        "tsne_metric": tsne_metric,
    }

    with open(folder_name + "/tsne_results.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--matrix_name", type=str)
    parser.add_argument("--matrix_path", type=str)
    parser.add_argument("--tsne_iter", type=int, default=3000)
    parser.add_argument("--tsne_metric", type=str, default='euclidean')
    parser.add_argument("--num_parts", type=int, default=10) # Number of class (10 in case of MNIST)
    parser.add_argument("--num_samples", type=int, default=1024) # Number of rows to sample from each part
    parser.add_argument("--path_suffix", type=str)

    args = parser.parse_args()
    print(args)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.join("results/tsne", f"{timestamp}_{args.path_suffix}")
    os.makedirs(folder_name, exist_ok=True)

    matrix, part_size = load_matrix(args.matrix_path, args.num_parts)
    matrix, labels = generate_subset_matrix(matrix, part_size, args.num_parts, args.num_samples)
    tsne_results = run_tsne(matrix, args.tsne_iter, args.tsne_metric, folder_name)
    visualize_tsne(tsne_results, labels, args.matrix_name, folder_name)
    store_config(timestamp, args.matrix_name, args.matrix_path, args.num_parts, args.num_samples, part_size, args.tsne_iter, args.tsne_metric, folder_name)