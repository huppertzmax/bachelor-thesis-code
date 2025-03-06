import os
import torch
import json
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = ArgumentParser()

parser.add_argument("--matrix1_name", type=str)
parser.add_argument("--matrix2_name", type=str)
parser.add_argument("--matrix1_path", type=str)
parser.add_argument("--matrix1_transformation_path", type=str)
parser.add_argument("--matrix2_path", type=str)
parser.add_argument("--path_suffix", type=str)
parser.add_argument("--matrix1_transformed", type=int, default=1)
parser.add_argument("--bootstrapping_iterations", type=int, default=0)

args = parser.parse_args()
print(args)

np.set_printoptions(precision=14)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/distances", f"{timestamp}_measures_{args.path_suffix}")
os.makedirs(folder_name, exist_ok=True)

matrix1_name = args.matrix1_name
matrix2_name = args.matrix2_name
matrix1_path=  args.matrix1_path
matrix2_path = args.matrix2_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device} as device')

matrix1 = np.load(matrix1_path)

if args.matrix1_transformed: 
    T = np.load(args.matrix1_transformation_path)
    matrix1 = matrix1 @ T

matrix1 = (torch.from_numpy(matrix1)).to(device)
matrix2 = np.load(matrix2_path)
matrix2 = (torch.from_numpy(matrix2)).to(device)

def visualize(mean_distances, highest_distances, lowest_distances,
                         title, output_path="batch_distances.png"):
    assert len(mean_distances) == len(highest_distances) == len(lowest_distances), \
        "All input lists must have the same length."

    batches = list(range(1, len(mean_distances) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(batches, mean_distances, label="Mean", marker="o")
    plt.plot(batches, highest_distances, label="Highest", marker="^")
    plt.plot(batches, lowest_distances, label="Lowest", marker="v")

    plt.xlabel("Batch Number")
    plt.ylabel("Distance")
    plt.title(title)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(output_path)
    plt.close()
    print(f"Stored visualization under {output_path}")

def append_infos(distances, mean_list, min_list, max_list):
    mean = torch.mean(distances)
    min = torch.min(distances, dim=0)
    max = torch.max(distances, dim=0)

    mean_list.append(mean.item())
    min_list.append(min.values.item())
    max_list.append(max.values.item())
    return mean.item()
    
cos = torch.nn.CosineSimilarity(dim=1)
euc = torch.nn.PairwiseDistance(p=2)
def normed_euc(row_m1, row_m2):
    normed_row_m1 = torch.nn.functional.normalize(row_m1, dim=1)
    normed_row_m2 = torch.nn.functional.normalize(row_m2, dim=1)
    return euc(normed_row_m1, normed_row_m2)

def compute_distances(matrix1, matrix2, visualization=True):
    cos_sim = []
    cos_min_sim = []
    cos_max_sim = []
    cos_dist = []
    cos_min_dist = []
    cos_max_dist = []
    euc_dist = []
    euc_min_dist = []
    euc_max_dist = []
    normed_euc_dist = []
    normed_euc_min_dist = []
    normed_euc_max_dist = []
    batch_size = 1024

    print(f"Calculating the distance measures of matrix with shape: {matrix1.size(0)} in {matrix1.size(0)//batch_size} batches of size {batch_size}")
    progress_bar = tqdm(range(matrix1.size(0)//batch_size), leave=False)
    for row in progress_bar:
        row_m1 = matrix1[row*batch_size:(row+1)*batch_size, :]
        row_m2 = matrix2[row*batch_size:(row+1)*batch_size, :]
        
        c_sim = cos(row_m1, row_m2)
        c_dist = torch.sub(torch.ones_like(c_sim), c_sim)
        e_dist = euc(row_m1, row_m2)
        normed_e_dist = normed_euc(row_m1, row_m2) 

        append_infos(c_sim, cos_sim, cos_min_sim, cos_max_sim)
        c_dist_mean = append_infos(c_dist, cos_dist, cos_min_dist, cos_max_dist)
        e_dist_mean = append_infos(e_dist, euc_dist, euc_min_dist, euc_max_dist)
        append_infos(normed_e_dist, normed_euc_dist, normed_euc_min_dist, normed_euc_max_dist)
        progress_bar.set_postfix(euc=e_dist_mean, cos=c_dist_mean)
    
    if visualization:
        visualize(cos_sim, cos_max_sim, cos_min_sim,
                        "Cosine similarity", folder_name + f"/cos_sim.png")
        visualize(cos_dist, cos_max_dist, cos_min_dist,
                        "Cosine distances", folder_name + f"/cos_dist.png")

        visualize(euc_dist, euc_max_dist, euc_min_dist,
                        "Euclidean distances", folder_name + f"/euc.png")
        
        visualize(normed_euc_dist, normed_euc_max_dist, normed_euc_min_dist,
                        "Normed euclidean distances", folder_name + f"/normed_euc.png")
    
    cos_sim_avg = sum(cos_sim) / len(cos_sim)
    cos_avg = sum(cos_dist) / len(cos_dist)
    euc_avg = sum(euc_dist) / len(euc_dist)
    normed_euc_avg = sum(normed_euc_dist) / len(normed_euc_dist)

    return cos_sim_avg, cos_avg, euc_avg, normed_euc_avg

cos_sim, cos_dist, euc_dist, normed_euc_dist = compute_distances(matrix1, matrix2)

if args.bootstrapping_iterations > 0: 
    cos_sim_values = []
    normed_euc_dist_values = []

    for i in range(args.bootstrapping_iterations):
        matrix2 = np.load(matrix2_path)
        np.random.shuffle(matrix2)
        matrix2 = (torch.from_numpy(matrix2)).to(device)

        cos_sim_iteration, _, _, normed_euc_dist_iteration = compute_distances(matrix1, matrix2, visualization=False)
        cos_sim_values.append(cos_sim_iteration)
        normed_euc_dist_values.append(normed_euc_dist_iteration)
        print(f"iteration: {i} with cos_sim: {cos_sim_iteration} and normed_euc_dist: {normed_euc_dist_iteration}")

    print(f"\nstandard cos_sim: {cos_sim}")
    print(f"bootstrapping cos_sim values length: {len(cos_sim_values)} and values: {cos_sim_values}\n")

    print(f"standard normed_euc_dist: {normed_euc_dist}")
    print(f"bootstrapping normed_euc_dist values length: {len(normed_euc_dist_values)} and values: {normed_euc_dist_values}\n")

    cos_sim_values = np.array(cos_sim_values)
    normed_euc_dist_values = np.array(normed_euc_dist_values)

    smaller_cos_sim_values = int(np.sum(cos_sim_values < cos_sim))
    larger_cos_sim_values = int(np.sum(cos_sim_values > cos_sim))
    smaller_normed_euc_dist_values = int(np.sum(normed_euc_dist_values < normed_euc_dist))
    larger_normed_euc_dist_values = int(np.sum(normed_euc_dist_values > normed_euc_dist))

results = {
    "timestamp": timestamp,
    "calculation_type": "batch-wise",
    "matrix1_name": matrix1_name,
    "matrix1_path": matrix1_path,
    "matrix2_name": matrix2_name,
    "matrix2_path": matrix2_path,
    "cosine_similarity": cos_sim,
    "cosine_distance": cos_dist,
    "euclidean_distance": euc_dist,
    "normed_euclidean_distance": normed_euc_dist,
    "transformed_matrix1": "True" if args.matrix1_transformed else "False", 
}

if args.matrix1_transformed: 
    results["matrix1_transformation_path"]= args.matrix1_transformation_path

if args.bootstrapping_iterations > 0: 
    results["bootstrapping_iterations"]=args.bootstrapping_iterations
    results["smaller_cos_sim_values"]=smaller_cos_sim_values
    results["larger_cos_sim_values"]=larger_cos_sim_values
    results["smaller_normed_euc_dist_values"]=smaller_normed_euc_dist_values
    results["larger_normed_euc_dist_values"]=larger_normed_euc_dist_values
    results["p-value_cos_sim"]=larger_cos_sim_values / args.bootstrapping_iterations
    results["p-value_normed_euc_dist"]=smaller_normed_euc_dist_values / args.bootstrapping_iterations

with open(folder_name + "/distances.json", 'w') as f:
    json.dump(results, f, indent=4)