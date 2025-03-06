import torch 
import random
from training.utils.loss_utils import compute_adjacency_matrix, compute_graph_laplacian, compute_gaussian_kernel_matrix

def sample_Ws(n, num_samples, device):
    Ws = []
    for _ in range(num_samples):
        W = torch.zeros(n, n, device=device)
        for i in range(n):
            choices = list(range(i)) + list(range(i+1, n))
            j = random.choice(choices)
            W[i, j] = 1.0
        Ws.append(W)
    return torch.stack(Ws) 

def calculate_sampled_regularization_term(z, Ws, sigma=0.75):
    num_samples, n, _ = Ws.shape
    kernel_matrix = torch.cdist(z, z, p=2)**2
    kernel_matrix = kernel_matrix.unsqueeze(0).expand(num_samples, n, n)
    c = -1./(2*sigma**2)
    scores = torch.sum(kernel_matrix * Ws, dim=(1,2))
    scores = c * scores
    return torch.sum(torch.exp(scores))

def calculate_regularization_term(z, pi, sigma=0.1):
    kernel_matrix = compute_gaussian_kernel_matrix(z, sigma=sigma)
    regularization_term = torch.sum(kernel_matrix * pi)
    return regularization_term / z.shape[0]

def lg_min_loss(out_1, out_2, graph_laplacian, num_samples=25, sigma=0.75):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.stack((out_1, out_2), dim=1)
    z = z.view(-1, out_1.size(1))

    if z.size(0) != graph_laplacian.size(0):
        graph_laplacian = compute_graph_laplacian(
            compute_adjacency_matrix(out_1.size(0), device), device)
        print("Shape of temporarily updated graph laplacian: ", graph_laplacian.shape)

    trace_term = torch.trace(z.T @ graph_laplacian @ z)
    W_samples = sample_Ws(z.shape[0], num_samples, device)
    regularization_term = calculate_sampled_regularization_term(z, W_samples, sigma)
    loss = trace_term + torch.log(regularization_term + 1e-08) #Preventing potential log(0)
    return loss, trace_term, regularization_term, torch.log(regularization_term)

def experimental_trace(out_1, out_2, graph_laplacian):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.stack((out_1, out_2), dim=1)
    z = z.view(-1, out_1.size(1))

    if z.size(0) != graph_laplacian.size(0):
        graph_laplacian = compute_graph_laplacian(
            compute_adjacency_matrix(out_1.size(0), device), device)
        print("Shape of temporarily updated graph laplacian: ", graph_laplacian.shape)

    return torch.trace(z.T @ graph_laplacian @ z)