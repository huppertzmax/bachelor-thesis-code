import torch
from training.utils.loss_utils import compute_adjacency_matrix, compute_heat_kernel 

def rq_min_loss(out_1, out_2, heat_kernel, penalty_constrained):
    z = torch.stack((out_1, out_2), dim=1)
    z = z.view(-1, out_1.size(1))

    if z.size(0) != heat_kernel.size(0):
        heat_kernel = compute_heat_kernel(compute_adjacency_matrix(out_1.size(0), 'cuda' if torch.cuda.is_available() else 'cpu'), 1., 'cuda' if torch.cuda.is_available() else 'cpu')
        print("Shape of temporarily updated head kernel: ", heat_kernel.shape)

    trace_term = torch.trace(z.T @ heat_kernel @ z)
    orthogonality_term = torch.norm(z.T @ z - torch.eye(z.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu'), p='fro')
    centering_term = torch.norm(z.mean(dim=0), p=2)
    loss = -trace_term + orthogonality_term + centering_term if penalty_constrained else -trace_term
    return loss, trace_term, orthogonality_term, centering_term
