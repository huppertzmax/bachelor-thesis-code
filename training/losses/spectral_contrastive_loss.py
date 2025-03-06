import torch
import numpy as np
import torch.nn.functional as F 


def spectral_contrastive_loss(out_1, out_2, mu=1.0):
    mask1 = (torch.norm(out_1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    mask2 = (torch.norm(out_2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    out_1 = mask1 * out_1 + (1-mask1) * F.normalize(out_1, dim=1) * np.sqrt(mu)
    out_2 = mask2 * out_2 + (1-mask2) * F.normalize(out_2, dim=1) * np.sqrt(mu)
    loss_part1 = -2 * torch.mean(out_1 * out_2) * out_1.shape[1]
    square_term = torch.matmul(out_1, out_2.T) ** 2
    loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                 out_1.shape[0] / (out_1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu