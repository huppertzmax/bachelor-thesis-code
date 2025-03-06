from training.losses.nt_xent_loss import nt_xent_loss
from training.losses.spectral_contrastive_loss import spectral_contrastive_loss
from training.losses.rq_min_loss import rq_min_loss
from training.losses.lg_min_loss import lg_min_loss, experimental_trace

def loss(out_1, out_2, temperature=0.1, loss_type="nt_xent", heat_kernel=None, penalty_constrained=False, graph_laplacian=None, num_samples_reg_term=100, sigma=0.1):
    if loss_type == "nt_xent":
        loss = nt_xent_loss(out_1=out_1, out_2=out_2, temperature=temperature)
    elif loss_type == "spectral_contrastive":
        loss = spectral_contrastive_loss(out_1=out_1, out_2=out_2)
    elif loss_type == "rq_min":
        return rq_min_loss(out_1=out_1, out_2=out_2, heat_kernel=heat_kernel, penalty_constrained=penalty_constrained)
    elif loss_type == "lg_min":
        return lg_min_loss(out_1=out_1, out_2=out_2, graph_laplacian=graph_laplacian, num_samples=num_samples_reg_term, sigma=sigma)
    elif loss_type == "experimental_trace":
        loss = experimental_trace(out_1=out_1, out_2=out_2, graph_laplacian=graph_laplacian)
    else:
        raise NotImplementedError

    return loss