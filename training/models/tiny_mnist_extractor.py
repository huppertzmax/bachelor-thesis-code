import torch
import pytorch_lightning as pl
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

from training.losses.loss import loss
from training.models.projection import Projection
from training.models.tiny_mnist_backbone import TinyMNISTBackbone
from training.utils.loss_utils import compute_adjacency_matrix, compute_heat_kernel, compute_graph_laplacian

class TinyMNISTExtractor(pl.LightningModule):
    def __init__(
        self,
        gpus: int= 1,
        num_samples: int = 1024,
        batch_size: int = 256,
        dataset: str ="mnist",
        num_nodes: int = 1,
        arch: str = "custom architecture",
        hidden_mlp: int = 32,
        feat_dim: int = 16,
        warmup_epochs: int = 10,
        max_epochs: int = 50,
        optimizer: str = "adam",
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        loss_type: str = "spectral_contrastive_loss",
        projection_mu: float=1.0,
        norm_p: float=2.0,
        use_lr_scheduler: bool=False,
        pre_augmented: bool=True,
        constrained_rqmin: bool=False, 
        orthogonal_constrained: bool=True,
        centering_constrained: bool=True,
        penalty_constrained: bool=False,
        num_samples_reg_term: int=100, 
        sigma: float=0.1,
        **kwargs
    ):
        super(TinyMNISTExtractor, self).__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.loss_type = loss_type
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim 
        self.max_epochs = max_epochs
        self.norm_p = norm_p
        self.projection_mu = projection_mu

        self.optim = optimizer
        self.use_lr_scheduler = use_lr_scheduler

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.pre_augmented = pre_augmented
        self.constrained_rqmin = constrained_rqmin
        self.orthogonal_constrained = orthogonal_constrained
        self.centering_constrained = centering_constrained
        self.penalty_constrained = penalty_constrained
        self.num_samples_reg_term = num_samples_reg_term
        self.sigma = sigma

        self.train_iters_per_epoch = self.num_samples // self.batch_size

        self.backbone = TinyMNISTBackbone()

        self.projection = Projection(input_dim=32, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim, norm_p=self.norm_p, mu=self.projection_mu)
        
        adj_matrix = compute_adjacency_matrix(self.batch_size, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.heat_kernel = compute_heat_kernel(adj_matrix, t=1., device='cuda' if torch.cuda.is_available() else 'cpu') 
        self.graph_laplacian = compute_graph_laplacian(adj_matrix, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def project_orthogonality(self, Z): 
        U, _, Vt = torch.linalg.svd(Z, full_matrices=False) 
        return U @ Vt 

    def project_zero_mean(self, Z): 
        mean = Z.mean(dim=0, keepdim=True) 
        return Z - mean 

    def project_constraints(self, Z): 
        if self.centering_constrained:
            print("Running centering projection")
            Z = self.project_zero_mean(Z) 
        if self.orthogonal_constrained:
            print("Running orthogonal projection")
            Z = self.project_orthogonality(Z) 
        return Z 

    def shared_step(self, batch):
        if self.pre_augmented:
            img, _ = batch 
            img1, img2 = torch.unbind(img, dim=1)
        else: 
            img1, img2, _ = batch

        h1 = self(img1)
        h2 = self(img2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        if self.constrained_rqmin:
            z1 = self.project_constraints(z1)
            z2 = self.project_constraints(z2) 

        return loss(out_1=z1, out_2=z2, loss_type=self.loss_type, heat_kernel=self.heat_kernel, penalty_constrained=self.penalty_constrained, graph_laplacian=self.graph_laplacian, num_samples_reg_term=self.num_samples_reg_term, sigma=self.sigma)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        if self.loss_type == "rq_min":
            loss, trace_term, orthogonality_term, centering_term = loss
            self.log("trace_term", trace_term)
            self.log("orthogonality_term", orthogonality_term)
            self.log("centering_term", centering_term)
        elif self.loss_type == "lg_min":
            loss, trace_term, regularization_term, log_reg_term = loss
            self.log("trace_term", trace_term)
            self.log("regularization_term", regularization_term)
            self.log("log_regularization_term", log_reg_term)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        if self.loss_type == "rq_min":
            loss, trace_term, orthogonality_term, centering_term = loss
            self.log("val_trace_term", trace_term)
            self.log("val_orthogonality_term", orthogonality_term)
            self.log("val_centering_term", centering_term)
        elif self.loss_type == "lg_min":
            loss, trace_term, regularization_term, log_reg_term = loss
            self.log("val_trace_term", trace_term)
            self.log("val_regularization_term", regularization_term)
            self.log("val_log_regularization_term", log_reg_term)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = self.parameters()
        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        
        if self.use_lr_scheduler:
            warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
            total_steps = self.train_iters_per_epoch * self.max_epochs
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        else:
            return optimizer