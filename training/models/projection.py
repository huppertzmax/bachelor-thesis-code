import numpy
from torch import nn
from torch.nn import functional as F

class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, norm_p=2., mu=1.):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.norm_p = norm_p
        self.mu = mu

        print(f"Projection head dimensions: input {input_dim}, output {output_dim}, hidden {hidden_dim}")
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1, p=self.norm_p) * numpy.sqrt(self.mu)
