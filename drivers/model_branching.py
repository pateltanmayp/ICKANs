import sys
sys.path.insert(0, "/home/simulation/thesis/ICKANs/")
import torch
import torch.nn as nn
from core import *
import drivers.config as c
from ickan import *


# Hypernetwork for material conditioning
class MaterialHyperNet(nn.Module):
    """
    Maps z to scaling parameters
    Does NOT touch convex structure directly
    """
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
            nn.Softplus()
        )

    def forward(self, z):
        return self.net(z)


# Branching Convex KAN Model
class BranchingConvexKAN(nn.Module):
    def __init__(self,
                 n_hidden,
                 grid_range,
                 z_dim=4,
                 use_kan=True,
                 seed=0):
        super().__init__()

        self.z_dim = z_dim

        # Elastic branch (convex, polyconvex energy)
        if use_kan:
            self.elastic_nn = KAN(width=n_hidden, grid=c.grid, k=c.spline_order, seed=seed, device='cuda', base_fun='zero', grid_eps = 1.0, 
                    grid_range_0=grid_range, sp_trainable=c.sp_trainable, sb_trainable=c.sb_trainable,symbolic_enabled=c.symbolic_enabled,
                    auto_save=False)
        else:
            self.elastic_nn = nn.Sequential(
                nn.Linear(3, n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, 1),
                nn.Softplus()
            )
    

        # Material conditioning for elastic branch
        self.elastic_scale = MaterialHyperNet(z_dim, out_dim=1)

        # Plastic activation (scalar gate)
        self.plastic_gate = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # [0,1]
        )

        # Branch weights (elastic vs plastic)
        self.branch_weights = nn.Sequential(
            nn.Linear(z_dim, 2),
            nn.Softplus()
        )

    # Invariant computation
    def compute_invariants(self, F):
        F = torch.clamp(F, min=-5.0, max=5.0)

        F11 = F[:, 0:1]
        F12 = F[:, 1:2]
        F21 = F[:, 2:3]
        F22 = F[:, 3:4]

        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = C12
        C22 = F12**2 + F22**2

        I1 = C11 + C22 + 1.0
        I2 = C11 + C22 - C12*C21 + C11*C22
        I3 = C11*C22 - C12*C21

        eps = 1e-6
        I3_safe = torch.clamp(I3, min=eps)

        J = torch.sqrt(I3_safe)
        K1 = I1 * torch.pow(I3_safe, -1/3) - 3.0
        K2 = torch.pow((I1 + I3_safe - 1.) * torch.pow(I3_safe, -2./3.), 3/2) \
            - 3.0 * torch.sqrt(torch.tensor(3.0, device=F.device))
        K3 = (J - 1)**2

        return torch.cat((K1, K2, K3), dim=1).float()

    def forward(self, F, z):
        """
        Input:
            F: (N, 4) deformation gradient
            z: (N, z_dim) material code
        Output:
            W: (N, 1) strain energy
        """

        K = self.compute_invariants(F)

        W_elastic = self.elastic_nn(K)

        elastic_scale = self.elastic_scale(z)

        W_elastic = elastic_scale * W_elastic

        plastic_factor = self.plastic_gate(z)
        W_plastic = plastic_factor * W_elastic.detach()

        weights = self.branch_weights(z)
        alpha = weights / torch.sum(weights, dim=1, keepdim=True)

        W = alpha[:, 0:1] * W_elastic + alpha[:, 1:2] * W_plastic

        return W


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def print_model_arch(model):
    print('\n\n', '-' * 80, '\n', model, '\n', '-' * 80, '\n\n')
