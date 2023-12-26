import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ



class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, in_dim=512, mlp_dim=1024, mlp_hidden_dim=512, mlp_layers=1, use_batch_norm=False, dropout=0.0):
        super(Encoder, self).__init__()
        self.in_dim = in_dim

        self.layers = nn.Sequential(
            # input is assumed to an already normalized clip embedding
            nn.Linear(in_dim, mlp_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(mlp_dim),
            *[
                Block(mlp_dim, mlp_hidden_dim)
                for _ in range(mlp_layers)
            ],
            nn.Linear(mlp_dim, in_dim, bias=False),
            # normalize before passing to VQ?
            # nn.GELU(),
            # nn.LayerNorm(args.clip_dim),
        )

    def forward(self, x):
        return self.layers(x)
    

class RQVAENew(nn.Module):
    def __init__(self, config):
        super(RQVAENew, self).__init__()
        torch.manual_seed(config.seed)
        self.debug = config.debug
        self.input_dim = config.item_embed_dim
        self.codebook_n_levels = config.rqvae_codebook_n_levels
        self.codebook_size = config.rqvae_codebook_size
        self.latent_dim = config.rqvae_latent_dim
        
        self.encoder = Encoder()

        self.residual_vq = ResidualVQ(
            dim=self.input_dim,
            num_quantizers=self.codebook_n_levels,
            codebook_dim=self.latent_dim,
            codebook_size=self.codebook_size,
            kmeans_init=True,
            kmeans_iters=1,
        )

    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def forward(self, x, return_all_codes=True):
        z = self.encode(x)
        x_hat, indices, commit_loss, all_codes = self.residual_vq(z, return_all_codes=return_all_codes)
        return x_hat, indices, commit_loss, all_codes


class RQVAEDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, out_dim, non_linear=False, use_batch_norm=False, dropout=0.0):
        super(RQVAEDecoder, self).__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim

        layers = []
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(z_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            if non_linear:
                # layers.append(nn.ReLU())
                layers.append(nn.PReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            z_dim = h_dim
        layers.append(nn.Linear(z_dim, out_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)
    
