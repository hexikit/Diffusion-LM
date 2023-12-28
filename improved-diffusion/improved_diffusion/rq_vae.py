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


class MergedCodebook(nn.Module):
    def __init__(self, codebooks, num_special_tokens, pad_token_id):
        super(MergedCodebook, self).__init__()
        self.n_levels, self.num_embeddings, self.embed_dim = codebooks.shape

        total_embeddings = (self.num_embeddings * self.n_levels) + num_special_tokens
        self.embedding = nn.Embedding(total_embeddings, self.embed_dim, padding_idx=pad_token_id)
        self.embedding.weight.data.uniform_(-1.0 / total_embeddings, 1.0 / total_embeddings)


        # Now, copy the pretrained weights to the new embedding layer
        start_idx = 0
        for level in range(self.n_levels):
            curr_codebook = codebooks[level, :, :]
            size = curr_codebook.shape[0]

            self.embedding.weight.data[start_idx:start_idx+size].copy_(curr_codebook)
            start_idx += size

    def freeze_embeddings(self):
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)
    

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super(AutoEncoder, self).__init__()
        # torch.manual_seed(config.seed)
        self.debug = config.debug
        self.input_dim = config.item_embed_dim
        self.latent_dim = config.rqvae_latent_dim

        self.mlp_dim = 1024
        self.mlp_hidden_dim = 512
        self.mlp_layers = 1
        self.use_batch_norm = False
        self.dropout = 0.0
        
        self.encoder = nn.Sequential(
            # input is assumed to an already normalized clip embedding
            nn.Linear(self.input_dim, self.mlp_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.mlp_dim),
            *[
                Block(self.mlp_dim, self.mlp_hidden_dim)
                for _ in range(self.mlp_layers)
            ],
            nn.Linear(self.mlp_dim, self.input_dim, bias=False),
            # normalize before passing to VQ?
            nn.GELU(),
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
        )

        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=False)

    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat
