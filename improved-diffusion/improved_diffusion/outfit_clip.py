import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from transformers.models.clip.modeling_clip import CLIPOutput, clip_loss
from .rq_vae import AutoEncoder

@dataclass
class VQCLIPOutput(CLIPOutput):
    text_codes: torch.LongTensor = None
    image_codes: torch.LongTensor = None
    quantization_loss: torch.FloatTensor = None
    contrastive_loss: torch.FloatTensor = None
    perplexity: torch.FloatTensor = None


class OutfitCLIP(nn.Module):
    def __init__(self, args):
        super(OutfitCLIP, self).__init__()
        txt_compress_model = args.txt_compress_model

        self.outfit_embed_dim = 64
        self.text_embed_dim = 64
        self.projection_dim = 64

        self.outfit_model = lambda x: x
        self.text_model = AutoEncoder(args)

        self.outfit_projection = nn.Linear(self.outfit_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(args.logit_scale_init_value))

        self.post_init()

    def forward(
        self,
        input_embs,
        text_embs,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        
        outfit_embs = self.outfit_model(input_embs)
        prompt_embs = self.text_model(text_embs)

        outfit_embs = self.outfit_projection(outfit_embs)
        prompt_embs = self.text_projection(prompt_embs)

        # normalized features
        outfit_embs = outfit_embs / outfit_embs.norm(p=2, dim=-1, keepdim=True)
        prompt_embs = prompt_embs / prompt_embs.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(prompt_embs, outfit_embs.t()) * logit_scale
        logits_per_image = logits_per_text.t()

    def project_outfit_description(self, outfit_desc_embeds):
        return F.normalize(self.outfit_description_proj(outfit_desc_embeds), p=2, dim=-1)
    
    def project_outfit_embed(self, outfit_embeds):
        return F.normalize(self.outfit_embed_proj(outfit_embeds), p=2, dim=-1)