from gymnasium import spaces
from typing import Dict, Optional, Tuple, Callable, List

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        
        self.attn_l = nn.Parameter(torch.Tensor(1, 1, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, 1, out_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, C = h.shape
        h_proj = self.fc(h)

        attn_l = (h_proj * self.attn_l).sum(dim=-1, keepdim=True) 
        attn_r = (h_proj * self.attn_r).sum(dim=-1, keepdim=True) 

        e = self.leaky_relu(attn_l + attn_r.transpose(1, 2))
        e = e.masked_fill(adj == 0, float("-1e9"))
        
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h_proj)
        return F.elu(h_prime)

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim harus habis dibagi num_heads"
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        
        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, 1, self.head_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, 1, self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, C = h.shape
        H = self.num_heads
        D = self.head_dim

        h_proj = self.fc(h).view(B, N, H, D).permute(0, 2, 1, 3)

        attn_l = (h_proj * self.attn_l).sum(dim=-1, keepdim=True)
        attn_r = (h_proj * self.attn_r).sum(dim=-1, keepdim=True)
        
        e = self.leaky_relu(attn_l + attn_r.transpose(-1, -2))

        if adj.dim() == 3:
            adj = adj.unsqueeze(1) 
        
        e = e.masked_fill(adj == 0, float("-1e18"))
        
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h_proj)
        
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        
        return F.elu(h_prime)

class GNNBinFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 64, gnn_embed_dim: int = 64):
        self.max_items = observation_space["items_state"].shape[0]
        self.item_feat_size = observation_space["items_state"].shape[1]
        
        total_dim = cnn_features_dim + (self.max_items * gnn_embed_dim) + self.max_items
        super().__init__(observation_space, features_dim=total_dim)
        
        self.cnn_features_dim = cnn_features_dim
        self.gnn_embed_dim = gnn_embed_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        with torch.no_grad():
            h, w, c = observation_space["height_map"].shape
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(dummy).shape[1]
        self.cnn_projection = nn.Linear(n_flatten, cnn_features_dim)

        self.item_projection = nn.Linear(self.item_feat_size, gnn_embed_dim)
        self.gnn_layer = GATLayer(gnn_embed_dim, gnn_embed_dim)
        self.norm = nn.LayerNorm(gnn_embed_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        h_map = observations["height_map"].permute(0, 3, 1, 2).contiguous()
        bin_feat = self.cnn_projection(self.cnn(h_map))

        items = observations["items_state"]
        item_mask = items[:, :, 0:1]
        
        x = self.item_projection(items)
        
        valid_nodes = item_mask.squeeze(-1).float()
        adj = torch.matmul(valid_nodes.unsqueeze(2), valid_nodes.unsqueeze(1))

        I = torch.eye(self.max_items, device=adj.device).unsqueeze(0)
        adj = ((adj + I) > 0).float()
        gnn_out = self.gnn_layer(x, adj) 

        x = self.norm(x + gnn_out)
        x = x * item_mask 

        flat_x = torch.cat([
            bin_feat, 
            x.view(x.size(0), -1), 
            item_mask.view(item_mask.size(0), -1)
        ], dim=1)
        
        return flat_x
    
class GNNMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        kwargs["features_extractor_class"] = GNNBinFeatureExtractor
        kwargs["features_extractor_kwargs"] = dict(cnn_features_dim=64, gnn_embed_dim=64)
        kwargs["net_arch"] = []

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self.max_items = observation_space["items_state"].shape[0]
        self.num_orientations = 6
        self.gnn_embed_dim = 64
        self.cnn_features_dim = 64
        
        self.action_item_proj = nn.Linear(self.gnn_embed_dim, 64)
        self.action_bin_proj = nn.Linear(self.cnn_features_dim, 64)
        self.action_out = nn.Linear(64, self.num_orientations)

        self.value_head = nn.Sequential(
            nn.Linear(self.gnn_embed_dim + self.cnn_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        batch_size = features.shape[0]
        
        bin_feat = features[:, :self.cnn_features_dim]
        
        item_feat_start = self.cnn_features_dim
        item_feat_end = item_feat_start + (self.max_items * self.gnn_embed_dim)
        
        item_feat_flat = features[:, item_feat_start : item_feat_end]
        item_feat = item_feat_flat.view(batch_size, self.max_items, self.gnn_embed_dim)
        
        item_mask = features[:, item_feat_end : item_feat_end + self.max_items].unsqueeze(-1)

        item_stream = self.action_item_proj(item_feat)
        bin_stream = self.action_bin_proj(bin_feat).unsqueeze(1)
        combined = torch.relu(item_stream + bin_stream)
        
        logits = self.action_out(combined) 
        
        logits = logits.masked_fill(item_mask.expand_as(logits) < 0.5, -1e7)
        latent_pi = logits.reshape(batch_size, -1)

        summed = (item_feat * item_mask).sum(dim=1)
        counts = item_mask.sum(dim=1).clamp(min=1.0)
        mean_items = summed / counts
        latent_vf = torch.cat([mean_items, bin_feat], dim=1)

        return latent_pi, latent_vf

    def forward(self, obs, deterministic=False, action_masks=None):
        latent_pi, latent_vf = self._get_latent(obs)
        values = self.value_head(latent_vf)
        dist = self._build_dist(latent_pi, action_masks)

        if deterministic:
            actions = torch.argmax(dist.logits, dim=-1)
        else:
            actions = dist.sample()

        log_prob = dist.log_prob(actions)
        
        return actions, values, log_prob

    def _build_dist(self, latent_pi, action_masks=None):
        if action_masks is not None:
            masks = torch.as_tensor(action_masks, device=latent_pi.device).bool()
            latent_pi = latent_pi.masked_fill(~masks, float("-1e9"))
        return Categorical(logits=latent_pi)

    def evaluate_actions(self, obs, actions, action_masks=None):
        latent_pi, latent_vf = self._get_latent(obs)
        values = self.value_head(latent_vf)
        dist = self._build_dist(latent_pi, action_masks)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        _, latent_vf = self._get_latent(obs)
        return self.value_head(latent_vf)
