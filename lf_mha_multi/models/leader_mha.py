"""Leader policy network with Multi-Head Attention and Multi-Hypothesis output"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class LeaderMHAFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Leader using Multi-Head Attention
    
    Processes:
    - Multi-view ray observations (tokenized)
    - Goal information
    - Group summary
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        n_heads: int = 4,
        d_model: int = 128,
        d_feedforward: int = 256,
        dropout: float = 0.1,
        ray_encoding_dim: int = 32,
        goal_encoding_dim: int = 32,
        summary_encoding_dim: int = 32
    ):
        """Initialize Leader MHA features extractor
        
        Args:
            observation_space: Observation space
            n_heads: Number of attention heads
            d_model: Model dimension
            d_feedforward: Feedforward dimension
            dropout: Dropout rate
            ray_encoding_dim: Ray encoding dimension
            goal_encoding_dim: Goal encoding dimension
            summary_encoding_dim: Summary encoding dimension
        """
        # Features dimension is d_model
        super().__init__(observation_space, features_dim=d_model)
        
        self.d_model = d_model
        
        # Get input dimensions from observation space
        # Expecting Dict with 'leader_obs' key
        leader_obs_shape = observation_space['leader_obs'].shape[0]
        
        # We'll structure the input as:
        # [n_rays distances, goal_x, goal_y, summary features...]
        # For tokenization, we split into: ray tokens, goal token, summary token
        
        # Encoders to project inputs to d_model
        self.ray_encoder = nn.Linear(1, ray_encoding_dim)  # Each ray is 1D (distance)
        self.ray_proj = nn.Linear(ray_encoding_dim, d_model)
        
        self.goal_encoder = nn.Linear(2, goal_encoding_dim)  # Goal is 2D (x, y)
        self.goal_proj = nn.Linear(goal_encoding_dim, d_model)
        
        # Summary includes: centroid (2), mean velocity (2), spread (1), etc.
        self.summary_encoder = nn.Linear(5, summary_encoding_dim)
        self.summary_proj = nn.Linear(summary_encoding_dim, d_model)
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Store n_rays for parsing
        self.n_rays = None
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass
        
        Args:
            observations: Dictionary with 'leader_obs' key
            
        Returns:
            Features tensor (batch_size, d_model)
        """
        leader_obs = observations['leader_obs']
        batch_size = leader_obs.shape[0]
        
        # Parse observation
        # Assuming structure: [n_rays distances, goal_x, goal_y, centroid_x, centroid_y, 
        #                      mean_vel_x, mean_vel_y, spread]
        n_rays = leader_obs.shape[1] - 7  # Subtract goal(2) + summary(5)
        
        ray_dists = leader_obs[:, :n_rays]  # (batch, n_rays)
        goal = leader_obs[:, n_rays:n_rays+2]  # (batch, 2)
        summary = leader_obs[:, n_rays+2:]  # (batch, 5)
        
        # Tokenize rays
        ray_tokens = []
        for i in range(n_rays):
            ray_feat = self.ray_encoder(ray_dists[:, i:i+1])  # (batch, ray_encoding_dim)
            ray_token = self.ray_proj(ray_feat)  # (batch, d_model)
            ray_tokens.append(ray_token)
        ray_tokens = torch.stack(ray_tokens, dim=1)  # (batch, n_rays, d_model)
        
        # Goal token
        goal_feat = self.goal_encoder(goal)  # (batch, goal_encoding_dim)
        goal_token = self.goal_proj(goal_feat).unsqueeze(1)  # (batch, 1, d_model)
        
        # Summary token
        summary_feat = self.summary_encoder(summary)  # (batch, summary_encoding_dim)
        summary_token = self.summary_proj(summary_feat).unsqueeze(1)  # (batch, 1, d_model)
        
        # Combine tokens
        tokens = torch.cat([ray_tokens, goal_token, summary_token], dim=1)  # (batch, n_tokens, d_model)
        
        # Multi-head attention
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        
        # Feedforward
        ffn_out = self.ffn(tokens)
        tokens = self.ln2(tokens + ffn_out)
        
        # Aggregate tokens (use mean pooling)
        features = tokens.mean(dim=1)  # (batch, d_model)
        
        return features


class LeaderMultiHypothesisHead(nn.Module):
    """
    Output head that produces K guidance hypotheses with confidence weights
    """
    
    def __init__(self, features_dim: int, n_hypotheses: int):
        """Initialize multi-hypothesis head
        
        Args:
            features_dim: Input features dimension
            n_hypotheses: Number of hypotheses (K)
        """
        super().__init__()
        self.n_hypotheses = n_hypotheses
        
        # Output K guidance directions (each 2D unit vector)
        # We output K * 2 values and normalize them
        self.guidance_net = nn.Linear(features_dim, n_hypotheses * 2)
        
        # Output K confidence logits (will apply softmax)
        self.confidence_net = nn.Linear(features_dim, n_hypotheses)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            features: Features tensor (batch_size, features_dim)
            
        Returns:
            (guidance_vectors, confidence_weights) tuple
            - guidance_vectors: (batch_size, n_hypotheses, 2) - unit vectors
            - confidence_weights: (batch_size, n_hypotheses) - softmax weights
        """
        batch_size = features.shape[0]
        
        # Guidance directions
        guidance_raw = self.guidance_net(features)  # (batch, K*2)
        guidance_raw = guidance_raw.view(batch_size, self.n_hypotheses, 2)  # (batch, K, 2)
        
        # Normalize to unit vectors
        guidance_vectors = F.normalize(guidance_raw, p=2, dim=-1)  # (batch, K, 2)
        
        # Confidence weights
        confidence_logits = self.confidence_net(features)  # (batch, K)
        confidence_weights = F.softmax(confidence_logits, dim=-1)  # (batch, K)
        
        return guidance_vectors, confidence_weights


class LeaderMHAPolicy(ActorCriticPolicy):
    """
    Custom policy for Leader with MHA and multi-hypothesis output
    """
    
    def __init__(self, *args, n_hypotheses: int = 3, **kwargs):
        """Initialize Leader policy
        
        Args:
            n_hypotheses: Number of guidance hypotheses
        """
        self.n_hypotheses = n_hypotheses
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Build MLP extractor (we use custom features extractor instead)"""
        # We use the features extractor directly, so this is a pass-through
        self.mlp_extractor = nn.Identity()
