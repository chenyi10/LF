"""Follower policy network"""

import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class FollowerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Follower
    
    Processes:
    - Local observations (position, velocity, rays)
    - Guidance vectors and weights from leader
    
    Note: For simplicity with SB3, we flatten multi-follower observations
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dims: list = [128, 128],
        use_layer_norm: bool = True
    ):
        """Initialize Follower features extractor
        
        Args:
            observation_space: Observation space
            hidden_dims: Hidden layer dimensions
            use_layer_norm: Whether to use layer normalization
        """
        features_dim = hidden_dims[-1]
        super().__init__(observation_space, features_dim=features_dim)
        
        self.use_layer_norm = use_layer_norm
        
        # Get input dimension from observation space
        # Expecting Dict with 'follower_obs' key
        follower_obs_shape = observation_space['follower_obs'].shape
        
        # Flatten the observation (n_followers, obs_dim) -> (n_followers * obs_dim)
        if len(follower_obs_shape) == 2:
            input_dim = follower_obs_shape[0] * follower_obs_shape[1]
        else:
            input_dim = follower_obs_shape[0]
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass
        
        Args:
            observations: Dictionary with 'follower_obs' key
            
        Returns:
            Features tensor
        """
        follower_obs = observations['follower_obs']
        
        # Flatten observations
        batch_size = follower_obs.shape[0]
        follower_obs_flat = follower_obs.reshape(batch_size, -1)
        
        features = self.mlp(follower_obs_flat)
        
        return features


class FollowerPolicy(ActorCriticPolicy):
    """
    Policy for Follower agents
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Follower policy"""
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Build MLP extractor (we use custom features extractor instead)"""
        # We use the features extractor directly
        self.mlp_extractor = nn.Identity()
