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
        if len(follower_obs_shape) == 2:
            # Multiple followers: (n_followers, obs_dim)
            input_dim = follower_obs_shape[1]
        else:
            # Single follower: (obs_dim,)
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
        
        # Handle both single and multiple followers
        original_shape = follower_obs.shape
        if len(original_shape) == 3:
            # (batch, n_followers, obs_dim) -> process each follower
            batch_size, n_followers, obs_dim = original_shape
            follower_obs = follower_obs.reshape(batch_size * n_followers, obs_dim)
            features = self.mlp(follower_obs)
            # Reshape back
            features = features.reshape(batch_size, n_followers, -1)
        else:
            # (batch, obs_dim) or (batch * n_followers, obs_dim)
            features = self.mlp(follower_obs)
        
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
