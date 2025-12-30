"""Reward calculation for Leader-Follower navigation"""

import numpy as np
from typing import Dict, Any


class RewardCalculator:
    """Calculate rewards for Leader-Follower navigation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reward calculator
        
        Args:
            config: Configuration dictionary
        """
        self.progress_scale = config['reward']['progress_scale']
        self.collision_penalty = config['reward']['collision_penalty']
        self.success_reward = config['reward']['success_reward']
        self.boundary_penalty = config['reward']['boundary_penalty']
        
        self.guidance_smooth_weight = config['reward'].get('guidance_smooth_weight', 0.01)
        self.guidance_smooth_enabled = config['reward'].get('guidance_smooth_enabled', False)
        
        self.prev_centroid_dist = None
        self.prev_guidance_mean = None
    
    def reset(self):
        """Reset internal state"""
        self.prev_centroid_dist = None
        self.prev_guidance_mean = None
    
    def calculate_reward(
        self,
        positions: np.ndarray,
        goal_pos: np.ndarray,
        collision: bool,
        boundary_violation: bool,
        success: bool,
        guidance_vectors: np.ndarray = None,
        guidance_weights: np.ndarray = None
    ) -> float:
        """Calculate reward for current step
        
        Args:
            positions: Agent positions (N, 2)
            goal_pos: Goal position (2,)
            collision: Whether collision occurred
            boundary_violation: Whether boundary was violated
            success: Whether goal was reached
            guidance_vectors: Leader guidance vectors (K, 2), optional
            guidance_weights: Leader guidance weights (K,), optional
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Collision penalty
        if collision:
            reward += self.collision_penalty
            return reward
        
        # Boundary violation penalty
        if boundary_violation:
            reward += self.boundary_penalty
            return reward
        
        # Success reward
        if success:
            reward += self.success_reward
            return reward
        
        # Progress reward (based on centroid distance to goal)
        centroid = np.mean(positions, axis=0)
        current_dist = np.linalg.norm(centroid - goal_pos)
        
        if self.prev_centroid_dist is not None:
            progress = self.prev_centroid_dist - current_dist
            reward += self.progress_scale * progress
        
        self.prev_centroid_dist = current_dist
        
        # Guidance smoothness penalty (optional, for leader training)
        if self.guidance_smooth_enabled and guidance_vectors is not None and guidance_weights is not None:
            # Compute weighted mean guidance
            guidance_mean = np.sum(guidance_vectors * guidance_weights[:, None], axis=0)
            
            if self.prev_guidance_mean is not None:
                diff = guidance_mean - self.prev_guidance_mean
                smoothness_penalty = -self.guidance_smooth_weight * np.sum(diff ** 2)
                reward += smoothness_penalty
            
            self.prev_guidance_mean = guidance_mean.copy()
        
        return reward
    
    def calculate_done(
        self,
        positions: np.ndarray,
        goal_pos: np.ndarray,
        goal_radius: float,
        collision: bool,
        boundary_violation: bool,
        steps: int,
        max_steps: int
    ) -> bool:
        """Check if episode is done
        
        Args:
            positions: Agent positions (N, 2)
            goal_pos: Goal position (2,)
            goal_radius: Goal radius
            collision: Whether collision occurred
            boundary_violation: Whether boundary was violated
            steps: Current step count
            max_steps: Maximum steps
            
        Returns:
            True if episode is done
        """
        # Collision or boundary violation
        if collision or boundary_violation:
            return True
        
        # Success: all agents within goal radius
        centroid = np.mean(positions, axis=0)
        if np.linalg.norm(centroid - goal_pos) < goal_radius:
            return True
        
        # Max steps reached
        if steps >= max_steps:
            return True
        
        return False
    
    def check_success(
        self,
        positions: np.ndarray,
        goal_pos: np.ndarray,
        goal_radius: float
    ) -> bool:
        """Check if goal is reached
        
        Args:
            positions: Agent positions (N, 2)
            goal_pos: Goal position (2,)
            goal_radius: Goal radius
            
        Returns:
            True if goal is reached
        """
        centroid = np.mean(positions, axis=0)
        return np.linalg.norm(centroid - goal_pos) < goal_radius
