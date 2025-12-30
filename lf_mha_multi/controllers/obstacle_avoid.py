"""Obstacle avoidance controller using ray-based repulsion and velocity projection"""

import numpy as np
from typing import Tuple


class ObstacleAvoidanceController:
    """
    Implements obstacle avoidance using:
    1. Ray-based repulsion potential field
    2. Velocity projection to prevent penetration
    """
    
    def __init__(
        self,
        d_safe: float = 1.0,
        k_avoid: float = 0.15,
        a_avoid_max: float = 0.5,
        d_proj: float = 0.8,
        enabled: bool = True
    ):
        """Initialize obstacle avoidance controller
        
        Args:
            d_safe: Safety distance threshold for repulsion
            k_avoid: Repulsion strength coefficient
            a_avoid_max: Maximum avoidance action magnitude
            d_proj: Distance threshold for velocity projection
            enabled: Whether avoidance is enabled
        """
        self.d_safe = d_safe
        self.k_avoid = k_avoid
        self.a_avoid_max = a_avoid_max
        self.d_proj = d_proj
        self.enabled = enabled
    
    def compute_avoidance_action(
        self,
        distances: np.ndarray,
        directions: np.ndarray,
        normals: np.ndarray
    ) -> np.ndarray:
        """Compute avoidance action based on ray sensor readings
        
        Uses repulsive potential field:
        - If d >= d_safe: contribution = 0
        - If d < d_safe: magnitude = k_avoid * (1/d - 1/d_safe) / d^2
          direction = -ray_direction (away from obstacle)
        
        Args:
            distances: Ray distances (n_rays,)
            directions: Ray directions (n_rays, 2)
            normals: Normal vectors at hit points (n_rays, 2)
            
        Returns:
            Avoidance action (2,)
        """
        if not self.enabled:
            return np.zeros(2, dtype=np.float32)
        
        action = np.zeros(2, dtype=np.float32)
        
        for i in range(len(distances)):
            d = distances[i]
            
            # Only apply repulsion if within safety distance
            if d < self.d_safe:
                # Repulsion magnitude (increases as distance decreases)
                magnitude = self.k_avoid * (1.0 / d - 1.0 / self.d_safe) / (d * d)
                
                # Repulsion direction: away from obstacle (opposite to ray direction)
                direction = -directions[i]
                
                # Contribution to avoidance action
                contribution = magnitude * direction
                action += contribution
        
        # Clip to maximum magnitude
        action_norm = np.linalg.norm(action)
        if action_norm > self.a_avoid_max:
            action = action * (self.a_avoid_max / action_norm)
        
        return action
    
    def project_velocity(
        self,
        velocity: np.ndarray,
        distances: np.ndarray,
        directions: np.ndarray,
        normals: np.ndarray
    ) -> np.ndarray:
        """Project velocity to prevent penetration into obstacles
        
        If agent is close to obstacle (d < d_proj) and moving toward it,
        remove the component of velocity pointing toward the obstacle.
        
        Args:
            velocity: Current velocity (2,)
            distances: Ray distances (n_rays,)
            directions: Ray directions (n_rays, 2)
            normals: Normal vectors at hit points (n_rays, 2)
            
        Returns:
            Projected velocity (2,)
        """
        if not self.enabled:
            return velocity
        
        # Find closest obstacle
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        # If within projection distance and moving toward obstacle
        if min_dist < self.d_proj:
            # Use normal vector pointing away from obstacle
            # (We approximate this as -ray_direction)
            normal = -directions[min_idx]
            
            # Check if velocity points toward obstacle
            v_dot_n = np.dot(velocity, normal)
            
            # If velocity has component toward obstacle (v·n < 0), remove it
            if v_dot_n < 0:
                # Project velocity onto plane perpendicular to normal
                velocity = velocity - v_dot_n * normal
        
        return velocity
    
    def get_closest_obstacle_info(
        self,
        distances: np.ndarray,
        directions: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Get information about closest obstacle
        
        Args:
            distances: Ray distances (n_rays,)
            directions: Ray directions (n_rays, 2)
            
        Returns:
            (distance, direction) to closest obstacle
        """
        min_idx = np.argmin(distances)
        return distances[min_idx], directions[min_idx]
