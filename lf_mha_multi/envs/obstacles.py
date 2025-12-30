"""Obstacle definitions and collision detection"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class Wall:
    """Line segment wall obstacle"""
    
    def __init__(self, start: np.ndarray, end: np.ndarray):
        """Initialize wall
        
        Args:
            start: Start point (2,)
            end: End point (2,)
        """
        self.start = np.array(start, dtype=np.float32)
        self.end = np.array(end, dtype=np.float32)
        self.vec = self.end - self.start
        self.length = np.linalg.norm(self.vec)
        self.dir = self.vec / (self.length + 1e-8)
        
        # Normal vector (perpendicular, pointing "outward" - left side of wall)
        self.normal = np.array([-self.dir[1], self.dir[0]], dtype=np.float32)
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to wall segment
        
        Args:
            point: Query point (2,)
            
        Returns:
            Minimum distance to wall
        """
        # Vector from start to point
        v = point - self.start
        
        # Project onto wall direction
        t = np.dot(v, self.dir)
        
        # Clamp to segment
        t = np.clip(t, 0, self.length)
        
        # Closest point on segment
        closest = self.start + t * self.dir
        
        # Distance
        return np.linalg.norm(point - closest)
    
    def ray_intersection(self, origin: np.ndarray, direction: np.ndarray, max_dist: float) -> Tuple[bool, float, np.ndarray]:
        """Calculate ray-wall intersection
        
        Args:
            origin: Ray origin (2,)
            direction: Ray direction (2,) - should be normalized
            max_dist: Maximum ray distance
            
        Returns:
            (hit, distance, hit_point) tuple
        """
        # Ray: P = origin + t * direction
        # Wall: P = start + s * vec, where 0 <= s <= 1
        
        # Solve: origin + t * direction = start + s * vec
        # origin - start = s * vec - t * direction
        
        # In matrix form: [vec, -direction] * [s, t]^T = origin - start
        
        diff = origin - self.start
        
        # Build matrix [vec, -direction]
        # Using Cramer's rule for 2x2 system
        denom = self.vec[0] * (-direction[1]) - self.vec[1] * (-direction[0])
        denom = self.vec[1] * direction[0] - self.vec[0] * direction[1]
        
        if abs(denom) < 1e-8:
            # Parallel or coincident
            return False, max_dist, origin + direction * max_dist
        
        # Solve for s and t
        s = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
        t = (diff[0] * self.vec[1] - diff[1] * self.vec[0]) / denom
        
        # Check if intersection is valid
        if 0 <= s <= 1 and 0 <= t <= max_dist:
            hit_point = origin + t * direction
            return True, t, hit_point
        
        return False, max_dist, origin + direction * max_dist


class ObstacleMap:
    """Collection of obstacles in the environment"""
    
    def __init__(self, obstacles_config: List[Dict]):
        """Initialize obstacle map
        
        Args:
            obstacles_config: List of obstacle configurations
        """
        self.walls = []
        
        for obs_cfg in obstacles_config:
            if obs_cfg['type'] == 'wall':
                wall = Wall(obs_cfg['start'], obs_cfg['end'])
                self.walls.append(wall)
    
    def check_collision_circle(self, pos: np.ndarray, radius: float) -> bool:
        """Check if a circle collides with any obstacle
        
        Args:
            pos: Circle center position (2,)
            radius: Circle radius
            
        Returns:
            True if collision detected
        """
        for wall in self.walls:
            if wall.distance_to_point(pos) < radius:
                return True
        return False
    
    def cast_ray(self, origin: np.ndarray, direction: np.ndarray, max_dist: float) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Cast a ray and find the nearest intersection
        
        Args:
            origin: Ray origin (2,)
            direction: Ray direction (2,) - should be normalized
            max_dist: Maximum ray distance
            
        Returns:
            (distance, hit_point, normal) tuple
            If no hit, distance=max_dist, hit_point=None, normal=None
        """
        min_dist = max_dist
        hit_point = None
        hit_normal = None
        
        for wall in self.walls:
            hit, dist, point = wall.ray_intersection(origin, direction, max_dist)
            if hit and dist < min_dist:
                min_dist = dist
                hit_point = point
                # Use wall normal as approximate normal
                hit_normal = wall.normal.copy()
        
        return min_dist, hit_point, hit_normal
    
    def get_obstacles_as_list(self) -> List[Dict]:
        """Get obstacles as list of dictionaries for rendering
        
        Returns:
            List of obstacle dictionaries
        """
        return [{'type': 'wall', 'start': wall.start, 'end': wall.end} for wall in self.walls]
