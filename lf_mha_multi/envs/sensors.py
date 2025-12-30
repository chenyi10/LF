"""Sensor modules for obstacle detection"""

import numpy as np
from typing import Tuple, List, Optional
from .obstacles import ObstacleMap


class RaySensor:
    """Ray-based distance sensor for obstacle detection"""
    
    def __init__(self, n_rays: int, max_range: float):
        """Initialize ray sensor
        
        Args:
            n_rays: Number of rays
            max_range: Maximum sensing range
        """
        self.n_rays = n_rays
        self.max_range = max_range
        
        # Pre-compute ray angles (evenly distributed around 2π)
        self.angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    
    def sense(
        self, 
        pos: np.ndarray, 
        obstacle_map: ObstacleMap,
        world_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform ray sensing from a position
        
        Args:
            pos: Agent position (2,)
            obstacle_map: Obstacle map
            world_size: Size of the world (for boundary detection)
            
        Returns:
            (distances, directions, normals) tuple
            - distances: (n_rays,) distance to nearest obstacle per ray
            - directions: (n_rays, 2) ray directions
            - normals: (n_rays, 2) normal vectors at hit points (or -direction if no hit)
        """
        distances = np.zeros(self.n_rays, dtype=np.float32)
        directions = np.zeros((self.n_rays, 2), dtype=np.float32)
        normals = np.zeros((self.n_rays, 2), dtype=np.float32)
        
        for i, angle in enumerate(self.angles):
            # Ray direction
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            directions[i] = direction
            
            # Cast ray through obstacles
            dist, hit_point, hit_normal = obstacle_map.cast_ray(pos, direction, self.max_range)
            
            # Also check world boundaries
            boundary_dist = self._check_boundary_intersection(pos, direction, world_size)
            
            if boundary_dist < dist:
                dist = boundary_dist
                # For boundary, normal points inward (opposite to ray direction)
                hit_normal = -direction
            
            distances[i] = dist
            
            if hit_normal is not None:
                normals[i] = hit_normal
            else:
                # No hit, use reverse ray direction as "normal"
                normals[i] = -direction
        
        return distances, directions, normals
    
    def _check_boundary_intersection(self, pos: np.ndarray, direction: np.ndarray, world_size: float) -> float:
        """Check intersection with world boundaries
        
        Args:
            pos: Position (2,)
            direction: Ray direction (2,)
            world_size: Size of world
            
        Returns:
            Distance to boundary intersection
        """
        half_size = world_size / 2
        min_dist = self.max_range
        
        # Check four boundaries
        # Right boundary: x = half_size
        if direction[0] > 1e-8:
            t = (half_size - pos[0]) / direction[0]
            if t > 0 and t < min_dist:
                y = pos[1] + t * direction[1]
                if abs(y) <= half_size:
                    min_dist = t
        
        # Left boundary: x = -half_size
        if direction[0] < -1e-8:
            t = (-half_size - pos[0]) / direction[0]
            if t > 0 and t < min_dist:
                y = pos[1] + t * direction[1]
                if abs(y) <= half_size:
                    min_dist = t
        
        # Top boundary: y = half_size
        if direction[1] > 1e-8:
            t = (half_size - pos[1]) / direction[1]
            if t > 0 and t < min_dist:
                x = pos[0] + t * direction[0]
                if abs(x) <= half_size:
                    min_dist = t
        
        # Bottom boundary: y = -half_size
        if direction[1] < -1e-8:
            t = (-half_size - pos[1]) / direction[1]
            if t > 0 and t < min_dist:
                x = pos[0] + t * direction[0]
                if abs(x) <= half_size:
                    min_dist = t
        
        return min_dist


class MultiViewSensor:
    """Multi-view sensor for leader (combines multiple ray sensors)"""
    
    def __init__(self, n_rays: int, max_range: float):
        """Initialize multi-view sensor
        
        Args:
            n_rays: Number of rays per view
            max_range: Maximum sensing range
        """
        self.ray_sensor = RaySensor(n_rays, max_range)
    
    def sense(
        self, 
        pos: np.ndarray, 
        obstacle_map: ObstacleMap,
        world_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform multi-view sensing
        
        Currently uses single view from agent position.
        Can be extended to multiple viewpoints.
        
        Args:
            pos: Agent position (2,)
            obstacle_map: Obstacle map
            world_size: Size of world
            
        Returns:
            (distances, directions, normals) from ray sensor
        """
        return self.ray_sensor.sense(pos, obstacle_map, world_size)


class AvoidanceSensor:
    """Short-range sensor specifically for obstacle avoidance"""
    
    def __init__(self, n_rays: int, max_range: float):
        """Initialize avoidance sensor
        
        Args:
            n_rays: Number of rays
            max_range: Maximum sensing range
        """
        self.ray_sensor = RaySensor(n_rays, max_range)
    
    def sense(
        self, 
        pos: np.ndarray, 
        obstacle_map: ObstacleMap,
        world_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform avoidance sensing
        
        Args:
            pos: Agent position (2,)
            obstacle_map: Obstacle map
            world_size: Size of world
            
        Returns:
            (distances, directions, normals) from ray sensor
        """
        return self.ray_sensor.sense(pos, obstacle_map, world_size)
