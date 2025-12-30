"""Main navigation environment for Leader-Follower multi-agent system"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List

from .obstacles import ObstacleMap
from .sensors import MultiViewSensor, AvoidanceSensor
from .reward import RewardCalculator
from ..controllers.obstacle_avoid import ObstacleAvoidanceController
from ..controllers.formation_pd import FormationPDController


class LeaderFollowerNavEnv(gym.Env):
    """
    Leader-Follower multi-agent navigation environment
    
    Supports two training stages:
    - Stage 1: Train leader (followers use fixed policy)
    - Stage 2: Train followers (leader frozen)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Dict[str, Any], training_stage: int = 1):
        """Initialize environment
        
        Args:
            config: Configuration dictionary
            training_stage: 1 for leader training, 2 for follower training
        """
        super().__init__()
        
        self.config = config
        self.training_stage = training_stage
        
        # Environment parameters
        self.n_agents = config['env']['n_agents']
        self.max_steps = config['env']['max_steps']
        self.dt = config['env']['dt']
        self.world_size = config['env']['world_size']
        self.agent_radius = config['env']['agent_radius']
        self.v_max = config['env']['v_max']
        self.a_max = config['env']['a_max']
        self.goal_radius = config['env']['goal_radius']
        self.spawn_radius = config['env']['spawn_radius']
        
        # Initialize obstacles
        self.obstacle_map = ObstacleMap(config['obstacles'])
        
        # Initialize sensors
        self.leader_sensor = MultiViewSensor(
            config['sensors']['n_rays_leader'],
            config['sensors']['ray_length_leader']
        )
        self.follower_sensor = AvoidanceSensor(
            config['sensors']['n_rays_follower'],
            config['sensors']['ray_length_follower']
        )
        self.avoid_sensor = AvoidanceSensor(
            config['sensors']['n_rays_avoid'],
            config['sensors']['d_max_avoid']
        )
        
        # Initialize controllers
        self.avoid_controller = ObstacleAvoidanceController(
            d_safe=config['obstacle_avoidance']['d_safe'],
            k_avoid=config['obstacle_avoidance']['k_avoid'],
            a_avoid_max=config['obstacle_avoidance']['a_avoid_max'],
            d_proj=config['obstacle_avoidance']['d_proj'],
            enabled=config['obstacle_avoidance']['enabled']
        )
        
        self.formation_controller = FormationPDController(
            k_p=config['formation']['k_p'],
            k_d=config['formation']['k_d'],
            a_form_max=config['formation']['a_form_max']
        )
        
        # Formation offsets
        self.formation_offsets = np.array(
            config['formation']['offsets'][:self.n_agents-1],
            dtype=np.float32
        )
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(config)
        
        # Leader parameters
        self.n_hypotheses = config['leader']['n_hypotheses']
        self.guidance_selection = config['leader']['guidance_selection']
        self.k_guidance = config['leader']['k_guidance']
        
        # State variables
        self.positions = None
        self.velocities = None
        self.goal_pos = None
        self.steps = 0
        
        # Guidance (for stage 1 and stage 2)
        self.guidance_vectors = None  # (K, 2)
        self.guidance_weights = None  # (K,)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Rendering
        self.renderer = None
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Leader observation: [ray_distances, goal_vec, group_summary]
        n_rays_leader = self.config['sensors']['n_rays_leader']
        leader_obs_dim = n_rays_leader + 2 + 5  # rays + goal(2) + summary(5)
        
        # Follower observation: [local_pos, local_vel, ray_distances, guidance_vecs, guidance_weights]
        n_rays_follower = self.config['sensors']['n_rays_follower']
        follower_obs_dim = 2 + 2 + n_rays_follower + self.n_hypotheses * 2 + self.n_hypotheses
        
        if self.training_stage == 1:
            # Stage 1: Train leader, output is dummy (guidance produced internally)
            self.observation_space = spaces.Dict({
                'leader_obs': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(leader_obs_dim,), dtype=np.float32
                )
            })
            # Dummy action space (guidance vectors and weights produced by policy network)
            # For simplicity, we use a small dummy action
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            )
        else:
            # Stage 2: Train followers
            self.observation_space = spaces.Dict({
                'follower_obs': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.n_agents-1, follower_obs_dim), dtype=np.float32
                )
            })
            # Follower actions: (n_agents-1, 2)
            self.action_space = spaces.Box(
                low=-1, high=1,
                shape=(self.n_agents-1, 2), dtype=np.float32
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        
        # Reset state
        self.steps = 0
        self.reward_calculator.reset()
        
        # Initialize positions in a cluster near origin
        self.positions = np.random.uniform(
            -self.spawn_radius, self.spawn_radius,
            size=(self.n_agents, 2)
        ).astype(np.float32)
        
        # Initialize velocities to zero
        self.velocities = np.zeros((self.n_agents, 2), dtype=np.float32)
        
        # Set goal position (far from start)
        self.goal_pos = np.array([8.0, 0.0], dtype=np.float32)
        
        # Initialize guidance (random)
        self.guidance_vectors = np.random.randn(self.n_hypotheses, 2).astype(np.float32)
        self.guidance_vectors /= (np.linalg.norm(self.guidance_vectors, axis=1, keepdims=True) + 1e-8)
        self.guidance_weights = np.ones(self.n_hypotheses, dtype=np.float32) / self.n_hypotheses
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step environment
        
        Args:
            action: Action from policy
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        self.steps += 1
        
        if self.training_stage == 1:
            # Stage 1: Leader training
            # Action from leader policy should contain guidance (handled externally)
            # For now, we use a simple guidance toward goal
            # In practice, the policy network outputs guidance_vectors and guidance_weights
            # and we extract them before calling step()
            
            # Generate guidance toward goal (as fallback)
            goal_vec = self.goal_pos - self.positions[0]
            goal_dist = np.linalg.norm(goal_vec)
            if goal_dist > 1e-6:
                main_direction = goal_vec / goal_dist
            else:
                main_direction = np.array([1.0, 0.0])
            
            # Create K hypotheses (perturbations of main direction)
            self.guidance_vectors = np.zeros((self.n_hypotheses, 2), dtype=np.float32)
            for k in range(self.n_hypotheses):
                angle_offset = (k - self.n_hypotheses // 2) * 0.3  # +/- 0.3 rad
                cos_a = np.cos(angle_offset)
                sin_a = np.sin(angle_offset)
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                self.guidance_vectors[k] = rot_mat @ main_direction
            
            # Equal weights for now (policy should learn this)
            self.guidance_weights = np.ones(self.n_hypotheses, dtype=np.float32) / self.n_hypotheses
            
            # Followers use fixed policy: follow guidance + avoidance + formation
            actions = self._compute_stage1_actions()
        
        else:
            # Stage 2: Follower training
            # Leader guidance is frozen (from trained leader policy)
            # Follower actions come from policy
            follower_actions_rl = action  # (n_agents-1, 2)
            actions = self._compute_stage2_actions(follower_actions_rl)
        
        # Apply actions (with clipping)
        actions = np.clip(actions, -1.0, 1.0) * self.a_max
        
        # Update velocities and positions
        self.velocities += actions * self.dt
        
        # Apply obstacle avoidance velocity projection
        for i in range(self.n_agents):
            avoid_dists, avoid_dirs, avoid_normals = self.avoid_sensor.sense(
                self.positions[i], self.obstacle_map, self.world_size
            )
            self.velocities[i] = self.avoid_controller.project_velocity(
                self.velocities[i], avoid_dists, avoid_dirs, avoid_normals
            )
        
        # Clip velocities
        vel_norms = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        vel_norms = np.maximum(vel_norms, 1e-8)
        self.velocities = np.where(
            vel_norms > self.v_max,
            self.velocities * (self.v_max / vel_norms),
            self.velocities
        )
        
        self.positions += self.velocities * self.dt
        
        # Check termination conditions
        collision = self._check_collision()
        boundary_violation = self._check_boundary()
        success = self.reward_calculator.check_success(
            self.positions, self.goal_pos, self.goal_radius
        )
        
        terminated = self.reward_calculator.calculate_done(
            self.positions, self.goal_pos, self.goal_radius,
            collision, boundary_violation, self.steps, self.max_steps
        )
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            self.positions, self.goal_pos,
            collision, boundary_violation, success,
            self.guidance_vectors, self.guidance_weights
        )
        
        obs = self._get_observation()
        truncated = False
        info = {
            'success': success,
            'collision': collision,
            'boundary_violation': boundary_violation,
            'steps': self.steps
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_stage1_actions(self) -> np.ndarray:
        """Compute actions for stage 1 (leader training)
        
        Followers use: a_i = a_i^avoid + a_i^form + k_g * g_selected
        Leader uses: a_0 = a_0^avoid + k_g * g_selected (no formation)
        
        Returns:
            Actions (n_agents, 2)
        """
        actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        
        # Select guidance (sample or argmax)
        if self.guidance_selection == "sample":
            k_selected = np.random.choice(self.n_hypotheses, p=self.guidance_weights)
        else:  # argmax
            k_selected = np.argmax(self.guidance_weights)
        
        g_selected = self.guidance_vectors[k_selected]
        
        # Leader action
        avoid_dists, avoid_dirs, avoid_normals = self.avoid_sensor.sense(
            self.positions[0], self.obstacle_map, self.world_size
        )
        a_avoid = self.avoid_controller.compute_avoidance_action(
            avoid_dists, avoid_dirs, avoid_normals
        )
        actions[0] = a_avoid + self.k_guidance * g_selected
        
        # Follower actions
        for i in range(1, self.n_agents):
            # Avoidance
            avoid_dists, avoid_dirs, avoid_normals = self.avoid_sensor.sense(
                self.positions[i], self.obstacle_map, self.world_size
            )
            a_avoid = self.avoid_controller.compute_avoidance_action(
                avoid_dists, avoid_dirs, avoid_normals
            )
            
            # Formation
            a_form = self.formation_controller.compute_formation_action(
                self.positions[i], self.velocities[i],
                self.positions[0], self.formation_offsets[i-1]
            )
            
            # Guidance
            a_guidance = self.k_guidance * g_selected
            
            actions[i] = a_avoid + a_form + a_guidance
        
        return actions
    
    def _compute_stage2_actions(self, follower_actions_rl: np.ndarray) -> np.ndarray:
        """Compute actions for stage 2 (follower training)
        
        Leader uses frozen policy
        Followers use: a_i = a_i^RL + a_i^avoid + a_i^form
        
        Args:
            follower_actions_rl: RL actions for followers (n_agents-1, 2)
            
        Returns:
            Actions (n_agents, 2)
        """
        actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        
        # Leader action (using guidance - should be from frozen policy)
        if self.guidance_selection == "sample":
            k_selected = np.random.choice(self.n_hypotheses, p=self.guidance_weights)
        else:
            k_selected = np.argmax(self.guidance_weights)
        g_selected = self.guidance_vectors[k_selected]
        
        avoid_dists, avoid_dirs, avoid_normals = self.avoid_sensor.sense(
            self.positions[0], self.obstacle_map, self.world_size
        )
        a_avoid = self.avoid_controller.compute_avoidance_action(
            avoid_dists, avoid_dirs, avoid_normals
        )
        actions[0] = a_avoid + self.k_guidance * g_selected
        
        # Follower actions
        for i in range(1, self.n_agents):
            # Avoidance
            avoid_dists, avoid_dirs, avoid_normals = self.avoid_sensor.sense(
                self.positions[i], self.obstacle_map, self.world_size
            )
            a_avoid = self.avoid_controller.compute_avoidance_action(
                avoid_dists, avoid_dirs, avoid_normals
            )
            
            # Formation
            a_form = self.formation_controller.compute_formation_action(
                self.positions[i], self.velocities[i],
                self.positions[0], self.formation_offsets[i-1]
            )
            
            # RL action
            a_rl = follower_actions_rl[i-1]
            
            actions[i] = a_rl + a_avoid + a_form
        
        return actions
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation
        
        Returns:
            Observation dictionary
        """
        if self.training_stage == 1:
            # Leader observation
            leader_pos = self.positions[0]
            
            # Ray sensing
            ray_dists, _, _ = self.leader_sensor.sense(
                leader_pos, self.obstacle_map, self.world_size
            )
            
            # Goal vector (relative)
            goal_vec = self.goal_pos - leader_pos
            
            # Group summary
            centroid = np.mean(self.positions, axis=0)
            mean_vel = np.mean(self.velocities, axis=0)
            spread = np.mean(np.linalg.norm(self.positions - centroid, axis=1))
            
            summary = np.concatenate([
                centroid, mean_vel, [spread]
            ]).astype(np.float32)
            
            leader_obs = np.concatenate([ray_dists, goal_vec, summary])
            
            return {'leader_obs': leader_obs}
        
        else:
            # Follower observations
            follower_obs_list = []
            
            for i in range(1, self.n_agents):
                # Local position and velocity (relative to leader)
                local_pos = self.positions[i] - self.positions[0]
                local_vel = self.velocities[i]
                
                # Ray sensing
                ray_dists, _, _ = self.follower_sensor.sense(
                    self.positions[i], self.obstacle_map, self.world_size
                )
                
                # Guidance vectors and weights
                guidance_flat = self.guidance_vectors.flatten()
                
                obs = np.concatenate([
                    local_pos, local_vel, ray_dists,
                    guidance_flat, self.guidance_weights
                ]).astype(np.float32)
                
                follower_obs_list.append(obs)
            
            follower_obs = np.stack(follower_obs_list, axis=0)
            
            return {'follower_obs': follower_obs}
    
    def _check_collision(self) -> bool:
        """Check if any agent collides with obstacles
        
        Returns:
            True if collision detected
        """
        for i in range(self.n_agents):
            if self.obstacle_map.check_collision_circle(
                self.positions[i], self.agent_radius
            ):
                return True
        
        # Check inter-agent collision
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < 2 * self.agent_radius:
                    return True
        
        return False
    
    def _check_boundary(self) -> bool:
        """Check if any agent is out of bounds
        
        Returns:
            True if boundary violation detected
        """
        half_size = self.world_size / 2
        for i in range(self.n_agents):
            if (abs(self.positions[i, 0]) > half_size or
                abs(self.positions[i, 1]) > half_size):
                return True
        return False
    
    def render(self, mode='human'):
        """Render environment
        
        Args:
            mode: Render mode
        """
        if self.renderer is None:
            from ..utils.render import Renderer
            self.renderer = Renderer(self.world_size)
        
        fig = self.renderer.render_frame(
            self.positions,
            self.velocities,
            self.goal_pos,
            self.obstacle_map.get_obstacles_as_list(),
            self.guidance_vectors,
            self.guidance_weights,
            self.agent_radius,
            self.goal_radius
        )
        
        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.pause(0.01)
        
        return fig
    
    def close(self):
        """Close environment"""
        if self.renderer is not None:
            self.renderer.close()
