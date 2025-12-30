"""Formation control using PD controller"""

import numpy as np


class FormationPDController:
    """PD controller for formation maintenance"""
    
    def __init__(
        self,
        k_p: float = 0.5,
        k_d: float = 0.3,
        a_form_max: float = 0.4
    ):
        """Initialize formation PD controller
        
        Args:
            k_p: Position gain
            k_d: Velocity damping gain
            a_form_max: Maximum formation action magnitude
        """
        self.k_p = k_p
        self.k_d = k_d
        self.a_form_max = a_form_max
    
    def compute_formation_action(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        leader_pos: np.ndarray,
        formation_offset: np.ndarray
    ) -> np.ndarray:
        """Compute formation control action for a single follower
        
        The desired position is: p_desired = leader_pos + formation_offset
        PD control: a = k_p * (p_desired - p_agent) - k_d * v_agent
        
        Args:
            agent_pos: Agent position (2,)
            agent_vel: Agent velocity (2,)
            leader_pos: Leader position (2,)
            formation_offset: Formation offset relative to leader (2,)
            
        Returns:
            Formation control action (2,)
        """
        # Desired position
        desired_pos = leader_pos + formation_offset
        
        # Position error
        pos_error = desired_pos - agent_pos
        
        # PD control
        action = self.k_p * pos_error - self.k_d * agent_vel
        
        # Clip to maximum magnitude
        action_norm = np.linalg.norm(action)
        if action_norm > self.a_form_max:
            action = action * (self.a_form_max / action_norm)
        
        return action
    
    def compute_formation_actions_batch(
        self,
        follower_positions: np.ndarray,
        follower_velocities: np.ndarray,
        leader_pos: np.ndarray,
        formation_offsets: np.ndarray
    ) -> np.ndarray:
        """Compute formation actions for all followers in batch
        
        Args:
            follower_positions: Follower positions (N-1, 2)
            follower_velocities: Follower velocities (N-1, 2)
            leader_pos: Leader position (2,)
            formation_offsets: Formation offsets (N-1, 2)
            
        Returns:
            Formation actions (N-1, 2)
        """
        n_followers = len(follower_positions)
        actions = np.zeros_like(follower_positions)
        
        for i in range(n_followers):
            actions[i] = self.compute_formation_action(
                follower_positions[i],
                follower_velocities[i],
                leader_pos,
                formation_offsets[i]
            )
        
        return actions
