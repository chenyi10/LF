"""Rendering and visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from typing import List, Tuple, Optional


class Renderer:
    """Renderer for Leader-Follower navigation environment"""
    
    def __init__(self, world_size: float = 20.0, figsize: Tuple[int, int] = (10, 10)):
        """Initialize renderer
        
        Args:
            world_size: Size of the world
            figsize: Figure size for rendering
        """
        self.world_size = world_size
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def setup(self):
        """Setup the figure and axis"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
        return self.fig, self.ax
    
    def render_frame(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List,
        guidance_vectors: Optional[np.ndarray] = None,
        guidance_weights: Optional[np.ndarray] = None,
        agent_radius: float = 0.3,
        goal_radius: float = 1.0
    ):
        """Render a single frame
        
        Args:
            positions: Agent positions (N, 2)
            velocities: Agent velocities (N, 2)
            goal_pos: Goal position (2,)
            obstacles: List of obstacles
            guidance_vectors: Leader guidance vectors (K, 2), optional
            guidance_weights: Leader guidance weights (K,), optional
            agent_radius: Radius of agents
            goal_radius: Radius of goal
        """
        fig, ax = self.setup()
        ax.clear()
        
        # Set limits
        lim = self.world_size / 2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Leader-Follower Navigation')
        
        # Draw obstacles
        for obs in obstacles:
            if obs['type'] == 'wall':
                start = obs['start']
                end = obs['end']
                ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=3)
        
        # Draw goal
        goal_circle = Circle(goal_pos, goal_radius, color='green', alpha=0.3, label='Goal')
        ax.add_patch(goal_circle)
        
        # Draw agents
        n_agents = len(positions)
        for i in range(n_agents):
            color = 'red' if i == 0 else 'blue'  # Leader is red, followers are blue
            label = 'Leader' if i == 0 else ('Follower' if i == 1 else None)
            
            # Agent circle
            agent_circle = Circle(positions[i], agent_radius, color=color, alpha=0.6, label=label)
            ax.add_patch(agent_circle)
            
            # Velocity vector
            if np.linalg.norm(velocities[i]) > 0.01:
                ax.arrow(
                    positions[i, 0], positions[i, 1],
                    velocities[i, 0] * 0.3, velocities[i, 1] * 0.3,
                    head_width=0.15, head_length=0.1, fc=color, ec=color, alpha=0.5
                )
        
        # Draw guidance vectors from leader
        if guidance_vectors is not None and guidance_weights is not None:
            leader_pos = positions[0]
            for k in range(len(guidance_vectors)):
                gvec = guidance_vectors[k]
                weight = guidance_weights[k]
                # Draw guidance direction with thickness proportional to weight
                ax.arrow(
                    leader_pos[0], leader_pos[1],
                    gvec[0] * 2.0, gvec[1] * 2.0,
                    head_width=0.2, head_length=0.15,
                    fc='orange', ec='orange', alpha=0.3 + 0.7 * weight,
                    linewidth=1 + 3 * weight,
                    linestyle='--'
                )
                # Add weight label
                text_pos = leader_pos + gvec * 2.2
                ax.text(text_pos[0], text_pos[1], f'{weight:.2f}',
                       fontsize=8, ha='center', color='orange')
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig
    
    def save_frame(self, filename: str):
        """Save current frame to file
        
        Args:
            filename: Output filename
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=100, bbox_inches='tight')
    
    def close(self):
        """Close the renderer"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def plot_trajectories(
    trajectories: List[np.ndarray],
    goal_pos: np.ndarray,
    obstacles: List,
    world_size: float = 20.0,
    save_path: Optional[str] = None
):
    """Plot agent trajectories
    
    Args:
        trajectories: List of trajectory arrays, each (T, N, 2)
        goal_pos: Goal position (2,)
        obstacles: List of obstacles
        world_size: Size of world
        save_path: Path to save figure, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    lim = world_size / 2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Agent Trajectories')
    
    # Draw obstacles
    for obs in obstacles:
        if obs['type'] == 'wall':
            start = obs['start']
            end = obs['end']
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=3)
    
    # Draw goal
    goal_circle = Circle(goal_pos, 1.0, color='green', alpha=0.3, label='Goal')
    ax.add_patch(goal_circle)
    
    # Draw trajectories
    colors = ['red'] + ['blue'] * 10  # Leader red, followers blue
    for traj in trajectories:
        T, N, _ = traj.shape
        for i in range(N):
            ax.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], alpha=0.5, linewidth=1)
            # Mark start and end
            ax.plot(traj[0, i, 0], traj[0, i, 1], 'o', color=colors[i], markersize=6)
            ax.plot(traj[-1, i, 0], traj[-1, i, 1], 's', color=colors[i], markersize=6)
    
    ax.legend(['Obstacles', 'Goal', 'Leader', 'Followers'], loc='upper right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
