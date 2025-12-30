"""Demo script to visualize obstacle avoidance in action"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from lf_mha_multi.envs.nav_env import LeaderFollowerNavEnv
from lf_mha_multi.utils.config import load_config


def demo_obstacle_avoidance():
    """Demonstrate obstacle avoidance behavior"""
    print("Obstacle Avoidance Demo")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Create environment
    env = LeaderFollowerNavEnv(config, training_stage=1)
    
    print(f"Environment created:")
    print(f"  Agents: {env.n_agents}")
    print(f"  Obstacles: {len(env.obstacle_map.walls)} walls")
    print(f"  Avoidance enabled: {env.avoid_controller.enabled}")
    print(f"  Safety distance: {env.avoid_controller.d_safe}")
    print(f"  Avoidance strength: {env.avoid_controller.k_avoid}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    # Run episode
    print("\nRunning episode...")
    trajectory = [env.positions.copy()]
    done = False
    truncated = False
    steps = 0
    max_steps = 200
    
    while not (done or truncated) and steps < max_steps:
        # Simple action: move toward goal
        action = env.action_space.sample() * 0.5  # Smaller actions
        obs, reward, done, truncated, info = env.step(action)
        trajectory.append(env.positions.copy())
        steps += 1
        
        if steps % 50 == 0:
            print(f"  Step {steps}: reward={reward:.3f}")
    
    print(f"\nEpisode finished:")
    print(f"  Steps: {steps}")
    print(f"  Success: {info['success']}")
    print(f"  Collision: {info['collision']}")
    print(f"  Boundary violation: {info['boundary_violation']}")
    
    # Plot trajectory
    print("\nPlotting trajectory...")
    trajectory = np.array(trajectory)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Setup
    lim = env.world_size / 2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Leader-Follower Navigation with Obstacle Avoidance')
    
    # Draw obstacles
    for wall in env.obstacle_map.walls:
        ax.plot([wall.start[0], wall.end[0]], 
               [wall.start[1], wall.end[1]], 
               'k-', linewidth=4, label='Obstacles' if wall == env.obstacle_map.walls[0] else '')
    
    # Draw goal
    goal_circle = Circle(env.goal_pos, env.goal_radius, color='green', alpha=0.3, label='Goal')
    ax.add_patch(goal_circle)
    
    # Draw trajectories
    colors = ['red', 'blue', 'cyan', 'magenta', 'yellow']
    labels = ['Leader', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4']
    
    for i in range(env.n_agents):
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else f'Agent {i}'
        
        # Plot trajectory
        ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], 
               color=color, alpha=0.6, linewidth=2, label=label)
        
        # Mark start
        ax.plot(trajectory[0, i, 0], trajectory[0, i, 1], 
               'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        # Mark end
        ax.plot(trajectory[-1, i, 0], trajectory[-1, i, 1], 
               's', color=color, markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    ax.legend(loc='upper left', fontsize=10)
    
    # Save figure
    output_path = '/tmp/obstacle_avoidance_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory saved to {output_path}")
    
    # Show obstacle avoidance in action at a specific timestep
    if len(trajectory) > 10:
        mid_step = len(trajectory) // 2
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'Obstacle Avoidance Visualization (Step {mid_step})')
        
        # Draw obstacles
        for wall in env.obstacle_map.walls:
            ax2.plot([wall.start[0], wall.end[0]], 
                    [wall.start[1], wall.end[1]], 
                    'k-', linewidth=4)
        
        # Draw agents with sensing rays
        for i in range(env.n_agents):
            pos = trajectory[mid_step, i]
            color = colors[i % len(colors)]
            
            # Agent circle
            agent_circle = Circle(pos, env.agent_radius, color=color, alpha=0.7)
            ax2.add_patch(agent_circle)
            
            # Draw avoidance sensor rays
            avoid_dists, avoid_dirs, _ = env.avoid_sensor.sense(
                pos, env.obstacle_map, env.world_size
            )
            
            for j in range(len(avoid_dists)):
                ray_end = pos + avoid_dirs[j] * avoid_dists[j]
                # Color based on distance (red=close, green=far)
                ray_color = 'red' if avoid_dists[j] < env.avoid_controller.d_safe else 'green'
                ax2.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], 
                        color=ray_color, alpha=0.3, linewidth=1)
        
        # Save figure
        output_path2 = '/tmp/obstacle_avoidance_rays.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"Ray visualization saved to {output_path2}")
    
    env.close()
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo_obstacle_avoidance()
