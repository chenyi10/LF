"""Evaluation script for trained Leader-Follower system"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO

from lf_mha_multi.envs.nav_env import LeaderFollowerNavEnv
from lf_mha_multi.utils.config import load_config
from lf_mha_multi.utils.render import plot_trajectories


def evaluate_model(
    config_path: str = None,
    leader_model_path: str = None,
    follower_model_path: str = None,
    n_episodes: int = 50,
    render: bool = False,
    save_trajectories: bool = True,
    output_dir: str = "results"
):
    """Evaluate trained models
    
    Args:
        config_path: Path to config file
        leader_model_path: Path to trained leader model
        follower_model_path: Path to trained follower model
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
        save_trajectories: Whether to save trajectory plots
        output_dir: Output directory
    """
    # Load config
    config = load_config(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating models")
    print(f"  n_episodes: {n_episodes}")
    print(f"  output_dir: {output_dir}")
    
    # Create environment (stage 2 for full evaluation)
    env = LeaderFollowerNavEnv(config, training_stage=2)
    
    # Load models if provided
    leader_model = None
    follower_model = None
    
    if leader_model_path and os.path.exists(leader_model_path + ".zip"):
        print(f"Loading leader model from {leader_model_path}")
        # Leader model not used directly in stage 2 env currently
        # (guidance is generated in env)
    
    if follower_model_path and os.path.exists(follower_model_path + ".zip"):
        print(f"Loading follower model from {follower_model_path}")
        follower_model = PPO.load(follower_model_path)
    
    # Evaluation metrics
    successes = []
    collisions = []
    boundary_violations = []
    episode_lengths = []
    trajectories = []
    
    # Run episodes
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        
        # Store trajectory
        traj = [env.positions.copy()]
        
        while not (done or truncated):
            # Get action
            if follower_model is not None:
                action, _ = follower_model.predict(obs, deterministic=True)
            else:
                # Random action
                action = env.action_space.sample()
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            # Store trajectory
            traj.append(env.positions.copy())
            
            # Render if requested
            if render:
                env.render()
        
        # Record metrics
        successes.append(info['success'])
        collisions.append(info['collision'])
        boundary_violations.append(info['boundary_violation'])
        episode_lengths.append(steps)
        
        if save_trajectories:
            trajectories.append(np.array(traj))
        
        print(f"Episode {episode+1}/{n_episodes}: "
              f"Success={info['success']}, "
              f"Collision={info['collision']}, "
              f"Steps={steps}")
    
    # Calculate statistics
    success_rate = np.mean(successes)
    collision_rate = np.mean(collisions)
    boundary_rate = np.mean(boundary_violations)
    avg_steps = np.mean(episode_lengths)
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Collision Rate: {collision_rate*100:.1f}%")
    print(f"  Boundary Violation Rate: {boundary_rate*100:.1f}%")
    print(f"  Average Steps: {avg_steps:.1f}")
    print("="*50)
    
    # Save results
    results = {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'boundary_violation_rate': boundary_rate,
        'average_steps': avg_steps,
        'successes': successes,
        'collisions': collisions,
        'boundary_violations': boundary_violations,
        'episode_lengths': episode_lengths
    }
    
    np.save(f"{output_dir}/eval_results.npy", results)
    print(f"Results saved to {output_dir}/eval_results.npy")
    
    # Plot trajectories
    if save_trajectories and len(trajectories) > 0:
        # Plot first 5 successful trajectories
        successful_trajs = [traj for traj, success in zip(trajectories, successes) if success]
        
        if len(successful_trajs) > 0:
            plot_trajectories(
                successful_trajs[:min(5, len(successful_trajs))],
                env.goal_pos,
                env.obstacle_map.get_obstacles_as_list(),
                env.world_size,
                save_path=f"{output_dir}/trajectories_success.png"
            )
            print(f"Success trajectories saved to {output_dir}/trajectories_success.png")
        
        # Plot first 5 failed trajectories
        failed_trajs = [traj for traj, success in zip(trajectories, successes) if not success]
        
        if len(failed_trajs) > 0:
            plot_trajectories(
                failed_trajs[:min(5, len(failed_trajs))],
                env.goal_pos,
                env.obstacle_map.get_obstacles_as_list(),
                env.world_size,
                save_path=f"{output_dir}/trajectories_failed.png"
            )
            print(f"Failed trajectories saved to {output_dir}/trajectories_failed.png")
    
    env.close()


def compare_configurations(
    config_path: str = None,
    output_dir: str = "results/comparison"
):
    """Compare different configurations (K=1 vs K=3, MHA vs MLP)
    
    Args:
        config_path: Path to config file
        output_dir: Output directory
    """
    # Load base config
    base_config = load_config(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Comparing configurations...")
    
    # Configurations to compare
    configs = [
        {"name": "K=1_MHA", "n_hypotheses": 1, "use_mha": True},
        {"name": "K=3_MHA", "n_hypotheses": 3, "use_mha": True},
        {"name": "K=3_MLP", "n_hypotheses": 3, "use_mha": False},
    ]
    
    results_comparison = {}
    
    for cfg in configs:
        print(f"\nEvaluating {cfg['name']}...")
        
        # Modify config
        config = base_config.copy()
        config['leader']['n_hypotheses'] = cfg['n_hypotheses']
        config['leader']['use_mha'] = cfg['use_mha']
        
        # Create environment
        env = LeaderFollowerNavEnv(config, training_stage=2)
        
        # Run episodes
        n_episodes = 20
        successes = []
        steps_list = []
        
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            steps = 0
            
            while not (done or truncated) and steps < 500:
                # Random action for comparison
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
            
            successes.append(info['success'])
            steps_list.append(steps)
        
        env.close()
        
        # Store results
        results_comparison[cfg['name']] = {
            'success_rate': np.mean(successes),
            'avg_steps': np.mean(steps_list)
        }
        
        print(f"  Success Rate: {np.mean(successes)*100:.1f}%")
        print(f"  Avg Steps: {np.mean(steps_list):.1f}")
    
    # Save comparison results
    np.save(f"{output_dir}/comparison_results.npy", results_comparison)
    print(f"\nComparison results saved to {output_dir}/comparison_results.npy")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = list(results_comparison.keys())
    success_rates = [results_comparison[name]['success_rate'] for name in names]
    avg_steps = [results_comparison[name]['avg_steps'] for name in names]
    
    axes[0].bar(names, success_rates)
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_ylim([0, 1])
    
    axes[1].bar(names, avg_steps)
    axes[1].set_ylabel('Average Steps')
    axes[1].set_title('Average Steps Comparison')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plot.png", dpi=150)
    print(f"Comparison plot saved to {output_dir}/comparison_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Leader-Follower models")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--leader-model", type=str, default=None,
        help="Path to trained leader model"
    )
    parser.add_argument(
        "--follower-model", type=str, default=None,
        help="Path to trained follower model"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=50,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render episodes"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run configuration comparison"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations(
            config_path=args.config,
            output_dir=f"{args.output_dir}/comparison"
        )
    else:
        evaluate_model(
            config_path=args.config,
            leader_model_path=args.leader_model,
            follower_model_path=args.follower_model,
            n_episodes=args.n_episodes,
            render=args.render,
            output_dir=args.output_dir
        )
