"""Training script for Follower (Stage 2)"""

import os
import argparse
from pathlib import Path
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from lf_mha_multi.envs.nav_env import LeaderFollowerNavEnv
from lf_mha_multi.models.follower_policy import FollowerFeaturesExtractor
from lf_mha_multi.utils.config import load_config
from lf_mha_multi.utils.seeding import set_global_seed


def make_env(config, rank, seed=0):
    """Create environment function for vectorization
    
    Args:
        config: Configuration dictionary
        rank: Process rank
        seed: Random seed
        
    Returns:
        Environment creation function
    """
    def _init():
        env = LeaderFollowerNavEnv(config, training_stage=2)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_follower(
    config_path: str = None,
    output_dir: str = "models/follower",
    leader_model_path: str = None,
    n_envs: int = None,
    total_timesteps: int = None
):
    """Train follower policy
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for models
        leader_model_path: Path to trained leader model (optional)
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
    """
    # Load config
    config = load_config(config_path)
    
    # Override with arguments if provided
    if n_envs is None:
        n_envs = config['training']['stage2']['n_envs']
    if total_timesteps is None:
        total_timesteps = config['training']['stage2']['total_timesteps']
    
    # Set seed
    seed = config.get('seed', 42)
    set_global_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training Follower (Stage 2)")
    print(f"  n_envs: {n_envs}")
    print(f"  total_timesteps: {total_timesteps}")
    print(f"  output_dir: {output_dir}")
    if leader_model_path:
        print(f"  leader_model: {leader_model_path}")
    
    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(config, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(config, 0, seed)])
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(config, 999, seed + 1000)])
    
    # TODO: Load frozen leader model if provided
    # For now, we use the built-in guidance generation in env
    
    # Policy kwargs with custom features extractor
    policy_kwargs = dict(
        features_extractor_class=FollowerFeaturesExtractor,
        features_extractor_kwargs=dict(
            hidden_dims=config['follower']['hidden_dims'],
            use_layer_norm=config['follower']['use_layer_norm']
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config['training']['stage2']['learning_rate'],
        n_steps=config['training']['stage2']['n_steps'],
        batch_size=config['training']['stage2']['batch_size'],
        n_epochs=config['training']['stage2']['n_epochs'],
        gamma=config['training']['stage2']['gamma'],
        gae_lambda=config['training']['stage2']['gae_lambda'],
        clip_range=config['training']['stage2']['clip_range'],
        ent_coef=config['training']['stage2']['ent_coef'],
        vf_coef=config['training']['stage2']['vf_coef'],
        max_grad_norm=config['training']['stage2']['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=f"{output_dir}/checkpoints",
        name_prefix="follower_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best",
        log_path=f"{output_dir}/eval",
        eval_freq=max(5000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    model.save(f"{output_dir}/final_model")
    print(f"Training complete! Model saved to {output_dir}/final_model")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Follower policy")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/follower",
        help="Output directory for models"
    )
    parser.add_argument(
        "--leader-model", type=str, default=None,
        help="Path to trained leader model"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Total training timesteps"
    )
    
    args = parser.parse_args()
    
    train_follower(
        config_path=args.config,
        output_dir=args.output_dir,
        leader_model_path=args.leader_model,
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps
    )
