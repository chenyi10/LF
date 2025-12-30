"""Training script for Leader (Stage 1)"""

import os
import argparse
from pathlib import Path
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from lf_mha_multi.envs.nav_env import LeaderFollowerNavEnv
from lf_mha_multi.models.leader_mha import LeaderMHAFeaturesExtractor
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
        env = LeaderFollowerNavEnv(config, training_stage=1)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_leader(
    config_path: str = None,
    output_dir: str = "models/leader",
    n_envs: int = None,
    total_timesteps: int = None
):
    """Train leader policy
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for models
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
    """
    # Load config
    config = load_config(config_path)
    
    # Override with arguments if provided
    if n_envs is None:
        n_envs = config['training']['stage1']['n_envs']
    if total_timesteps is None:
        total_timesteps = config['training']['stage1']['total_timesteps']
    
    # Set seed
    seed = config.get('seed', 42)
    set_global_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training Leader (Stage 1)")
    print(f"  n_envs: {n_envs}")
    print(f"  total_timesteps: {total_timesteps}")
    print(f"  output_dir: {output_dir}")
    
    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(config, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(config, 0, seed)])
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(config, 999, seed + 1000)])
    
    # Policy kwargs with custom features extractor
    policy_kwargs = dict(
        features_extractor_class=LeaderMHAFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_heads=config['leader']['n_heads'],
            d_model=config['leader']['d_model'],
            d_feedforward=config['leader']['d_feedforward'],
            dropout=config['leader']['dropout'],
            ray_encoding_dim=config['leader']['ray_encoding_dim'],
            goal_encoding_dim=config['leader']['goal_encoding_dim'],
            summary_encoding_dim=config['leader']['summary_encoding_dim']
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config['training']['stage1']['learning_rate'],
        n_steps=config['training']['stage1']['n_steps'],
        batch_size=config['training']['stage1']['batch_size'],
        n_epochs=config['training']['stage1']['n_epochs'],
        gamma=config['training']['stage1']['gamma'],
        gae_lambda=config['training']['stage1']['gae_lambda'],
        clip_range=config['training']['stage1']['clip_range'],
        ent_coef=config['training']['stage1']['ent_coef'],
        vf_coef=config['training']['stage1']['vf_coef'],
        max_grad_norm=config['training']['stage1']['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=f"{output_dir}/checkpoints",
        name_prefix="leader_model"
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
    parser = argparse.ArgumentParser(description="Train Leader policy")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/leader",
        help="Output directory for models"
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
    
    train_leader(
        config_path=args.config,
        output_dir=args.output_dir,
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps
    )
