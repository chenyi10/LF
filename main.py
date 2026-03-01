"""Main entry point for the multi-UAV residual policy control simulation.

Usage examples::

    # Run a simulation with the base PD controller only (no training)
    python main.py --mode simulate

    # Train the residual policy network
    python main.py --mode train

    # Train then simulate with the trained residual policy
    python main.py --mode train_and_simulate

    # Run simulation with a pre-trained residual policy checkpoint
    python main.py --mode simulate --checkpoint checkpoints/residual_policy_ep100.pt
"""

import argparse
import os
import sys

import yaml
import torch


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_simulation(config: dict, checkpoint: str | None = None) -> None:
    """Run the multi-UAV simulation.

    Args:
        config: Configuration dictionary.
        checkpoint: Optional path to a residual policy checkpoint (.pt file).
    """
    from simulation.engine import SimulationEngine
    from controllers.residual_controller import ResidualPolicyNetwork, ResidualPolicyController
    from utils.logger import SimulationLogger

    log_cfg = config.get("logging", {})
    logger = SimulationLogger(
        output_dir=log_cfg.get("output_dir", "logs"),
        log_interval=log_cfg.get("log_interval", 1),
    )

    engine = SimulationEngine(config)

    if checkpoint is not None:
        res_cfg = config["residual_policy"]
        network = ResidualPolicyNetwork(
            input_dim=9,
            hidden_dims=res_cfg["hidden_dims"],
            output_dim=3,
            activation=res_cfg["activation"],
            a_residual_max=res_cfg["a_residual_max"],
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        network.load_state_dict(torch.load(checkpoint, map_location=device))
        ctrl = ResidualPolicyController(network, device=device)
        engine.set_residual_controller(ctrl)
        print(f"Loaded residual policy from {checkpoint}")
    else:
        print("No checkpoint provided – running with base PD controller only.")

    print(f"Running simulation for {config['simulation']['max_steps']} steps...")
    log = engine.run()
    logger.log_all(log)

    final_error = log[-1]["mean_formation_error"]
    print(f"Simulation complete. Final mean formation error: {final_error:.4f} m")

    if log_cfg.get("save_trajectories", True):
        csv_path = logger.save_csv()
        traj_2d = logger.plot_trajectories_2d()
        traj_3d = logger.plot_trajectories_3d()
        err_plot = logger.plot_formation_errors()
        ctrl_plot = logger.plot_control_inputs()
        print(f"Saved logs to {log_cfg.get('output_dir', 'logs')}/")
        for p in [csv_path, traj_2d, traj_3d, err_plot, ctrl_plot]:
            if p:
                print(f"  {p}")


def run_training(config: dict) -> str:
    """Train the residual policy network.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to the final saved checkpoint.
    """
    from training.trainer import ResidualPolicyTrainer

    trainer = ResidualPolicyTrainer(config)
    metrics = trainer.train()

    n_episodes = len(metrics["episode_errors"])
    if n_episodes > 0:
        print(
            f"\nTraining complete. Final mean formation error: "
            f"{metrics['episode_errors'][-1]:.4f} m"
        )

    checkpoint_dir = config["training"]["checkpoint_dir"]
    final_checkpoint = os.path.join(checkpoint_dir, "residual_policy_final.pt")
    trainer.save(final_checkpoint)
    return final_checkpoint


def main() -> None:
    """Parse arguments and run the requested mode."""
    parser = argparse.ArgumentParser(
        description="Multi-UAV Residual Policy Control Simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "train", "train_and_simulate"],
        default="simulate",
        help="Execution mode (default: simulate)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a pre-trained residual policy checkpoint (.pt file)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    if args.mode == "simulate":
        run_simulation(config, checkpoint=args.checkpoint)
    elif args.mode == "train":
        run_training(config)
    elif args.mode == "train_and_simulate":
        checkpoint = run_training(config)
        run_simulation(config, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
