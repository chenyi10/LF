"""Training framework for the residual policy neural network."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple

from controllers.base_controller import BaseLeaderFollowerController
from controllers.residual_controller import ResidualPolicyNetwork, ResidualPolicyController
from simulation.engine import SimulationEngine


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._next_states: List[np.ndarray] = []
        self._ptr = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        if len(self._states) < self.capacity:
            self._states.append(state)
            self._actions.append(action)
            self._rewards.append(reward)
            self._next_states.append(next_state)
        else:
            idx = self._ptr % self.capacity
            self._states[idx] = state
            self._actions[idx] = action
            self._rewards[idx] = reward
            self._next_states[idx] = next_state
        self._ptr += 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        n = len(self._states)
        idxs = np.random.randint(0, n, size=batch_size)
        states = np.stack([self._states[i] for i in idxs])
        actions = np.stack([self._actions[i] for i in idxs])
        rewards = np.array([self._rewards[i] for i in idxs], dtype=np.float32)
        next_states = np.stack([self._next_states[i] for i in idxs])
        return states, actions, rewards, next_states

    def __len__(self) -> int:
        return len(self._states)


class ResidualPolicyTrainer:
    """Trains the residual policy network via supervised imitation.

    The training minimises the mean squared formation error:
        L = mean( ||p_follower - (p_leader + offset)||^2 )

    The network is trained by running the full simulation, collecting
    (state, base_accel, formation_error) tuples, and using the position
    error signal to supervise the residual correction.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialise the trainer.

        Args:
            config: Full configuration dictionary.
        """
        self.config = config
        res_cfg = config["residual_policy"]
        train_cfg = config["training"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = res_cfg["learning_rate"]
        self.batch_size = res_cfg["batch_size"]
        self.gamma = res_cfg["gamma"]
        self.n_episodes = train_cfg["n_episodes"]
        self.max_steps = train_cfg["max_steps_per_episode"]
        self.eval_interval = train_cfg["eval_interval"]
        self.save_interval = train_cfg["save_interval"]
        self.checkpoint_dir = train_cfg["checkpoint_dir"]

        # Build network
        self.network = ResidualPolicyNetwork(
            input_dim=9,
            hidden_dims=res_cfg["hidden_dims"],
            output_dim=3,
            activation=res_cfg["activation"],
            a_residual_max=res_cfg["a_residual_max"],
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer()

        # Simulation engine (residual ctrl not yet attached)
        self.engine = SimulationEngine(config)

        # Training metrics
        self.episode_errors: List[float] = []
        self.episode_losses: List[float] = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_state(
        self,
        follower_pos: np.ndarray,
        follower_vel: np.ndarray,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
    ) -> np.ndarray:
        """Build the flat state vector for one follower."""
        return np.concatenate([
            follower_pos - leader_pos,
            follower_vel - leader_vel,
            leader_vel,
        ]).astype(np.float32)

    def _collect_episode(self, use_residual: bool = True) -> Tuple[float, List]:
        """Run one episode and collect transitions.

        Args:
            use_residual: Whether to use the current residual policy.

        Returns:
            (mean_formation_error, list_of_transitions)
        """
        ctrl = ResidualPolicyController(self.network, device=self.device) if use_residual else None
        self.engine.set_residual_controller(ctrl)
        self.engine.reset()

        transitions = []
        total_error = 0.0

        for _ in range(self.max_steps):
            leader = self.engine.uavs[0]
            followers = self.engine.uavs[1:]
            offsets = self.engine.offsets

            # Build states before stepping
            states = [
                self._build_state(f.position, f.velocity, leader.position, leader.velocity)
                for f in followers
            ]

            info = self.engine.step()
            total_error += info["mean_formation_error"]

            # Compute supervised target: the residual should correct the pos error
            new_leader = self.engine.uavs[0]
            for i, follower in enumerate(self.engine.uavs[1:]):
                desired = new_leader.position + offsets[i]
                pos_error = desired - follower.position
                # Target residual ~ proportional to remaining position error
                target_residual = np.clip(
                    pos_error,
                    -self.config["residual_policy"]["a_residual_max"],
                    self.config["residual_policy"]["a_residual_max"],
                ).astype(np.float32)

                transitions.append((states[i], target_residual, info["formation_errors"][i]))

        mean_error = total_error / max(self.max_steps, 1)
        return mean_error, transitions

    def _update(self) -> float:
        """Perform one gradient update step from the replay buffer.

        Returns:
            Loss value.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, targets, _, _ = self.buffer.sample(self.batch_size)
        x = torch.tensor(states, device=self.device)
        y = torch.tensor(targets, device=self.device)

        pred = self.network(x)
        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            Dictionary with 'episode_errors' and 'episode_losses' lists.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.network.train()

        for episode in range(self.n_episodes):
            mean_error, transitions = self._collect_episode(use_residual=True)

            # Push transitions into buffer (state, target, reward, next_state placeholder)
            for state, target, err in transitions:
                self.buffer.push(state, target, -err, state)

            loss = self._update()

            self.episode_errors.append(mean_error)
            self.episode_losses.append(loss)

            if (episode + 1) % self.eval_interval == 0:
                print(
                    f"Episode {episode + 1}/{self.n_episodes} | "
                    f"Mean formation error: {mean_error:.4f} m | "
                    f"Loss: {loss:.6f}"
                )

            if (episode + 1) % self.save_interval == 0:
                self.save(os.path.join(self.checkpoint_dir, f"residual_policy_ep{episode + 1}.pt"))

        return {
            "episode_errors": self.episode_errors,
            "episode_losses": self.episode_losses,
        }

    def get_controller(self) -> ResidualPolicyController:
        """Return a ResidualPolicyController wrapping the trained network."""
        self.network.eval()
        return ResidualPolicyController(self.network, device=self.device)

    def save(self, path: str) -> None:
        """Save the network weights to a file.

        Args:
            path: Path to save the .pt file.
        """
        torch.save(self.network.state_dict(), path)
        print(f"Saved residual policy to {path}")

    def load(self, path: str) -> None:
        """Load network weights from a file.

        Args:
            path: Path to the .pt file.
        """
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded residual policy from {path}")
