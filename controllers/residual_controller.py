"""Residual policy controller using a neural network."""

import numpy as np
import torch
import torch.nn as nn
from typing import List


class ResidualPolicyNetwork(nn.Module):
    """Neural network that learns a residual correction for the base controller.

    The network takes as input the relative positions and velocities of the
    UAVs as well as the leader's state, and outputs a residual acceleration
    correction to be added on top of the baseline PD controller output.

    Input features for each follower:
        - relative position to leader: (3,)
        - relative velocity to leader: (3,)
        - leader velocity: (3,)
        Total input dim = 9
    Output:
        - residual acceleration correction: (3,)
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dims: List[int] = None,
        output_dim: int = 3,
        activation: str = "relu",
        a_residual_max: float = 1.0,
    ):
        """Initialize the residual policy network.

        Args:
            input_dim: Dimension of the input feature vector.
            hidden_dims: List of hidden layer widths.
            output_dim: Dimension of the output (residual acceleration).
            activation: Activation function name ('relu' or 'tanh').
            a_residual_max: Maximum magnitude of residual acceleration (m/s^2).
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.a_residual_max = a_residual_max

        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Residual acceleration tensor of shape (batch, output_dim).
        """
        out = self.net(x)
        # Scale output to [-a_residual_max, a_residual_max] via tanh
        out = torch.tanh(out) * self.a_residual_max
        return out


class ResidualPolicyController:
    """Controller that combines base PD output with a learned residual correction.

    The final acceleration command is:
        a_total = a_base + a_residual
    where ``a_residual`` is produced by the ``ResidualPolicyNetwork``.
    """

    def __init__(
        self,
        network: ResidualPolicyNetwork,
        device: str = "cpu",
    ):
        """Initialize the residual policy controller.

        Args:
            network: Trained (or untrained) residual policy network.
            device: Torch device string ('cpu' or 'cuda').
        """
        self.network = network.to(device)
        self.device = device

    def _build_input(
        self,
        follower_pos: np.ndarray,
        follower_vel: np.ndarray,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
    ) -> torch.Tensor:
        """Build the input feature vector for a single follower.

        Args:
            follower_pos: Follower position, shape (3,).
            follower_vel: Follower velocity, shape (3,).
            leader_pos: Leader position, shape (3,).
            leader_vel: Leader velocity, shape (3,).

        Returns:
            Input tensor of shape (1, 9).
        """
        rel_pos = follower_pos - leader_pos
        rel_vel = follower_vel - leader_vel
        features = np.concatenate([rel_pos, rel_vel, leader_vel]).astype(np.float32)
        return torch.tensor(features, device=self.device).unsqueeze(0)

    def compute(
        self,
        follower_pos: np.ndarray,
        follower_vel: np.ndarray,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute residual acceleration for a single follower.

        Args:
            follower_pos: Follower position, shape (3,).
            follower_vel: Follower velocity, shape (3,).
            leader_pos: Leader position, shape (3,).
            leader_vel: Leader velocity, shape (3,).

        Returns:
            Residual acceleration, shape (3,).
        """
        x = self._build_input(follower_pos, follower_vel, leader_pos, leader_vel)
        with torch.no_grad():
            residual = self.network(x).cpu().numpy().squeeze(0)
        return residual

    def compute_batch(
        self,
        follower_positions: np.ndarray,
        follower_velocities: np.ndarray,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute residual accelerations for all followers.

        Args:
            follower_positions: Follower positions, shape (N, 3).
            follower_velocities: Follower velocities, shape (N, 3).
            leader_pos: Leader position, shape (3,).
            leader_vel: Leader velocity, shape (3,).

        Returns:
            Residual accelerations, shape (N, 3).
        """
        n = len(follower_positions)
        inputs = []
        for i in range(n):
            inputs.append(
                self._build_input(
                    follower_positions[i],
                    follower_velocities[i],
                    leader_pos,
                    leader_vel,
                )
            )
        x = torch.cat(inputs, dim=0)
        with torch.no_grad():
            residuals = self.network(x).cpu().numpy()
        return residuals
