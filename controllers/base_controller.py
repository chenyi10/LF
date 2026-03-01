"""Base PD leader-follower controller."""

import numpy as np


class BaseLeaderFollowerController:
    """PD controller that computes acceleration commands for follower UAVs.

    Each follower is commanded to maintain a desired offset from the leader
    using a proportional-derivative control law:

        a = k_p * (p_desired - p_follower) + k_d * (v_leader - v_follower)

    where ``p_desired = p_leader + offset``.
    """

    def __init__(
        self,
        k_p: float = 1.0,
        k_d: float = 0.5,
        a_max: float = 3.0,
    ):
        """Initialize the PD controller.

        Args:
            k_p: Proportional gain on position error.
            k_d: Derivative gain on follower velocity.
            a_max: Maximum magnitude of the commanded acceleration (m/s^2).
        """
        self.k_p = k_p
        self.k_d = k_d
        self.a_max = a_max

    def compute(
        self,
        follower_pos: np.ndarray,
        follower_vel: np.ndarray,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
        offset: np.ndarray,
    ) -> np.ndarray:
        """Compute PD acceleration command for a single follower.

        Args:
            follower_pos: Follower position, shape (3,).
            follower_vel: Follower velocity, shape (3,).
            leader_pos: Leader position, shape (3,).
            leader_vel: Leader velocity, shape (3,).
            offset: Desired offset of follower relative to leader, shape (3,).

        Returns:
            Commanded acceleration, shape (3,).
        """
        desired_pos = leader_pos + offset
        pos_error = desired_pos - follower_pos
        accel = self.k_p * pos_error + self.k_d * (leader_vel - follower_vel)

        accel_norm = np.linalg.norm(accel)
        if accel_norm > self.a_max:
            accel = accel * (self.a_max / accel_norm)

        return accel

    def compute_batch(
        self,
        follower_positions: np.ndarray,
        follower_velocities: np.ndarray,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        """Compute PD acceleration commands for all followers.

        Args:
            follower_positions: Follower positions, shape (N, 3).
            follower_velocities: Follower velocities, shape (N, 3).
            leader_pos: Leader position, shape (3,).
            leader_vel: Leader velocity, shape (3,).
            offsets: Desired offsets for each follower, shape (N, 3).

        Returns:
            Commanded accelerations, shape (N, 3).
        """
        n = len(follower_positions)
        accels = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            accels[i] = self.compute(
                follower_positions[i],
                follower_velocities[i],
                leader_pos,
                leader_vel,
                offsets[i],
            )
        return accels
