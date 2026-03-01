"""Simulation engine for multi-UAV residual policy control."""

import numpy as np
from typing import Dict, List, Optional, Any

from models.uav import UAV
from controllers.base_controller import BaseLeaderFollowerController
from controllers.residual_controller import ResidualPolicyController


def _make_circle_waypoint(t: float, radius: float, speed: float, altitude: float) -> np.ndarray:
    """Return leader position on a circular trajectory at time t."""
    omega = speed / radius
    return np.array([
        radius * np.cos(omega * t),
        radius * np.sin(omega * t),
        altitude,
    ])


def _make_circle_velocity(t: float, radius: float, speed: float, altitude: float) -> np.ndarray:
    """Return leader velocity on a circular trajectory at time t."""
    omega = speed / radius
    return np.array([
        -radius * omega * np.sin(omega * t),
        radius * omega * np.cos(omega * t),
        0.0,
    ])


def _make_lemniscate_waypoint(t: float, radius: float, speed: float, altitude: float) -> np.ndarray:
    """Return leader position on a lemniscate (figure-8) trajectory."""
    scale = speed
    denom = 1.0 + np.sin(scale * t) ** 2
    return np.array([
        radius * np.cos(scale * t) / denom,
        radius * np.sin(scale * t) * np.cos(scale * t) / denom,
        altitude,
    ])


class SimulationEngine:
    """Runs the multi-UAV leader-follower simulation.

    The engine manages all UAV states, applies control inputs (base + residual),
    advances the simulation by one time step, and optionally injects disturbances.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the simulation engine from a configuration dict.

        Args:
            config: Configuration dictionary loaded from ``config.yaml``.
        """
        self.config = config
        sim_cfg = config["simulation"]
        uav_cfg = config["uav"]
        leader_cfg = config["leader"]
        form_cfg = config["formation"]
        ctrl_cfg = config["base_controller"]
        dist_cfg = config.get("disturbance", {"enabled": False})

        self.dt = sim_cfg["dt"]
        self.max_steps = sim_cfg["max_steps"]
        self.n_uavs = sim_cfg["n_uavs"]
        self.n_followers = self.n_uavs - 1

        # Formation offsets: shape (n_followers, 3)
        self.offsets = np.array(form_cfg["offsets"][: self.n_followers], dtype=np.float64)

        # Leader trajectory parameters
        self.traj_type = leader_cfg["trajectory"]
        self.traj_radius = leader_cfg["radius"]
        self.traj_speed = leader_cfg["speed"]
        self.traj_altitude = leader_cfg["altitude"]

        # Disturbance settings
        self.disturbance_enabled = dist_cfg.get("enabled", False)
        self.wind_magnitude = dist_cfg.get("wind_magnitude", 0.0)
        self.wind_direction = np.array(
            dist_cfg.get("wind_direction", [1.0, 0.0, 0.0]), dtype=np.float64
        )

        # Base controller
        self.base_ctrl = BaseLeaderFollowerController(
            k_p=ctrl_cfg["k_p"],
            k_d=ctrl_cfg["k_d"],
            a_max=ctrl_cfg["a_max"],
        )

        # Residual controller (set after training or loaded from file)
        self.residual_ctrl: Optional[ResidualPolicyController] = None

        # Create UAVs
        rng = np.random.default_rng(sim_cfg.get("seed", 0))
        self.uavs: List[UAV] = self._init_uavs(uav_cfg, rng)

        # Simulation state
        self.step_count = 0
        self.time = 0.0

        # Step log (populated during run)
        self.log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_uavs(self, uav_cfg: Dict, rng: np.random.Generator) -> List[UAV]:
        """Create and return all UAV objects."""
        leader_pos = _make_circle_waypoint(0.0, self.traj_radius, self.traj_speed, self.traj_altitude)
        leader_vel = _make_circle_velocity(0.0, self.traj_radius, self.traj_speed, self.traj_altitude)

        uavs = [
            UAV(
                uav_id=0,
                position=leader_pos.copy(),
                velocity=leader_vel.copy(),
                mass=uav_cfg["mass"],
                v_max=uav_cfg["v_max"],
                a_max=uav_cfg["a_max"],
                drag=uav_cfg["drag"],
            )
        ]

        for i in range(self.n_followers):
            init_pos = leader_pos + self.offsets[i] + rng.uniform(-0.1, 0.1, size=3)
            init_vel = leader_vel + rng.uniform(-0.05, 0.05, size=3)
            uavs.append(
                UAV(
                    uav_id=i + 1,
                    position=init_pos,
                    velocity=init_vel,
                    mass=uav_cfg["mass"],
                    v_max=uav_cfg["v_max"],
                    a_max=uav_cfg["a_max"],
                    drag=uav_cfg["drag"],
                )
            )

        return uavs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_residual_controller(self, controller: ResidualPolicyController) -> None:
        """Attach a residual policy controller to the engine.

        Args:
            controller: Trained residual policy controller.
        """
        self.residual_ctrl = controller

    def reset(self) -> None:
        """Reset the simulation to the initial state."""
        sim_cfg = self.config["simulation"]
        uav_cfg = self.config["uav"]

        rng = np.random.default_rng(sim_cfg.get("seed", 0))
        self.uavs = self._init_uavs(uav_cfg, rng)
        self.step_count = 0
        self.time = 0.0
        self.log = []

    def step(self) -> Dict[str, Any]:
        """Advance the simulation by one time step.

        Returns:
            A dictionary with positions, velocities, accelerations, and
            formation errors for the current step.
        """
        leader = self.uavs[0]
        followers = self.uavs[1:]

        # Update leader: follow reference trajectory
        t = self.time
        if self.traj_type == "circle":
            leader_ref_pos = _make_circle_waypoint(t, self.traj_radius, self.traj_speed, self.traj_altitude)
            leader_ref_vel = _make_circle_velocity(t, self.traj_radius, self.traj_speed, self.traj_altitude)
        else:
            leader_ref_pos = _make_lemniscate_waypoint(t, self.traj_radius, self.traj_speed, self.traj_altitude)
            leader_ref_vel = np.zeros(3)

        # Simple proportional tracking for leader
        leader_accel = 5.0 * (leader_ref_pos - leader.position) + 2.0 * (leader_ref_vel - leader.velocity)
        leader.step(leader_accel, self.dt)

        # Collect follower states
        follower_positions = np.array([f.position for f in followers])
        follower_velocities = np.array([f.velocity for f in followers])

        # Base PD commands
        base_accels = self.base_ctrl.compute_batch(
            follower_positions,
            follower_velocities,
            leader.position,
            leader.velocity,
            self.offsets,
        )

        # Residual corrections
        if self.residual_ctrl is not None:
            residuals = self.residual_ctrl.compute_batch(
                follower_positions,
                follower_velocities,
                leader.position,
                leader.velocity,
            )
        else:
            residuals = np.zeros_like(base_accels)

        # Optional wind disturbance
        if self.disturbance_enabled:
            wind = self.wind_magnitude * self.wind_direction
        else:
            wind = np.zeros(3)

        # Apply combined commands to followers
        total_accels = base_accels + residuals + wind

        for i, follower in enumerate(followers):
            follower.step(total_accels[i], self.dt)

        # Compute formation errors
        desired_positions = leader.position + self.offsets
        formation_errors = np.linalg.norm(
            np.array([f.position for f in followers]) - desired_positions, axis=1
        )

        step_info = {
            "step": self.step_count,
            "time": self.time,
            "leader_pos": leader.position.copy(),
            "leader_vel": leader.velocity.copy(),
            "follower_positions": np.array([f.position for f in followers]).copy(),
            "follower_velocities": np.array([f.velocity for f in followers]).copy(),
            "base_accels": base_accels.copy(),
            "residual_accels": residuals.copy(),
            "total_accels": total_accels.copy(),
            "formation_errors": formation_errors.copy(),
            "mean_formation_error": float(formation_errors.mean()),
        }

        self.log.append(step_info)
        self.step_count += 1
        self.time += self.dt

        return step_info

    def run(self) -> List[Dict[str, Any]]:
        """Run the full simulation for ``max_steps`` steps.

        Returns:
            List of per-step info dictionaries.
        """
        self.reset()
        for _ in range(self.max_steps):
            self.step()
        return self.log

    def is_done(self) -> bool:
        """Return True if the simulation has reached max_steps."""
        return self.step_count >= self.max_steps
