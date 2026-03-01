"""Logging and visualization utilities for multi-UAV simulation."""

import os
import csv
import json
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class SimulationLogger:
    """Logs UAV positions, velocities, and control inputs during simulation.

    Data can be exported to CSV / JSON and visualised as 2-D or 3-D plots.
    """

    def __init__(self, output_dir: str = "logs", log_interval: int = 1):
        """Initialise the logger.

        Args:
            output_dir: Directory where log files and figures are saved.
            log_interval: Record every N steps.
        """
        self.output_dir = output_dir
        self.log_interval = log_interval
        os.makedirs(output_dir, exist_ok=True)

        self._records: List[Dict[str, Any]] = []

    def log_step(self, step_info: Dict[str, Any]) -> None:
        """Record information from a single simulation step.

        Args:
            step_info: Dictionary returned by ``SimulationEngine.step()``.
        """
        if step_info["step"] % self.log_interval != 0:
            return

        record = {
            "step": int(step_info["step"]),
            "time": float(step_info["time"]),
            "leader_pos": step_info["leader_pos"].tolist(),
            "leader_vel": step_info["leader_vel"].tolist(),
            "follower_positions": step_info["follower_positions"].tolist(),
            "follower_velocities": step_info["follower_velocities"].tolist(),
            "base_accels": step_info["base_accels"].tolist(),
            "residual_accels": step_info["residual_accels"].tolist(),
            "total_accels": step_info["total_accels"].tolist(),
            "formation_errors": step_info["formation_errors"].tolist(),
            "mean_formation_error": float(step_info["mean_formation_error"]),
        }
        self._records.append(record)

    def log_all(self, log: List[Dict[str, Any]]) -> None:
        """Log all steps from a completed simulation run.

        Args:
            log: List of step-info dicts returned by ``SimulationEngine.run()``.
        """
        for step_info in log:
            self.log_step(step_info)

    def save_csv(self, filename: str = "simulation_log.csv") -> str:
        """Save logged data to a CSV file (scalar columns only).

        Args:
            filename: Output filename (placed inside ``output_dir``).

        Returns:
            Full path to the saved file.
        """
        if not self._records:
            return ""

        path = os.path.join(self.output_dir, filename)
        n_followers = len(self._records[0]["formation_errors"])
        fieldnames = ["step", "time", "mean_formation_error"] + [
            f"formation_error_follower_{i}" for i in range(n_followers)
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._records:
                row: Dict[str, Any] = {
                    "step": rec["step"],
                    "time": rec["time"],
                    "mean_formation_error": rec["mean_formation_error"],
                }
                for i, err in enumerate(rec["formation_errors"]):
                    row[f"formation_error_follower_{i}"] = err
                writer.writerow(row)

        return path

    def save_json(self, filename: str = "simulation_log.json") -> str:
        """Save the full logged data to a JSON file.

        Args:
            filename: Output filename (placed inside ``output_dir``).

        Returns:
            Full path to the saved file.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(self._records, f)
        return path

    def plot_formation_errors(
        self,
        filename: str = "formation_errors.png",
        title: str = "Formation Errors Over Time",
    ) -> str:
        """Plot per-follower formation errors over time.

        Args:
            filename: Output image filename.
            title: Plot title.

        Returns:
            Full path to the saved image.
        """
        if not self._records:
            return ""

        times = [r["time"] for r in self._records]
        n_followers = len(self._records[0]["formation_errors"])
        errors = np.array([r["formation_errors"] for r in self._records])

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(n_followers):
            ax.plot(times, errors[:, i], label=f"Follower {i + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Formation Error (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_trajectories_2d(
        self,
        filename: str = "trajectories_2d.png",
        title: str = "UAV Trajectories (XY Plane)",
    ) -> str:
        """Plot 2-D (XY) trajectories for all UAVs.

        Args:
            filename: Output image filename.
            title: Plot title.

        Returns:
            Full path to the saved image.
        """
        if not self._records:
            return ""

        leader_xs = [r["leader_pos"][0] for r in self._records]
        leader_ys = [r["leader_pos"][1] for r in self._records]

        n_followers = len(self._records[0]["follower_positions"])
        follower_xs = [
            [r["follower_positions"][i][0] for r in self._records]
            for i in range(n_followers)
        ]
        follower_ys = [
            [r["follower_positions"][i][1] for r in self._records]
            for i in range(n_followers)
        ]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(leader_xs, leader_ys, "k-", linewidth=2, label="Leader")
        for i in range(n_followers):
            ax.plot(follower_xs[i], follower_ys[i], label=f"Follower {i + 1}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_trajectories_3d(
        self,
        filename: str = "trajectories_3d.png",
        title: str = "UAV Trajectories (3D)",
    ) -> str:
        """Plot 3-D trajectories for all UAVs.

        Args:
            filename: Output image filename.
            title: Plot title.

        Returns:
            Full path to the saved image.
        """
        if not self._records:
            return ""

        leader_pos = np.array([r["leader_pos"] for r in self._records])
        n_followers = len(self._records[0]["follower_positions"])
        follower_pos = [
            np.array([r["follower_positions"][i] for r in self._records])
            for i in range(n_followers)
        ]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(leader_pos[:, 0], leader_pos[:, 1], leader_pos[:, 2], "k-", linewidth=2, label="Leader")
        for i, fp in enumerate(follower_pos):
            ax.plot(fp[:, 0], fp[:, 1], fp[:, 2], label=f"Follower {i + 1}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_control_inputs(
        self,
        filename: str = "control_inputs.png",
        title: str = "Control Inputs Over Time",
    ) -> str:
        """Plot the norm of base and residual accelerations over time.

        Args:
            filename: Output image filename.
            title: Plot title.

        Returns:
            Full path to the saved image.
        """
        if not self._records:
            return ""

        times = [r["time"] for r in self._records]
        n_followers = len(self._records[0]["base_accels"])
        base_norms = np.array([
            [np.linalg.norm(r["base_accels"][i]) for i in range(n_followers)]
            for r in self._records
        ])
        res_norms = np.array([
            [np.linalg.norm(r["residual_accels"][i]) for i in range(n_followers)]
            for r in self._records
        ])

        fig, axes = plt.subplots(n_followers, 1, figsize=(10, 3 * n_followers), sharex=True)
        if n_followers == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(times, base_norms[:, i], label="Base PD", linewidth=1.5)
            ax.plot(times, res_norms[:, i], label="Residual", linewidth=1.5, linestyle="--")
            ax.set_ylabel(f"|a| Follower {i + 1} (m/s²)")
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def clear(self) -> None:
        """Clear all recorded data."""
        self._records = []
