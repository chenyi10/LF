"""UAV model for multi-UAV simulation."""

import numpy as np


class UAV:
    """Simple 3D UAV model with position and velocity state.

    The UAV dynamics follow a double-integrator model:
        dv/dt = a - drag * v
        dp/dt = v
    where ``a`` is the acceleration command and ``drag`` is a drag coefficient.
    """

    def __init__(
        self,
        uav_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float = 1.0,
        v_max: float = 3.0,
        a_max: float = 5.0,
        drag: float = 0.1,
    ):
        """Initialize the UAV.

        Args:
            uav_id: Unique identifier for this UAV.
            position: Initial position as a (3,) array [x, y, z].
            velocity: Initial velocity as a (3,) array [vx, vy, vz].
            mass: UAV mass in kg.
            v_max: Maximum speed in m/s.
            a_max: Maximum acceleration in m/s^2.
            drag: Drag coefficient for velocity damping.
        """
        self.uav_id = uav_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.mass = mass
        self.v_max = v_max
        self.a_max = a_max
        self.drag = drag

        # History for logging
        self.position_history: list = [self.position.copy()]
        self.velocity_history: list = [self.velocity.copy()]
        self.accel_history: list = []

    def step(self, acceleration: np.ndarray, dt: float) -> None:
        """Update UAV state given an acceleration command and time step.

        Args:
            acceleration: Commanded acceleration as a (3,) array [ax, ay, az].
            dt: Time step in seconds.
        """
        accel = np.array(acceleration, dtype=np.float64)

        # Clip acceleration to physical limits
        accel_norm = np.linalg.norm(accel)
        if accel_norm > self.a_max:
            accel = accel * (self.a_max / accel_norm)

        # Euler integration: dv/dt = a - drag*v
        self.velocity += (accel - self.drag * self.velocity) * dt

        # Clip velocity to maximum speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.v_max:
            self.velocity = self.velocity * (self.v_max / speed)

        # Update position: dp/dt = v
        self.position += self.velocity * dt

        # Record history
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.accel_history.append(accel.copy())

    def get_state(self) -> np.ndarray:
        """Return the current state as a (6,) array [x, y, z, vx, vy, vz]."""
        return np.concatenate([self.position, self.velocity])

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Reset the UAV to the given position and velocity.

        Args:
            position: New position as a (3,) array.
            velocity: New velocity as a (3,) array.
        """
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.accel_history = []
