"""
| File: fixed_wing_target.py
| Description: A scripted fixed-wing aircraft that flies along a predefined waypoint path.
|              This is a kinematic object (no physics simulation) - it simply moves along
|              the path at a constant speed, with orientation following the velocity vector.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# Isaac Sim imports
from pxr import UsdGeom, Gf
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


class FixedWingTarget:
    """
    A scripted fixed-wing target that follows a looping waypoint path.
    No physics simulation - purely kinematic movement.
    """

    def __init__(self, stage_prefix: str, usd_path: str, waypoints: list, speed: float = 15.0):
        """
        Args:
            stage_prefix: USD stage path (e.g., "/World/fixed_wing")
            usd_path: Path to the fixed-wing USD model file
            waypoints: List of np.array([x, y, z]) positions forming the flight path
            speed: Flight speed in m/s (default 15.0)
        """
        self._stage_prefix = stage_prefix
        self._waypoints = waypoints
        self._speed = speed
        self._current_waypoint_idx = 0
        self._position = np.array(waypoints[0], dtype=np.float64)
        self._world = PegasusInterface().world

        # Spawn the USD prim
        self._prim = define_prim(self._stage_prefix, "Xform")
        self._prim = get_prim_at_path(self._stage_prefix)
        self._prim.GetReferences().AddReference(usd_path)

        # Get the xformable interface for setting transforms
        self._xformable = UsdGeom.Xformable(self._prim)

        # Set initial position
        self._update_transform()

        # Register physics callback to update position each step
        self._world.add_physics_callback(
            self._stage_prefix + "/update",
            self._update
        )

    @property
    def position(self) -> np.ndarray:
        """Current position of the fixed-wing target."""
        return self._position.copy()

    def _update(self, dt):
        """Physics callback: move the target along the waypoint path."""
        # dt is passed directly as a float, not as event.payload

        # Get current target waypoint
        target_wp = np.array(self._waypoints[self._current_waypoint_idx], dtype=np.float64)

        # Direction and distance to next waypoint
        direction = target_wp - self._position
        distance = np.linalg.norm(direction)

        if distance < 1.0:
            # Reached waypoint, advance to next (loop around)
            self._current_waypoint_idx = (self._current_waypoint_idx + 1) % len(self._waypoints)
            target_wp = np.array(self._waypoints[self._current_waypoint_idx], dtype=np.float64)
            direction = target_wp - self._position
            distance = np.linalg.norm(direction)

        if distance > 0:
            # Normalize direction and move
            direction_normalized = direction / distance
            step = self._speed * dt
            self._position += direction_normalized * min(step, distance)

        # Update the USD prim transform
        self._update_transform()

    def _update_transform(self):
        """Update the USD prim position and orientation."""
        # Compute orientation: nose points toward next waypoint
        target_wp = np.array(self._waypoints[self._current_waypoint_idx], dtype=np.float64)
        direction = target_wp - self._position
        dist = np.linalg.norm(direction)

        if dist > 0.01:
            # Compute yaw (rotation around Z) and pitch (rotation around Y)
            direction_normalized = direction / dist
            yaw = np.arctan2(direction_normalized[1], direction_normalized[0])
            pitch = np.arctan2(-direction_normalized[2],
                               np.sqrt(direction_normalized[0]**2 + direction_normalized[1]**2))
            rotation = Rotation.from_euler("ZYX", [yaw, pitch, 0.0])
            quat = rotation.as_quat()  # [x, y, z, w]
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0])

        # Clear existing xform ops and set new ones
        self._xformable.ClearXformOpOrder()
        translate_op = self._xformable.AddTranslateOp()
        orient_op = self._xformable.AddOrientOp()

        translate_op.Set(Gf.Vec3d(
            float(self._position[0]),
            float(self._position[1]),
            float(self._position[2])
        ))

        # USD uses (w, x, y, z) quaternion order - use Quatf for single precision
        orient_op.Set(Gf.Quatf(
            float(quat[3]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2])
        ))
