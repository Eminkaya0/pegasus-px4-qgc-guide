#!/usr/bin/env python
"""
| File: detection_tracking_scenario.py
| Description: Main scenario script - a camera-equipped quadrotor detects and actively
|              tracks a fixed-wing aircraft using YOLOv8 in NVIDIA Isaac Sim.
|
| Architecture:
|   Fixed-Wing Target (scripted path) → appears in camera FOV
|   Observer Drone (Iris + Camera)    → captures frames at 15 FPS
|   YoloDetectionBackend              → runs YOLOv8, produces bounding boxes
|   TrackingController                → visual servoing via MAVLink offboard mode
|   PX4 SITL                          → flight controller
|   QGroundControl                    → monitoring on UDP 18570
|
| Usage:
|   1. Clean stale PX4:  pkill -9 px4; rm -f /tmp/px4_lock-* /tmp/px4-sock-*
|   2. Run:  <isaac-sim-python> detection_tracking_scenario.py
|   3. Open QGroundControl, connect to UDP 18570
|   4. The drone will take off, then search and track the fixed-wing
"""
import carb
from isaacsim import SimulationApp

# Start Isaac Sim (must be first)
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# Imports after SimulationApp init
# -----------------------------------
import os
import sys
import time
import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

# Pegasus imports
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_detection_backend import YoloDetectionBackend, YoloDetectionBackendConfig
from fixed_wing_target import FixedWingTarget
from tracking_controller import TrackingController


# ============================================================
# CONFIGURATION - Edit these paths for your setup
# ============================================================

# Path to your fixed-wing USD model
# You can create a simple one in Isaac Sim: Create > Mesh > Cube, scale it to look like a plane,
# then File > Export to save as USD. Or import a free 3D model.
FIXED_WING_USD = os.path.join(os.path.dirname(__file__), "assets", "fixed_wing.usd")

# PX4 Autopilot directory
PX4_DIR = PegasusInterface().px4_path  # Uses path from Pegasus configs.yaml


class DetectionTrackingApp:
    """
    Main application: spawns an observer drone with camera + YOLO detection,
    a fixed-wing target on a scripted path, and a tracking controller.
    """

    def __init__(self):

        # Acquire the timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # Start Pegasus Interface
        self.pg = PegasusInterface()

        # Create the World
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Load environment (synchronous - do NOT use load_environment)
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # --- 1. Setup YOLO detection backend ---
        yolo_config = YoloDetectionBackendConfig()
        yolo_config.display_window = True           # Show live detection window
        yolo_config.confidence_threshold = 0.25
        yolo_config.target_classes = [4]            # COCO: 4 = aeroplane
        self.yolo_backend = YoloDetectionBackend(yolo_config)

        # --- 2. Setup camera ---
        camera = MonocularCamera("tracking_camera", config={
            "frequency": 15,                        # 15 FPS for YOLO inference
            "resolution": (640, 480),               # YOLO-friendly resolution
            "position": np.array([0.30, 0.0, 0.0]), # Forward-facing on body
            "orientation": np.array([0.0, 0.0, 0.0]),
            "clipping_range": (0.1, 200.0),         # Extended to 200m for long-range detection
            "depth": False,                         # Not needed for detection
        })

        # --- 3. Setup PX4 backend ---
        px4_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "connection_type": "tcpin",
            "connection_ip": "localhost",
            "connection_baseport": 4560,
            "px4_autolaunch": True,
            "px4_dir": PX4_DIR,
            "px4_vehicle_model": "gazebo-classic_iris",
            "enable_lockstep": True,
            "num_rotors": 4,
            "input_offset": [0.0, 0.0, 0.0, 0.0],
            "input_scaling": [1000.0, 1000.0, 1000.0, 1000.0],
            "zero_position_armed": [100.0, 100.0, 100.0, 100.0],
            "update_rate": 250.0,
        })

        # --- 4. Create observer drone ---
        # IMPORTANT: PX4MavlinkBackend must be at index 0 (it controls the rotors)
        config_multirotor = MultirotorConfig()
        config_multirotor.graphical_sensors = [camera]
        config_multirotor.backends = [
            PX4MavlinkBackend(px4_config),          # Index 0: rotor control
            self.yolo_backend,                       # Index 1: detection only
        ]

        Multirotor(
            "/World/observer",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # --- 5. Create fixed-wing target ---
        # Racetrack pattern at ~30m altitude, ~80m radius
        waypoints = [
            np.array([80.0, 0.0, 30.0]),
            np.array([80.0, 80.0, 32.0]),
            np.array([-80.0, 80.0, 30.0]),
            np.array([-80.0, 0.0, 28.0]),
            np.array([0.0, -40.0, 30.0]),
        ]

        if os.path.exists(FIXED_WING_USD):
            self.target = FixedWingTarget(
                "/World/fixed_wing",
                FIXED_WING_USD,
                waypoints=waypoints,
                speed=15.0,             # 15 m/s (~54 km/h)
            )
            print("[Scenario] Fixed-wing target spawned")
        else:
            self.target = None
            print(f"[Scenario] WARNING: Fixed-wing USD not found at {FIXED_WING_USD}")
            print("[Scenario] Place your fixed-wing USD model in the assets/ folder")
            print("[Scenario] Running without target - use QGC to fly manually")

        # --- 6. Setup tracking controller ---
        self.tracker = TrackingController(
            self.yolo_backend,
            mavlink_port=14540,             # PX4 offboard port (14540 + vehicle_id)
            takeoff_altitude=30.0,          # Same altitude as fixed-wing
        )

        # Reset the simulation
        self.world.reset()

        # Track whether we've started the tracking sequence
        self._tracking_started = False
        self._startup_time = None

    def run(self):
        """Main simulation loop."""

        # Start the simulation
        self.timeline.play()

        self._startup_time = time.time()

        # The "infinite" loop
        while simulation_app.is_running():

            # Update the simulation
            self.world.step(render=True)

            # After 10 seconds of warmup (camera init, PX4 connection),
            # start the tracking controller
            elapsed = time.time() - self._startup_time
            if not self._tracking_started and elapsed > 10.0:
                print("[Scenario] Starting tracking controller...")
                try:
                    self.tracker.start()
                    self.tracker.arm_and_takeoff()
                    self._tracking_started = True
                    print("[Scenario] Tracking controller active")
                except Exception as e:
                    print(f"[Scenario] Failed to start tracker: {e}")
                    print("[Scenario] You can still fly manually via QGroundControl")
                    self._tracking_started = True  # Don't retry

            # Update tracking controller
            if self._tracking_started:
                try:
                    self.tracker.update()
                except Exception:
                    pass

        # Cleanup
        carb.log_warn("DetectionTrackingApp is closing.")
        self.tracker.stop()
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = DetectionTrackingApp()
    pg_app.run()


if __name__ == "__main__":
    main()
