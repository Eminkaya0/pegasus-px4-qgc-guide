#!/usr/bin/env python
"""
| File: 5_detection_tracking.py
| Description: Drone with camera + fixed-wing target + YOLO detection.
|              Uses MonocularCamera sensor and YoloDetectionBackend (proper Pegasus pattern).
|
| Usage:
|   cd /home/emin/PegasusSimulator/examples
|   pkill -9 px4; rm -f /tmp/px4_lock-* /tmp/px4-sock-*
|   ISAACSIM_PYTHON 5_detection_tracking.py
|
| Then:
|   - Open QGroundControl → connect to UDP 18570
|   - In another terminal: python3 yolo_viewer.py
"""
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
import numpy as np
import os
import sys
import time

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera

from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from fixed_wing_target import FixedWingTarget
from yolo_detection_backend import YoloDetectionBackend, YoloDetectionBackendConfig

FIXED_WING_USD = os.path.join(os.path.dirname(__file__), "utils", "fixed_wing.usda")
YOLO_MODEL = os.path.join(os.path.dirname(__file__), "yolov8_avci.pt")


def log(msg):
    print(msg)
    sys.stdout.flush()


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # --- YOLO detection backend ---
        yolo_config = YoloDetectionBackendConfig()
        yolo_config.model_path = YOLO_MODEL if os.path.exists(YOLO_MODEL) else "yolov8n.pt"
        yolo_config.display_window = False  # Use yolo_viewer.py instead
        yolo_config.confidence_threshold = 0.25
        yolo_config.target_classes = [0]  # 0 = uav in avci model
        yolo_config.save_frames = False

        self.yolo_backend = YoloDetectionBackend(yolo_config)
        log(f"[App] YOLO backend configured with model: {yolo_config.model_path}")

        # --- Camera sensor (proper Pegasus pattern) ---
        # Camera is forward-facing, slightly tilted down
        camera = MonocularCamera("tracking_camera", config={
            "frequency": 15,
            "resolution": (640, 480),
            "position": np.array([0.15, 0.0, -0.05]),  # forward, down
            "orientation": np.array([-15.0, 0.0, 180.0]),  # tilt down, face forward
            "clipping_range": (0.1, 500.0),
            "depth": False,
        })
        log("[App] Camera configured: 640x480 @ 15 FPS")

        # --- PX4 backend ---
        px4_backend_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "connection_type": "tcpin",
            "connection_ip": "localhost",
            "connection_baseport": 4560,
            "px4_autolaunch": True,
            "px4_dir": "/home/emin/PX4-Autopilot",
            "px4_vehicle_model": "gazebo-classic_iris",
            "enable_lockstep": True,
            "num_rotors": 4,
            "input_offset": [0.0, 0.0, 0.0, 0.0],
            "input_scaling": [1000.0, 1000.0, 1000.0, 1000.0],
            "zero_position_armed": [100.0, 100.0, 100.0, 100.0],
            "update_rate": 250.0
        })

        # --- Create multirotor with camera and backends ---
        config_multirotor = MultirotorConfig()
        config_multirotor.graphical_sensors = [camera]
        config_multirotor.backends = [
            PX4MavlinkBackend(px4_backend_config),  # Must be first (controls rotors)
            self.yolo_backend,                       # Detection only
        ]

        Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )
        log("[App] Multirotor created with camera and YOLO backend")

        # --- Fixed-wing target ---
        waypoints = [
            np.array([80.0, 0.0, 30.0]),
            np.array([80.0, 80.0, 32.0]),
            np.array([-80.0, 80.0, 30.0]),
            np.array([-80.0, 0.0, 28.0]),
            np.array([0.0, -40.0, 30.0]),
        ]

        if os.path.exists(FIXED_WING_USD):
            self.target = FixedWingTarget(
                "/World/fixed_wing", FIXED_WING_USD,
                waypoints=waypoints, speed=15.0,
            )
            log("[App] Fixed-wing target spawned")
        else:
            self.target = None
            log(f"[App] WARNING: No fixed-wing USD at {FIXED_WING_USD}")

        self.world.reset()

    def run(self):
        self.timeline.play()
        log("[App] Startup warmup (100 steps)...")

        for i in range(100):
            if not simulation_app.is_running():
                return
            self.world.step(render=True)
            if i % 20 == 0:
                log(f"[App] Warmup step {i}/100")

        # Warm restart for PX4
        log("[App] Warm restart...")
        self.timeline.stop()
        time.sleep(1.5)
        self.timeline.play()
        log("[App] Timeline resumed")

        # Additional warmup after restart (camera needs ~100 frames)
        log("[App] Camera warmup (100 frames)...")
        for i in range(100):
            if not simulation_app.is_running():
                return
            self.world.step(render=True)
            if i % 30 == 0:
                log(f"[App] Camera warmup {i}/100")

        log("[App] ======================================")
        log("[App] Simulation running!")
        log("[App] - QGroundControl: UDP 18570")
        log("[App] - YOLO viewer: python3 yolo_viewer.py")
        log("[App] - Fly the drone to chase the fixed-wing!")
        log("[App] ======================================")

        step_count = 0
        while simulation_app.is_running():
            self.world.step(render=True)
            step_count += 1

            # Periodic status
            if step_count % 500 == 0:
                det = self.yolo_backend.latest_detection
                status = "TRACKING" if det.detected else "SEARCHING"
                log(f"[App] Step {step_count} - {status}")

        # Cleanup
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
