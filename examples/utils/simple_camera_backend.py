"""
| File: simple_camera_backend.py
| Description: Simple backend that captures camera frames and displays them with OpenCV.
|              This is the simplest way to access Isaac Sim camera data in Pegasus.
"""
import cv2
import numpy as np
import sys
from pegasus.simulator.logic.backends.backend import Backend, BackendConfig


def log(msg):
    """Print with immediate flush to ensure output is visible."""
    print(msg)
    sys.stdout.flush()


class SimpleCameraBackendConfig(BackendConfig):
    """Configuration for SimpleCameraBackend."""

    def __init__(self):
        super().__init__()
        self.window_name = "Camera View"


class SimpleCameraBackend(Backend):
    """
    Simple backend that captures camera frames from MonocularCamera
    and displays them in an OpenCV window.
    """

    def __init__(self, config=None):
        # Create default config if not provided
        if config is None:
            config = SimpleCameraBackendConfig()

        super().__init__(config)

        self._window_name = config.window_name
        self._frame_count = 0
        self._latest_frame = None
        self._callback_count = 0  # Track total callbacks

    def start(self):
        """Called when simulation starts."""
        log(f"[SimpleCameraBackend] Started - window: {self._window_name}")
        log(f"[SimpleCameraBackend] Waiting for camera warmup (~7 seconds at 15 FPS)...")

    def stop(self):
        """Called when simulation stops."""
        # Note: Don't call cv2.destroyAllWindows() - Isaac Sim's OpenCV doesn't have GTK support
        log(f"[SimpleCameraBackend] Stopped - captured {self._frame_count} frames from {self._callback_count} callbacks")

    def update_graphical_sensor(self, sensor_type: str, data: dict):
        """
        Called when graphical sensor (camera) has new data.

        Args:
            sensor_type: Type of sensor (e.g., "MonocularCamera")
            data: Dictionary containing sensor data
                  - data["camera"]: The camera object
                  - Call camera.get_rgba() to get RGBA numpy array
        """
        self._callback_count += 1

        # Log first few callbacks to verify data flow
        if self._callback_count <= 5:
            log(f"[CAMERA] Callback #{self._callback_count}: sensor_type={sensor_type}, data_keys={list(data.keys()) if data else None}")

        if sensor_type != "MonocularCamera":
            return

        # Get camera object from data
        camera = data.get("camera")
        if camera is None:
            if self._callback_count <= 10:
                log("[CAMERA] ERROR: camera object is None in data dict")
            return

        # Get RGBA image (returns RGBA numpy array or None if not ready)
        # Note: The method is get_rgba(), NOT get_rgb_image()
        try:
            rgba_image = camera.get_rgba()
        except Exception as e:
            log(f"[CAMERA] ERROR: get_rgba() raised exception: {e}")
            return

        # Check if image is None or empty
        if rgba_image is None:
            if self._callback_count <= 10 or self._callback_count % 50 == 0:
                log(f"[CAMERA] Callback #{self._callback_count}: get_rgba() returned None (warming up)")
            return

        # Check if image has valid shape (should be H x W x 4 for RGBA)
        if not hasattr(rgba_image, 'shape') or len(rgba_image.shape) != 3 or rgba_image.shape[2] != 4:
            if self._callback_count <= 10 or self._callback_count % 50 == 0:
                shape_info = rgba_image.shape if hasattr(rgba_image, 'shape') else 'no shape'
                log(f"[CAMERA] Callback #{self._callback_count}: Invalid image shape: {shape_info} (warming up)")
            return

        # Check if image is empty (size 0)
        if rgba_image.size == 0:
            if self._callback_count <= 10 or self._callback_count % 50 == 0:
                log(f"[CAMERA] Callback #{self._callback_count}: Empty image (size=0, warming up)")
            return

        # Success! Got a valid image
        self._frame_count += 1

        # Log first successful frame
        if self._frame_count == 1:
            log(f"[CAMERA] *** FIRST FRAME RECEIVED! Shape: {rgba_image.shape}, dtype: {rgba_image.dtype}")

        # Convert RGBA to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)

        # Store latest frame (can be accessed from outside)
        self._latest_frame = bgr_image

        # Save first 5 frames and then every 30th frame
        if self._frame_count <= 5 or self._frame_count % 30 == 0:
            save_path = f"/tmp/drone_camera_{self._frame_count:05d}.png"
            cv2.imwrite(save_path, bgr_image)
            log(f"[CAMERA] Frame #{self._frame_count} saved to: {save_path}")

    @property
    def latest_frame(self):
        """Get the latest captured frame (BGR format)."""
        return self._latest_frame

    @property
    def frame_count(self):
        """Get total number of frames captured."""
        return self._frame_count

    # --- Required Backend methods (not used for camera-only backend) ---

    def update_sensor(self, sensor_type: str, data: dict):
        """Called for non-graphical sensors (IMU, GPS, etc.)."""
        pass

    def update_state(self, state: dict):
        """Called with vehicle state (position, velocity, attitude)."""
        pass

    def input_reference(self):
        """Return motor commands - not used for camera-only backend."""
        return []

    def update(self, dt: float):
        """Called every physics step."""
        pass

    def reset(self):
        """Called when simulation resets."""
        log(f"[SimpleCameraBackend] Reset called (had {self._frame_count} frames)")
        self._frame_count = 0
        self._latest_frame = None
        self._callback_count = 0
