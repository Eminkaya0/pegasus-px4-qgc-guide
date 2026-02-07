"""
| File: standalone_camera.py
| Description: Capture frames from a camera attached to the drone using replicator.
|              Creates a camera prim and render product for that specific camera.
"""
import numpy as np
import time
import sys
import os

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

def log(msg):
    print(msg)
    sys.stdout.flush()


class StandaloneCamera:
    """
    Creates a camera at the specified prim path and captures frames using replicator.
    This ensures we get the drone's POV, not the main viewport.
    """

    def __init__(self, prim_path, resolution=(640, 480), parent_prim=None, local_position=(0.3, 0.0, 0.0), local_rotation=(0.0, 0.0, 180.0)):
        """
        Args:
            prim_path: Full USD path for the camera (e.g., "/World/quadrotor1/body/yolo_cam")
            resolution: (width, height) tuple
            parent_prim: Parent prim path (for reference, camera is created at prim_path)
            local_position: Position relative to parent (x=forward, y=left, z=up)
            local_rotation: Euler angles in degrees (ZYX order)
        """
        self.prim_path = prim_path
        self.width, self.height = resolution
        self.local_position = local_position
        self.local_rotation = local_rotation

        self._camera = None
        self._render_product = None
        self._annotator = None
        self._initialized = False
        self._frame_count = 0
        self._warmup_frames = 0

    def initialize(self):
        """Create camera prim and set up replicator capture."""
        try:
            from isaacsim.sensors.camera.camera import Camera
            from scipy.spatial.transform import Rotation
            import omni.replicator.core as rep

            log(f"[StandaloneCamera] Creating camera at: {self.prim_path}")

            # Create the camera prim
            self._camera = Camera(
                prim_path=self.prim_path,
                frequency=30,
                resolution=(self.width, self.height),
            )

            # Set local pose (position and orientation relative to parent)
            quat = Rotation.from_euler("ZYX", self.local_rotation, degrees=True).as_quat()
            self._camera.set_local_pose(
                np.array(self.local_position),
                quat  # [x, y, z, w]
            )

            # Initialize the camera
            self._camera.initialize()
            log(f"[StandaloneCamera] Camera initialized")

            # Create render product for this camera
            self._render_product = rep.create.render_product(self.prim_path, (self.width, self.height))
            log(f"[StandaloneCamera] Render product created")

            # Create RGB annotator and attach to render product
            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._annotator.attach([self._render_product])
            log(f"[StandaloneCamera] RGB annotator attached")

            # Store rep module for orchestrator access
            self._rep = rep

            self._initialized = True
            return True

        except Exception as e:
            log(f"[StandaloneCamera] Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def capture(self):
        """Trigger frame capture (for compatibility, actual capture happens in get_frame)."""
        self._warmup_frames += 1

    def get_frame(self):
        """
        Get the latest frame from the drone's camera.
        Returns BGR numpy array or None.
        """
        if not self._initialized:
            return None

        # Need some warmup frames
        if self._warmup_frames < 10:
            return None

        # Try method 1: Isaac Camera's get_rgba() - most reliable
        try:
            rgba = self._camera.get_rgba()
            if rgba is not None and isinstance(rgba, np.ndarray) and rgba.size > 0:
                if len(rgba.shape) == 3 and rgba.shape[2] >= 3:
                    frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR) if rgba.shape[2] == 4 else cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
                    
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))

                    self._frame_count += 1
                    if self._frame_count == 1:
                        log(f"[StandaloneCamera] *** FIRST FRAME via get_rgba()! Shape: {frame.shape}")
                    return frame
        except Exception as e:
            if self._frame_count < 3:
                log(f"[StandaloneCamera] get_rgba() error: {e}")

        # Try method 2: Replicator annotator with orchestrator step
        if self._annotator is not None:
            try:
                # Step the orchestrator to ensure frame is rendered
                if hasattr(self, '_rep') and self._rep is not None:
                    try:
                        self._rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
                    except:
                        pass

                data = self._annotator.get_data()

                if data is not None and isinstance(data, np.ndarray) and data.size > 0:
                    if len(data.shape) == 3 and data.shape[2] >= 3:
                        frame = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR) if data.shape[2] == 4 else cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                        
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))

                        self._frame_count += 1
                        if self._frame_count == 1:
                            log(f"[StandaloneCamera] *** FIRST FRAME via annotator! Shape: {frame.shape}")
                        return frame
            except Exception as e:
                if self._frame_count < 3:
                    log(f"[StandaloneCamera] annotator error: {e}")

        return None

    @property
    def frame_count(self):
        return self._frame_count


class StandaloneCameraViewport:
    """
    DEPRECATED: Captures from main viewport (not drone camera).
    Use StandaloneCamera instead for drone POV.
    """

    def __init__(self, prim_path, resolution=(640, 480), **kwargs):
        log("[WARNING] StandaloneCameraViewport captures main viewport, not drone camera!")
        self.prim_path = prim_path
        self.width, self.height = resolution

        self._viewport = None
        self._initialized = False
        self._frame_count = 0
        
        self._capture_paths = ["/dev/shm/isaac_A.png", "/dev/shm/isaac_B.png"]
        self._capture_count = 0

    def initialize(self):
        try:
            from omni.kit.viewport.utility import get_active_viewport
            self._viewport = get_active_viewport()
            if self._viewport is None:
                log("[StandaloneCameraViewport] No active viewport!")
                return False

            active_cam = self._viewport.get_active_camera()
            log(f"[StandaloneCameraViewport] Viewport ready (camera: {active_cam})")
            self._initialized = True
            return True

        except Exception as e:
            log(f"[StandaloneCameraViewport] Init failed: {e}")
            return False

    def capture(self):
        if not self._initialized or self._viewport is None:
            return
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file
            write_path = self._capture_paths[self._capture_count % 2]
            capture_viewport_to_file(self._viewport, write_path)
            self._capture_count += 1
        except Exception as e:
            if self._capture_count < 5:
                log(f"[StandaloneCameraViewport] Capture error: {e}")

    def get_frame(self):
        if not self._initialized:
            return None
        
        if self._capture_count < 2:
            return None
        
        read_index = (self._capture_count - 2) % 2
        read_path = self._capture_paths[read_index]
        
        if not os.path.exists(read_path):
            return None

        try:
            if CV2_AVAILABLE:
                frame = cv2.imread(read_path, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    self._frame_count += 1
                    return frame
        except Exception as e:
            if self._frame_count < 5:
                log(f"[StandaloneCameraViewport] Read error: {e}")

        return None

    @property
    def frame_count(self):
        return self._frame_count
