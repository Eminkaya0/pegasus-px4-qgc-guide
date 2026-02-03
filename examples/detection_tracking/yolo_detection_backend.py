"""
| File: yolo_detection_backend.py
| Description: A custom Pegasus Backend that captures camera frames from MonocularCamera
|              and runs YOLOv8 inference to detect fixed-wing aircraft.
|
| Install dependencies:
|   <isaac-sim-python> -m pip install ultralytics opencv-python-headless
"""
import os
import time
import numpy as np

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[YoloDetectionBackend] WARNING: ultralytics not installed. Run:")
    print("  <isaac-sim-python> -m pip install ultralytics opencv-python-headless")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DetectionResult:
    """Stores a single detection result."""

    def __init__(self, detected=False, bbox_center_x=0.0, bbox_center_y=0.0,
                 bbox_width=0.0, bbox_height=0.0, confidence=0.0,
                 frame_width=640, frame_height=480):
        self.detected = detected
        self.bbox_center_x = bbox_center_x
        self.bbox_center_y = bbox_center_y
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height
        self.confidence = confidence
        self.frame_width = frame_width
        self.frame_height = frame_height


class YoloDetectionBackendConfig(BackendConfig):
    """Configuration for the YOLO detection backend."""

    def __init__(self):
        self.model_path = "yolov8n.pt"          # YOLOv8 nano (fast, downloads automatically)
        self.confidence_threshold = 0.25
        self.target_classes = [4]                # COCO class 4 = "aeroplane"
        self.display_window = True               # Show annotated frames via cv2.imshow
        self.save_frames = False                 # Save annotated frames to disk
        self.output_dir = "./detection_output"


class YoloDetectionBackend(Backend):
    """
    Custom Pegasus Backend that:
    1. Receives camera frames via update_graphical_sensor()
    2. Runs YOLOv8 inference on each frame
    3. Stores detection results for the tracking controller to read
    4. Optionally displays annotated frames in a window
    """

    def __init__(self, config: YoloDetectionBackendConfig = YoloDetectionBackendConfig()):
        super().__init__(config)

        self._model = None
        self._latest_detection = DetectionResult()
        self._latest_frame = None
        self._frame_count = 0
        self._detection_count = 0
        self._vehicle_state = None
        self._start_time = None

    @property
    def latest_detection(self) -> DetectionResult:
        """The most recent detection result. Read by the tracking controller."""
        return self._latest_detection

    @property
    def latest_frame(self) -> np.ndarray:
        """The most recent annotated camera frame (BGR)."""
        return self._latest_frame

    # --- Backend abstract method implementations ---

    def start(self):
        """Called when simulation starts. Load the YOLO model."""
        self._start_time = time.time()

        if YOLO_AVAILABLE:
            print(f"[YoloDetectionBackend] Loading YOLO model: {self.config.model_path}")
            self._model = YOLO(self.config.model_path)
            print("[YoloDetectionBackend] Model loaded successfully")
        else:
            print("[YoloDetectionBackend] YOLO not available, running in passthrough mode")

        if self.config.save_frames:
            os.makedirs(self.config.output_dir, exist_ok=True)

    def stop(self):
        """Called when simulation stops."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        print(f"[YoloDetectionBackend] Stopped. Processed {self._frame_count} frames, "
              f"{self._detection_count} detections in {elapsed:.1f}s")

        if CV2_AVAILABLE and self.config.display_window:
            cv2.destroyAllWindows()

    def reset(self):
        """Called when simulation resets."""
        self._latest_detection = DetectionResult()
        self._latest_frame = None
        self._frame_count = 0
        self._detection_count = 0

    def update_state(self, state):
        """Receive vehicle state each physics step."""
        self._vehicle_state = state

    def update_sensor(self, sensor_type: str, data):
        """Receive regular sensor data (IMU, GPS, etc.) - not used here."""
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """
        Receive graphical sensor data. This is where camera frames arrive.
        Runs YOLO inference and stores the result.
        """
        if sensor_type != "MonocularCamera":
            return

        # Get the camera object from the data dict
        camera = data.get("camera")
        if camera is None:
            return

        # Get RGB image as numpy array (RGBA, H x W x 4)
        try:
            rgba_frame = camera.get_rgb()
        except Exception:
            return

        if rgba_frame is None or rgba_frame.size == 0:
            return

        frame_height, frame_width = rgba_frame.shape[:2]

        # Convert RGBA to BGR for OpenCV/YOLO
        if CV2_AVAILABLE:
            frame_bgr = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)
        else:
            # Fallback: just strip alpha channel
            frame_bgr = rgba_frame[:, :, :3]

        self._frame_count += 1

        # Run YOLO inference
        if self._model is not None:
            results = self._model(frame_bgr, verbose=False, conf=self.config.confidence_threshold)
            self._process_detections(results, frame_bgr, frame_width, frame_height)
        else:
            # No model loaded, just mark as no detection
            self._latest_detection = DetectionResult(
                frame_width=frame_width, frame_height=frame_height
            )
            self._latest_frame = frame_bgr

        # Display annotated frame
        if CV2_AVAILABLE and self.config.display_window and self._latest_frame is not None:
            cv2.imshow("YOLO Detection", self._latest_frame)
            cv2.waitKey(1)

        # Save frame to disk
        if self.config.save_frames and CV2_AVAILABLE and self._latest_frame is not None:
            path = os.path.join(self.config.output_dir, f"frame_{self._frame_count:06d}.jpg")
            cv2.imwrite(path, self._latest_frame)

    def input_reference(self):
        """
        This backend does NOT control rotors. PX4MavlinkBackend handles that.
        Must be at index 0 in the backends list.
        """
        return []

    def update(self, dt: float):
        """Called every physics step. Nothing to do here."""
        pass

    # --- Internal methods ---

    def _process_detections(self, results, frame_bgr, frame_width, frame_height):
        """Parse YOLO results and update latest_detection."""
        best_detection = None
        best_confidence = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                # Filter for target classes (aeroplane = 4 in COCO)
                if cls_id not in self.config.target_classes:
                    continue

                if confidence > best_confidence:
                    # Get bounding box in xyxy format
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    bbox_center_x = (x1 + x2) / 2.0
                    bbox_center_y = (y1 + y2) / 2.0
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    best_detection = DetectionResult(
                        detected=True,
                        bbox_center_x=bbox_center_x,
                        bbox_center_y=bbox_center_y,
                        bbox_width=bbox_width,
                        bbox_height=bbox_height,
                        confidence=confidence,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    best_confidence = confidence

        if best_detection is not None:
            self._latest_detection = best_detection
            self._detection_count += 1
        else:
            self._latest_detection = DetectionResult(
                frame_width=frame_width, frame_height=frame_height
            )

        # Annotate frame
        annotated = frame_bgr.copy()
        if CV2_AVAILABLE and best_detection is not None:
            d = best_detection
            x1 = int(d.bbox_center_x - d.bbox_width / 2)
            y1 = int(d.bbox_center_y - d.bbox_height / 2)
            x2 = int(d.bbox_center_x + d.bbox_width / 2)
            y2 = int(d.bbox_center_y + d.bbox_height / 2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"aircraft {d.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw crosshair at frame center
        if CV2_AVAILABLE:
            cx, cy = frame_width // 2, frame_height // 2
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 1)

        # Draw detection status
        if CV2_AVAILABLE:
            status = "TRACKING" if (best_detection and best_detection.detected) else "SEARCHING"
            color = (0, 255, 0) if status == "TRACKING" else (0, 0, 255)
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        self._latest_frame = annotated
