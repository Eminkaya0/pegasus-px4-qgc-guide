"""
| File: tracking_controller.py
| Description: Visual servoing tracking controller that reads YOLO detection results
|              and sends velocity commands to PX4 via MAVLink offboard mode.
|
| The controller:
|   - Centers the detected target in the camera frame using yaw and vertical velocity
|   - Maintains forward velocity to approach/follow the target
|   - Enters search mode (slow yaw rotation) when target is lost
"""
import time
import math
import threading

from pymavlink import mavutil


class TrackingController:
    """
    Visual servoing controller that tracks a detected target by sending
    velocity commands to PX4 SITL via MAVLink offboard mode.
    """

    def __init__(self, yolo_backend, mavlink_port=14540, takeoff_altitude=30.0):
        """
        Args:
            yolo_backend: YoloDetectionBackend instance to read detections from
            mavlink_port: PX4 offboard MAVLink port (14540 + vehicle_id)
            takeoff_altitude: Target altitude in meters (NED: negative = up)
        """
        self._yolo_backend = yolo_backend
        self._takeoff_altitude = takeoff_altitude
        self._mavlink_port = mavlink_port

        # Visual servoing gains
        self._kp_yaw = 0.3          # Proportional gain for yaw (deg/s per pixel error)
        self._kp_vz = 0.005         # Proportional gain for vertical velocity (m/s per pixel error)
        self._forward_speed = 5.0   # Forward speed when tracking (m/s)
        self._search_yaw_rate = 15.0  # Yaw rate during search mode (deg/s)

        # State
        self._connection = None
        self._armed = False
        self._offboard = False
        self._running = False
        self._last_detection_time = 0.0
        self._search_timeout = 3.0  # Switch to search mode after N seconds without detection

        # Heartbeat thread
        self._hb_thread = None

    def start(self):
        """Connect to PX4 and start the heartbeat thread."""
        connection_string = f"udpin:localhost:{self._mavlink_port}"
        print(f"[TrackingController] Connecting to PX4 at {connection_string}")

        self._connection = mavutil.mavlink_connection(connection_string)
        print("[TrackingController] Waiting for PX4 heartbeat...")
        self._connection.wait_heartbeat()
        print(f"[TrackingController] Heartbeat received (system {self._connection.target_system}, "
              f"component {self._connection.target_component})")

        self._running = True

        # Start heartbeat thread (PX4 needs heartbeats to stay in offboard)
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def stop(self):
        """Stop the controller and close MAVLink connection."""
        self._running = False
        if self._connection:
            self._connection.close()
        print("[TrackingController] Stopped")

    def arm_and_takeoff(self):
        """Arm the vehicle and take off to the configured altitude."""
        if not self._connection:
            return

        # Set offboard mode
        self._set_offboard_mode()

        # Send a few setpoints before arming (PX4 requirement)
        for _ in range(20):
            self._send_position_ned(0, 0, -self._takeoff_altitude, 0)
            time.sleep(0.05)

        # Arm the vehicle
        self._arm()

        # Command takeoff position
        print(f"[TrackingController] Taking off to {self._takeoff_altitude}m")
        self._send_position_ned(0, 0, -self._takeoff_altitude, 0)

    def update(self):
        """
        Main update loop. Call this every iteration from the simulation loop.
        Reads detection results and sends appropriate velocity commands.
        """
        if not self._connection or not self._running:
            return

        detection = self._yolo_backend.latest_detection
        now = time.time()

        if detection.detected:
            self._last_detection_time = now
            self._track_target(detection)
        else:
            # Check if we should enter search mode
            time_since_last = now - self._last_detection_time
            if time_since_last > self._search_timeout:
                self._search_mode()
            else:
                # Brief loss - hold position
                self._send_velocity_ned(0, 0, 0, 0)

    def _track_target(self, detection):
        """Send velocity commands to center the target in the camera frame."""
        # Compute pixel error from frame center
        error_x = detection.bbox_center_x - detection.frame_width / 2.0
        error_y = detection.bbox_center_y - detection.frame_height / 2.0

        # Yaw rate to center target horizontally (positive error = target is right = yaw right)
        yaw_rate = self._kp_yaw * error_x

        # Clamp yaw rate
        yaw_rate = max(-45.0, min(45.0, yaw_rate))

        # Vertical velocity to center target vertically
        # Positive error_y = target below center = need to descend (positive Vz in NED)
        vz = self._kp_vz * error_y

        # Forward velocity (body-frame X)
        # Reduce speed when target is large (close) to maintain distance
        bbox_area_ratio = (detection.bbox_width * detection.bbox_height) / \
                          (detection.frame_width * detection.frame_height)

        if bbox_area_ratio > 0.15:
            # Target is close, slow down
            vx = 1.0
        elif bbox_area_ratio > 0.05:
            # Target at medium distance
            vx = self._forward_speed * 0.5
        else:
            # Target is far, full speed
            vx = self._forward_speed

        # Send velocity command in body frame
        self._send_velocity_body(vx, 0, vz, yaw_rate)

    def _search_mode(self):
        """Slowly rotate to search for the target."""
        # Yaw rotation only, hold altitude
        self._send_velocity_ned(0, 0, 0, self._search_yaw_rate)

    # --- MAVLink command helpers ---

    def _heartbeat_loop(self):
        """Send heartbeats to keep MAVLink alive."""
        while self._running:
            if self._connection:
                self._connection.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
            time.sleep(0.5)

    def _set_offboard_mode(self):
        """Switch PX4 to offboard mode."""
        self._connection.mav.command_long_send(
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,                              # confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6,                              # PX4 offboard mode = 6
            0, 0, 0, 0, 0
        )
        self._offboard = True
        print("[TrackingController] Offboard mode set")

    def _arm(self):
        """Arm the vehicle."""
        self._connection.mav.command_long_send(
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,                              # 1 = arm
            0, 0, 0, 0, 0, 0
        )
        self._armed = True
        print("[TrackingController] Vehicle armed")

    def _send_position_ned(self, x, y, z, yaw_deg):
        """Send position setpoint in NED frame."""
        self._connection.mav.set_position_target_local_ned_send(
            0,                              # time_boot_ms
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,             # type_mask: position only
            x, y, z,                        # position NED (m)
            0, 0, 0,                        # velocity (ignored)
            0, 0, 0,                        # acceleration (ignored)
            math.radians(yaw_deg), 0        # yaw, yaw_rate
        )

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate_deg):
        """Send velocity setpoint in NED frame with yaw rate."""
        self._connection.mav.set_position_target_local_ned_send(
            0,                              # time_boot_ms
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000011111000111,             # type_mask: velocity + yaw_rate only
            0, 0, 0,                        # position (ignored)
            vx, vy, vz,                     # velocity NED (m/s)
            0, 0, 0,                        # acceleration (ignored)
            0, math.radians(yaw_rate_deg)   # yaw (ignored), yaw_rate (rad/s)
        )

    def _send_velocity_body(self, vx, vy, vz, yaw_rate_deg):
        """Send velocity setpoint in body frame with yaw rate."""
        self._connection.mav.set_position_target_local_ned_send(
            0,                              # time_boot_ms
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000011111000111,             # type_mask: velocity + yaw_rate only
            0, 0, 0,                        # position (ignored)
            vx, vy, vz,                     # velocity body frame (m/s)
            0, 0, 0,                        # acceleration (ignored)
            0, math.radians(yaw_rate_deg)   # yaw (ignored), yaw_rate (rad/s)
        )
