"""
| File: frame_streamer.py
| Description: Shared memory frame streamer for passing annotated YOLO frames
|              from Isaac Sim (headless OpenCV) to an external viewer window.
|
| Usage:
|   Inside Isaac Sim: FrameStreamer writes frames to shared memory
|   External viewer:  Reads and displays with full OpenCV (GTK support)
"""
import struct
import numpy as np

try:
    from multiprocessing import shared_memory
    SHM_AVAILABLE = True
except ImportError:
    SHM_AVAILABLE = False

# Shared memory layout:
# [4 bytes: width][4 bytes: height][8 bytes: frame_id][8 bytes: write_seq][data: H*W*3 BGR]
HEADER_SIZE = 24
HEADER_FORMAT = '<IIqq'  # width, height, frame_id, write_seq


class FrameStreamer:
    """Writes annotated frames to shared memory for external display."""

    def __init__(self, name="yolo_detection_feed", width=640, height=480):
        self.name = name
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.total_size = HEADER_SIZE + self.frame_size
        self.shm = None
        self.frame_id = 0

        if not SHM_AVAILABLE:
            print("[FrameStreamer] shared_memory not available")
            return

        # Clean up any stale segment
        try:
            old = shared_memory.SharedMemory(name=self.name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(
            name=self.name, create=True, size=self.total_size
        )
        print(f"[FrameStreamer] Created /dev/shm/{self.name} ({self.total_size} bytes)")

    def write_frame(self, bgr_frame: np.ndarray):
        """Write a BGR frame to shared memory."""
        if self.shm is None:
            return

        h, w = bgr_frame.shape[:2]
        if h != self.height or w != self.width:
            # Resize if needed
            import cv2
            bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

        self.frame_id += 1

        # Write header with odd seq (writing)
        header = struct.pack(HEADER_FORMAT, self.width, self.height, self.frame_id, self.frame_id * 2 - 1)
        self.shm.buf[:HEADER_SIZE] = header

        # Write frame data
        self.shm.buf[HEADER_SIZE:HEADER_SIZE + self.frame_size] = bgr_frame.tobytes()

        # Write header with even seq (complete)
        header = struct.pack(HEADER_FORMAT, self.width, self.height, self.frame_id, self.frame_id * 2)
        self.shm.buf[:HEADER_SIZE] = header

    def close(self):
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self.shm = None


class FrameReader:
    """Reads frames from shared memory for display."""

    def __init__(self, name="yolo_detection_feed"):
        self.name = name
        self.shm = None
        self.last_frame_id = 0

        if not SHM_AVAILABLE:
            raise RuntimeError("shared_memory not available")

        self.shm = shared_memory.SharedMemory(name=self.name)
        print(f"[FrameReader] Attached to /dev/shm/{self.name}")

    def read_frame(self):
        """Read a frame. Returns (frame_bgr, new_frame) or (None, False)."""
        if self.shm is None:
            return None, False

        # Read header
        header = struct.unpack(HEADER_FORMAT, bytes(self.shm.buf[:HEADER_SIZE]))
        width, height, frame_id, write_seq = header

        # Check if write is complete (even seq) and new frame
        if write_seq % 2 != 0:
            return None, False  # Write in progress

        if frame_id == self.last_frame_id:
            return None, False  # Same frame

        # Read frame data
        frame_size = width * height * 3
        data = bytes(self.shm.buf[HEADER_SIZE:HEADER_SIZE + frame_size])

        # Verify write didn't happen during read
        header2 = struct.unpack(HEADER_FORMAT, bytes(self.shm.buf[:HEADER_SIZE]))
        if header2[3] != write_seq:
            return None, False  # Torn read

        self.last_frame_id = frame_id
        frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        return frame, True

    def close(self):
        if self.shm:
            try:
                self.shm.close()
            except Exception:
                pass
            self.shm = None
