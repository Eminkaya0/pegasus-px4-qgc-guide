# Pegasus Simulator + PX4 SITL + QGroundControl Integration Guide

A complete guide for connecting **NVIDIA Isaac Sim** (with Pegasus Simulator) to **PX4 SITL** and monitoring your drone in **QGroundControl**.

## 🎬 Demo Videos

### Real-time UAV Detection with YOLOv8 in NVIDIA Isaac Sim

[![YOLOv8 UAV Detection Demo](https://img.youtube.com/vi/AUYeP_fm134/maxresdefault.jpg)](https://youtu.be/AUYeP_fm134)

> **Click the image to watch the full demo on YouTube**

Demonstration of real-time UAV detection and tracking using YOLOv8 custom-trained model in NVIDIA Isaac Sim with Pegasus Simulator. Features camera-equipped quadrotor detecting fixed-wing aircraft, with PX4 SITL integration and QGroundControl monitoring.

**Features demonstrated:**
- 🎯 Real-time UAV detection using YOLOv8 custom model
- 📷 Onboard camera mounted on drone body in Isaac Sim
- ✈️ Fixed-wing target tracking scenario
- 🎮 PX4 SITL + QGroundControl integration
- 📊 Live detection feed with bounding boxes

This guide documents the correct configuration that took significant debugging to figure out, since the official documentation doesn't clearly explain the connection architecture between these three systems.

## Architecture Overview

```
┌─────────────────────────┐       TCP        ┌──────────────┐       UDP        ┌──────────────────┐
│   NVIDIA Isaac Sim      │  localhost:4560   │   PX4 SITL   │  localhost:18570 │  QGroundControl  │
│   + Pegasus Simulator   │◄────────────────►│              │◄────────────────►│                  │
│                         │                   │              │                  │                  │
│  Simulates physics,     │  Sensor data ──►  │  Runs flight │  Telemetry ──►   │  Displays HUD,   │
│  drone dynamics,        │  ◄── Motor cmds   │  controller  │  ◄── Commands    │  map, controls   │
│  sensors, environment   │                   │  firmware     │                  │                  │
└─────────────────────────┘                   └──────────────┘                  └──────────────────┘
     TCP Server (port 4560)                    TCP Client + UDP Server            UDP Client
```

**Key insight:** QGroundControl does NOT connect directly to Pegasus/Isaac Sim. PX4 SITL acts as the bridge:
1. **Pegasus <-> PX4:** TCP connection on port `4560` (Pegasus is the server, PX4 connects as client)
2. **PX4 <-> QGC:** UDP connection on port `18570` (PX4 broadcasts telemetry, QGC receives it)

## Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) | 4.x / 5.x | Physics simulation engine |
| [Pegasus Simulator](https://github.com/PegasusResearch/PegasusSimulator) | 5.1.0+ | Drone simulation extension for Isaac Sim |
| [PX4-Autopilot](https://github.com/PX4/PX4-Autopilot) | v1.14+ | Flight controller firmware (SITL) |
| [QGroundControl](http://qgroundcontrol.com/) | Latest | Ground control station |
| Ubuntu | 22.04 / 24.04 | Operating system |

## Step 1: Install and Build PX4 SITL

```bash
# Clone PX4-Autopilot
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot

# Install dependencies
bash ./Tools/setup/ubuntu.sh

# Build SITL (this is required - Pegasus auto-launches this binary)
make px4_sitl_default

# Verify the binary exists
ls build/px4_sitl_default/bin/px4
```

## Step 2: Install Pegasus Simulator

Follow the [official Pegasus installation guide](https://pegasusresearch.github.io/PegasusSimulator/). Make sure to configure the PX4 path in Pegasus:

**File:** `PegasusSimulator/extensions/pegasus.simulator/config/configs.yaml`
```yaml
px4_dir: /path/to/your/PX4-Autopilot
```

## Step 3: Configure the Python Script

The critical part is the `PX4MavlinkBackendConfig`. Here are the correct parameters:

```python
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

px4_backend_config = PX4MavlinkBackendConfig({
    "vehicle_id": 0,
    "connection_type": "tcpin",          # TCP server mode - Pegasus listens, PX4 connects
    "connection_ip": "localhost",
    "connection_baseport": 4560,         # Default PX4 SITL simulator port
    "px4_autolaunch": True,              # Auto-start PX4 SITL binary
    "px4_dir": "/path/to/PX4-Autopilot",
    "px4_vehicle_model": "gazebo-classic_iris",
    "enable_lockstep": True,             # Sync simulation with PX4 steps
    "num_rotors": 4,
    "input_offset": [0.0, 0.0, 0.0, 0.0],
    "input_scaling": [1000.0, 1000.0, 1000.0, 1000.0],
    "zero_position_armed": [100.0, 100.0, 100.0, 100.0],
    "update_rate": 250.0
})

# IMPORTANT: Wrap the config in PX4MavlinkBackend, not just the config object
config_multirotor.backends = [PX4MavlinkBackend(px4_backend_config)]
```

### Common Mistakes to Avoid

| Mistake | Why It Fails | Correct Value |
|---------|-------------|---------------|
| `PX4MavlinkBackendConfig(connection_type="tcpin", ...)` | Class takes a **dict**, not keyword args | `PX4MavlinkBackendConfig({...})` |
| `backends = [px4_backend_config]` | Backends list needs **Backend** objects, not **Config** objects | `backends = [PX4MavlinkBackend(config)]` |
| `"connection_type": "udpin"` | PX4 SITL connects via **TCP**, not UDP | `"connection_type": "tcpin"` |
| `"connection_baseport": 14550` | Port 14550 is for QGC, not Pegasus<->PX4 | `"connection_baseport": 4560` |
| `"px4_vehicle_model": "iris"` | PX4 SITL needs the full model name | `"px4_vehicle_model": "gazebo-classic_iris"` |
| `pg.load_environment(...)` | Async - invalidates World object | `pg.load_asset(..., "/World/layout")` |

See [examples/px4_single_vehicle.py](examples/px4_single_vehicle.py) for a complete working script.

## Step 4: Connect QGroundControl

This is the part that most guides miss. PX4 SITL does **NOT** broadcast telemetry on port 14550. Instead, it uses:

```
GCS port = 18570 + vehicle_id
```

For vehicle_id=0, the telemetry port is **18570**.

### QGroundControl Setup

1. Open **QGroundControl**
2. Click the **Q icon** (top-left) -> **Application Settings**
3. Go to **Comm Links**
4. Click **Add** to create a new connection:
   - **Name:** `PX4 SITL`
   - **Type:** `UDP`
   - **Port:** `18570`
   - Click **Add Server**: `127.0.0.1`
5. Click **OK**, then **Connect**

### Demo Video

[![Pegasus + PX4 + QGroundControl Setup Demo](https://img.youtube.com/vi/y8IqwwldfJA/maxresdefault.jpg)](https://youtu.be/y8IqwwldfJA)

> Click the image above to watch the full setup walkthrough on YouTube.

After connecting, you should see the drone appear on the QGC map with full telemetry (attitude, GPS position, battery, etc.).

### Multi-Vehicle Port Mapping

| Vehicle ID | Pegasus<->PX4 Port (TCP) | PX4<->QGC Port (UDP) |
|-----------|--------------------------|----------------------|
| 0         | 4560                     | 18570                |
| 1         | 4561                     | 18571                |
| 2         | 4562                     | 18572                |
| N         | 4560 + N                 | 18570 + N            |

## Step 5: Run the Simulation

```bash
# 1. Kill any stale PX4 processes (important!)
pkill -9 px4 2>/dev/null; rm -f /tmp/px4_lock-* /tmp/px4-sock-*

# 2. Run the example script with Isaac Sim's Python
cd /path/to/PegasusSimulator
~/.local/share/ov/pkg/isaac-sim-*/python.sh examples/px4_single_vehicle.py

# 3. Open QGroundControl and connect to UDP 18570
```

### Expected Output

When everything works correctly, you should see in the terminal:

```
[Warning] Waiting for first hearbeat
[Warning] Received first hearbeat
INFO  [mavlink] mode: Normal, data rate: 4000000 B/s on udp port 18570 remote port 14550
INFO  [commander] Ready for takeoff!
```

## Troubleshooting

### "PX4 server already running for instance 0"

A previous PX4 SITL process is still running. Clean it up:

```bash
pkill -9 px4
rm -f /tmp/px4_lock-* /tmp/px4-sock-*
```

### "Waiting for first heartbeat" never resolves

1. Check that PX4 SITL binary exists:
   ```bash
   ls /path/to/PX4-Autopilot/build/px4_sitl_default/bin/px4
   ```
   If not, build it: `cd PX4-Autopilot && make px4_sitl_default`

2. Verify connection settings use `tcpin` on port `4560`

3. Check no other process is using port 4560:
   ```bash
   ss -tlnp | grep 4560
   ```

### QGroundControl shows "Disconnected"

1. Verify PX4 SITL is running and past the heartbeat stage
2. Check you're connecting to port **18570**, not 14550
3. Verify the port is open:
   ```bash
   ss -ulnp | grep 18570
   ```

### World/Scene crashes on load

Use `load_asset()` instead of `load_environment()`:

```python
# Wrong - async, causes World invalidation
self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

# Correct - synchronous loading
self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")
```

### ROS2 Bridge errors

These warnings can be safely ignored if you're not using ROS2:
```
[Error] [isaacsim.ros2.bridge.impl.extension] ROS2 Bridge startup failed
```

## Port Architecture Reference

```
Pegasus Simulator (Isaac Sim)
├── TCP Server: localhost:4560  ──────────┐
│   Sends: HIL_SENSOR, HIL_GPS           │
│   Receives: HIL_ACTUATOR_CONTROLS      │
│                                         ▼
PX4 SITL ◄───────────────────────────────┘
├── TCP Client: localhost:4560 (to Pegasus)
├── UDP GCS: localhost:18570 ─────────────┐
│   Sends: HEARTBEAT, ATTITUDE,           │
│          GLOBAL_POSITION_INT, etc.       │
├── UDP Offboard: localhost:14580         │
├── UDP Onboard: localhost:14280          │
│                                         ▼
QGroundControl ◄──────────────────────────┘
└── UDP: localhost:18570 (connect here!)
```

---

## Advanced: Fixed-Wing Detection & Tracking Scenario

An advanced scenario where a camera-equipped quadrotor detects and actively tracks a fixed-wing aircraft using YOLOv8.

### Scenario Architecture

```
Fixed-Wing Target (scripted racetrack path at 30m altitude)
    ↓ appears in camera FOV (50-100m range)
Observer Drone (Iris + MonocularCamera at 15 FPS)
    ↓ camera frames (640x480)
YoloDetectionBackend (YOLOv8 nano inference)
    ↓ bounding box + confidence
TrackingController (visual servoing via MAVLink offboard)
    ↓ velocity commands
PX4 SITL → QGroundControl (monitoring on UDP 18570)
```

### Components

| File | Purpose |
|------|---------|
| `detection_tracking_scenario.py` | Main entry point - wires everything together |
| `yolo_detection_backend.py` | Custom Pegasus Backend: captures camera frames, runs YOLOv8 |
| `fixed_wing_target.py` | Scripted fixed-wing that flies a looping waypoint path |
| `tracking_controller.py` | Visual servoing controller via MAVLink offboard mode |

### Setup

```bash
# 1. Install Python dependencies in Isaac Sim's environment
~/.local/share/ov/pkg/isaac-sim-*/python.sh -m pip install -r examples/detection_tracking/requirements.txt

# 2. Provide a fixed-wing USD model
#    Place your model at: examples/detection_tracking/assets/fixed_wing.usd
#    Options:
#    - Create in Isaac Sim: Create > Mesh > Cube, scale to plane shape, export as USD
#    - Import a free 3D model (.obj/.fbx) via Isaac Sim's File > Import
#    - Download from Sketchfab/TurboSquid (free models available)

# 3. Kill stale PX4 processes
pkill -9 px4 2>/dev/null; rm -f /tmp/px4_lock-* /tmp/px4-sock-*

# 4. Run the scenario
cd /path/to/pegasus-px4-qgc-guide
~/.local/share/ov/pkg/isaac-sim-*/python.sh examples/detection_tracking/detection_tracking_scenario.py
```

### How It Works

1. **Observer drone** spawns with a forward-facing camera (640x480, 15 FPS, 200m clipping range)
2. **Fixed-wing target** flies a racetrack pattern at ~30m altitude, 15 m/s
3. **YOLOv8** runs on each camera frame, looking for COCO class "aeroplane" (class 4)
4. A live **detection window** shows annotated frames with bounding boxes and "TRACKING"/"SEARCHING" status
5. After 10s warmup, the **tracking controller** connects to PX4 via MAVLink offboard:
   - Arms and takes off to 30m
   - When target detected: adjusts yaw/altitude/speed to keep it centered in frame
   - When target lost: enters search mode (slow yaw rotation)
6. Monitor everything in **QGroundControl** on UDP 18570

### Detection Notes

- YOLOv8 nano (COCO-trained) has an "aeroplane" class but may not reliably detect 3D-rendered aircraft at distance
- For better results, consider fine-tuning YOLOv8 on synthetic data rendered from Isaac Sim
- The `target_classes` config can be extended to include other COCO classes (e.g., bird=14) if using different target models
- Detection window shows a red crosshair at frame center and green bounding boxes around detections

### Tracking Controller Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kp_yaw` | 0.3 | Proportional gain for yaw tracking (deg/s per pixel) |
| `kp_vz` | 0.005 | Proportional gain for vertical tracking (m/s per pixel) |
| `forward_speed` | 5.0 | Forward speed when tracking (m/s) |
| `search_yaw_rate` | 15.0 | Yaw rate during search mode (deg/s) |
| `search_timeout` | 3.0 | Seconds without detection before entering search mode |

## License

This guide is provided as-is for educational purposes. The example scripts are based on [Pegasus Simulator](https://github.com/PegasusResearch/PegasusSimulator) (BSD-3-Clause).
