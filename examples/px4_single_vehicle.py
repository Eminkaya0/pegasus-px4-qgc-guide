#!/usr/bin/env python
"""
| File: px4_single_vehicle.py
| Description: Complete working example for running a single PX4-controlled drone
|              in Isaac Sim with Pegasus Simulator, viewable in QGroundControl.
|
| Usage:
|   1. Kill stale PX4: pkill -9 px4; rm -f /tmp/px4_lock-* /tmp/px4-sock-*
|   2. Run: <isaac-sim-python> px4_single_vehicle.py
|   3. Connect QGroundControl to UDP port 18570
"""
# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

# Use pathlib for parsing the desired trajectory from a CSV file
from pathlib import Path
import os


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Load the environment synchronously using load_asset (NOT load_environment which is async
        # and will invalidate the World object)
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

        # Get the current directory used to read trajectories and save results
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        # Create the vehicle
        config_multirotor = MultirotorConfig()

        # PX4 backend configuration
        # IMPORTANT: PX4MavlinkBackendConfig takes a DICTIONARY, not keyword arguments
        px4_backend_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "connection_type": "tcpin",              # Pegasus = TCP server, PX4 connects as client
            "connection_ip": "localhost",
            "connection_baseport": 4560,             # PX4 SITL default simulator port
            "px4_autolaunch": True,                  # Auto-start PX4 SITL binary
            "px4_dir": PegasusInterface().px4_path,  # Uses path from configs.yaml
            "px4_vehicle_model": "gazebo-classic_iris",
            "enable_lockstep": True,                 # Synchronize simulation with PX4
            "num_rotors": 4,
            "input_offset": [0.0, 0.0, 0.0, 0.0],
            "input_scaling": [1000.0, 1000.0, 1000.0, 1000.0],
            "zero_position_armed": [100.0, 100.0, 100.0, 100.0],
            "update_rate": 250.0                     # Hz
        })

        # IMPORTANT: Wrap config in PX4MavlinkBackend - backends list expects Backend objects, not Config objects
        config_multirotor.backends = [PX4MavlinkBackend(px4_backend_config)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running():

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()


if __name__ == "__main__":
    main()
