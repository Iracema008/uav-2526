# Stationary Landing Controller
#
# Uses MAVLink to command Pixhawk velocity
# Converts camera frame to body frame
# Implements velocity control
# Lands drone and cuts motors
# Has a takeoff function for testing 
#
# Assumptions:
# Downward facing camera
# Camera centered on drone
# Pixhawk stabilizes drone

from pymavlink import mavutil
import time

# These are proportional control gains (adjust as needed for testing) 
# Controls how aggressively we move to the tag
Kp_xy = 0.4
Kp_z  = 0.3

# Safety limit on velocity commands (adjust as we test)
MAX_VELOCITY = 0.3

# 5 cm threshold for disarming after landing
DISARM_ALTITUDE_THRESHOLD = 0.05

# Takeoff to 3 meters for testing
TAKEOFF_ALTITUDE = 3.0


class StationaryLandingController:

    def __init__(self, connection_string, baudrate):

        print("[INFO] Connecting to Pixhawk...")

        self.master = mavutil.mavlink_connection(connection_string, baud=baudrate)
        self.master.wait_heartbeat()

        print("[INFO] Pixhawk Connected")

    def test_arm_and_takeoff(self):
        """
        Arm the drone and take off to target_alt (meters)
        """

        print("[INFO] Arming vehicle...")
        self.master.arducopter_arm()
        time.sleep(2)

        print("[INFO] Setting GUIDED mode...")
        self.master.set_mode(mavutil.mavlink.MAV_MODE_GUIDED_ARMED)
        time.sleep(2)

        print(f"[INFO] Taking off to {TAKEOFF_ALTITUDE} meters...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,0,0,0,0,0,
            TAKEOFF_ALTITUDE
        )

        # Wait until altitude reached
        while True:
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            alt = msg.relative_alt / 1000.0  # mm → m
            if alt >= TAKEOFF_ALTITUDE * 0.95:
                print(f"[INFO] Target altitude reached: {alt:.2f} m")
                break
            time.sleep(0.2)

    def send_velocity(self, vx, vy, vz):
        # Send the velocity command to the Pixhawk using MAVLink
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    def convert_camera_to_body_frame(self, cam_x, cam_y, cam_z):
        """
        Convert camera frame to the body frame
        """
        # Camera frame to body frame conversion
        #
        # Downward camera: (Got this from chat, we need to check this with actual testing))
        # body_x = -cam_y
        # body_y =  cam_x
        # body_z =  cam_z

        body_x = -cam_y
        body_y = cam_x
        body_z = cam_z

        return body_x, body_y, body_z
    
    def adjust_velocity_and_send(self, body_x, body_y, body_z):
        """
        Apply proportional control and send velocity command
        """
       # Apply proportional controls on the gain
        vx = Kp_xy * body_x
        vy = Kp_xy * body_y
        vz = Kp_z  * body_z

        # Clip velocities to max velocity
        vx = max(min(vx, MAX_VELOCITY), -MAX_VELOCITY)
        vy = max(min(vy, MAX_VELOCITY), -MAX_VELOCITY)
        vz = max(min(vz, MAX_VELOCITY), -MAX_VELOCITY)

        self.send_velocity(vx, vy, vz)

    def send_land_command(self):
        """
        Send land command to Pixhawk
        """
        print("[INFO] Landing...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0,0,0,0,0,0,0
        )
        time.sleep(1)  # give it a moment to send
    
    def land_and_disarm(self, body_z):
        """
        Land and automatically disarm once below threshold
        """
        print("[INFO] Initiating LAND command...")
        self.send_land_command()

        # Ensure we are close to the ground before cutting the motors
        while body_z > DISARM_ALTITUDE_THRESHOLD:
            msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
            body_z = msg.z  # down is positive in LOCAL_POSITION_NED
            print(f"[INFO] Altitude during landing: {body_z:.2f} m")
            time.sleep(0.1)

        # Once landed, disarm
        print("[INFO] Drone has landed. Disarming motors...")
        self.master.arducopter_disarm()