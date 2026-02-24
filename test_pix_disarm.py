# This code tests and shows how to disarm the drone using the Pi and confirm it disarmed.
# It also details managing the message queue/buffer to prevent unwanted errors.
from pymavlink import mavutil
import time

def test_disarm_drone():
    print("Waiting for heartbeat from Pixhawk...")
    master.wait_heartbeat()
    print("Heartbeat Received\n")

    # 1. We wait a few seconds to let sensors settle while actively clearing the buffer. It will reject/delete 
    # newest messeges that the Pixhawk sends to the Pi, this is because the message buffer overflows. We solve 
    # the overflow issue by using the recv_match() helper function from the pymavlink/mavutil library
    # to read any incoming messages and clear the buffer.
    # IMPORTANT: SIMPLY DOING TIME.SLEEP() WILL NOT WORK AS WE NEED TO ALSO READ/CLEAR THE BUFFER 
    print("Waiting 3 seconds for sensors, GPS, and reading message queue...")
    start_time = time.time()
    while time.time() - start_time < 3:
        # Grab any waiting message and immediately discard it
        master.recv_match(blocking=False)
        # A tiny 0.1s sleep keeps CPU from maxing out at 100%
        time.sleep(0.1) 
        
    print("Done waiting!\n")

    # 2. Disarm drone using the built-in helper function from the pymavlink/mavutil library
    print("Disarming Drone...")
    master.arducopter_disarm()

    # 3. To catch the specific acknowledgment We must explicitly look for the acknowledgment of 
    # the command ID 400 (ARM/DISARM). If we don't filter for 400, pymavlink might grab a random
    # ACK from a background process. We use recv_match() helper function and wait up to 5 seconds.
    print("Waiting for disarm acknowledgment from flight controller...")
    # Get message
    msg = master.recv_match(type='COMMAND_ACK', condition='COMMAND_ACK.command==400', blocking=True, timeout=5)

    if msg:
        if msg.result == 0:
            print("Disarm command accepted by flight controller.\n")
        else:
            print(f"FAILURE! Disarm rejected. Result code: {msg.result}\n")
    else:
        print("FAILURE! No acknowledgment received (Timed Out).\n")

    # 4. Print the Disarmed State and read message buffer to confirm motors are disarmed
    # by using the motors_disarmed_wait() helper function from the Pymavlink/mavutil library
    print(f"Current Armed status is: {master.motors_armed()}")

    print("Reading message buffer to confirm motors are powered down...")
    start_time = time.time()
    while time.time() - start_time < 3:
        master.motors_disarmed_wait()
        print("SUCCESS! Motors are disarmed\n")
        break
    else:
        print("FAILURE! Motors failed to disarm\n")
    
if __name__ == "__main__":
    serial_port = '/dev/serial0'
    baudrate = 57600

    print("\nConnecting to Pixhawk...")
    master = mavutil.mavlink_connection(serial_port, baud=baudrate)
    test_disarm_drone()