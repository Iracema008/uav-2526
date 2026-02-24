# This code tests and shows how to switch flight modes using the Pi and confirm it switched modes.
# It also goes into some detail about the message queue/buffer, and how it must be kept from overflowing
# to prevent any unwanted errors/behavior. And the necessary steps for startup.
from pymavlink import mavutil
import time

def change_pixhawk_flight_mode(master):
    print("Waiting for heartbeat from Pixhawk...")
    master.wait_heartbeat()
    print("Heartbeat Received\n")

    # 1. ArduPilot will actively reject a switch to GUIDED mode if its (EKF) hasn't secured a solid GPS lock 
    # and stabilized its sensors. So this loop waits for 5 seconds to allow for this to happen.
    # It will also reject a switch/delete newest messeges that the Pixhawk sends to the Pi, this is because
    # the message buffer overflows. We solve the overflow issue by using the recv_match() helper function from
    # the pymavlink/mavutil library to read any incoming messages and clear the buffer.
    # IMPORTANT: SIMPLY DOING TIME.SLEEP() WILL NOT WORK AS WE NEED TO ALSO READ/CLEAR THE BUFFER
    print("Waiting 2 seconds for sensors, GPS, and reading message queue")
    start_time = time.time()
    while time.time() - start_time < 2:
        # Grab any waiting message and immediately discard it
        master.recv_match(blocking=False)
        # A tiny 0.1s sleep keep CPU from maxing out at 100%
        time.sleep(0.1) 
        
    print("Done waiting!\n")

    # 2. Switch to GUIDED mode using the built-in helper function from pymavlink/mavutil library
    print("Switching to GUIDED flight mode...")
    master.set_mode("GUIDED")

    # 3. Flush the buffer until we catch the updated heartbeat by pulling the next heartbeat from the queue.
    # Pymavlink caches the flight mode based on the LAST heartbeat it read. If we print the mode immediately,
    # it will falsely say print the wrong mode. To solve this we must actively pull new heartbeats from the queue 
    # until we catch up to the message that proves the flight controller actually did switched modes.
    print("Reading message buffer to catch up to mode change...")
    start_time = time.time()
    while time.time() - start_time < 5: # 5-second timeout
        # Get message
        msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        
        if msg and master.flightmode == 'GUIDED':
            print(f"SUCCESS! Pymavlink caught up and sees: {master.flightmode} flight mode\n")
            break
    else:
        # If the loop finishes the 5 seconds without breaking, it timed out
        print(f"FAILURE! Timed out waiting for GUIDED. Current mode seen: {master.flightmode}\n")

def arm(master):
    print("Arming")
    
    print("Starting full message stream... Press Ctrl+C to stop.")

    try:
        while True:
            # recv_match() with no arguments catches the next available message
            master.arducopter_arm()
            msg = master.recv_match(blocking=True)
        
            if not msg:
                continue
            
            # .to_dict() turns the MAVLink object into a readable Python dictionary
            print(f"MESSAGE: {msg.get_type()}")
            print(f"DATA:    {msg.to_dict()}\n")
        
    except KeyboardInterrupt:
        print("\nStream stopped by user.")


if __name__ == "__main__":
    serial_port = '/dev/serial0'
    baudrate =  57600

    print("\nConnecting to Pixhawk...")
    master = mavutil.mavlink_connection(serial_port, baud=baudrate)
    change_pixhawk_flight_mode(master)
    arm(master)
