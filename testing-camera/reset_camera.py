#!/usr/bin/env python3
"""
Script to reset OAK-D S-2 camera connection
"""

import depthai as dai
import time

def reset_camera():
    """Try to reset the camera connection"""
    print("Attempting to reset OAK-D S-2 camera...")
    
    try:
        # Try to get all available devices
        devices = dai.Device.getAllAvailableDevices()
        print(f"Available devices: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"Device {i}: {device}")
        
        if len(devices) == 0:
            print("No devices found!")
            return False
            
        # Try to connect and immediately close to reset the connection
        print("Attempting to connect and reset...")
        device = dai.Device()
        print("Connected successfully!")
        time.sleep(1)
        device.close()
        print("Connection closed successfully!")
        
        # Wait a moment
        time.sleep(2)
        
        # Try to connect again
        print("Testing reconnection...")
        device2 = dai.Device()
        print("Reconnection successful!")
        device2.close()
        print("Camera reset completed!")
        return True
        
    except Exception as e:
        print(f"Error during reset: {e}")
        return False

if __name__ == "__main__":
    if reset_camera():
        print("✅ Camera reset successful!")
    else:
        print("❌ Camera reset failed.")

