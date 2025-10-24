#!/usr/bin/env python3
"""
Simple OAK-D S-2 camera test using the most basic DepthAI API
"""

import cv2
import depthai as dai
import numpy as np

def test_simple_camera():
    """Test the simplest possible camera connection"""
    print("Testing simple OAK-D S-2 camera connection...")
    
    try:
        # Check available devices
        devices = dai.Device.getAllAvailableDevices()
        print(f"Available devices: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"Device {i}: {device}")
        
        if len(devices) == 0:
            print("No OAK-D devices found!")
            return False
            
        # Try the simplest possible connection
        with dai.Device() as device:
            print("Successfully connected to OAK-D S-2!")
            print("Device info:", device.getDeviceInfo())
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_simple_camera()
    if success:
        print("✅ Camera connection successful!")
    else:
        print("❌ Camera connection failed!")


