#!/usr/bin/env python3
"""
Simple OAK-D S-2 camera test to diagnose connection issues
"""

import depthai as dai
import sys

def test_camera_connection():
    """Test basic camera connection"""
    print("Testing OAK-D S-2 camera connection...")
    
    try:
        # Check available devices
        devices = dai.Device.getAllAvailableDevices()
        print(f"Available devices: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"Device {i}: {device}")
        
        if len(devices) == 0:
            print("No OAK-D devices found!")
            return False
            
        # Try to create a simple pipeline
        pipeline = dai.Pipeline()
        
        # Create camera node
        cam_rgb = pipeline.create(dai.node.Camera)
        cam_rgb.build(dai.CameraBoardSocket.CAM_A)
        
        print("Pipeline created successfully")
        
        # Try to connect to device without output for now
        with dai.Device() as device:
            device.startPipeline(pipeline)
            print("Successfully connected to OAK-D S-2!")
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_camera_connection()
    sys.exit(0 if success else 1)
