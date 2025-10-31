#!/usr/bin/env python3
"""
Simplified OAK-D S-2 Grid Test Camera - No Class Encapsulation
"""

import cv2
import depthai as dai
import numpy as np
from datetime import timedelta
import math

def quaternion_to_euler(q_i, q_j, q_k, q_real):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    # Convert quaternion to Euler angles
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_real * q_i + q_j * q_k)
    cosr_cosp = 1 - 2 * (q_i * q_i + q_j * q_j)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (q_real * q_j - q_k * q_i)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_real * q_k + q_i * q_j)
    cosy_cosp = 1 - 2 * (q_j * q_j + q_k * q_k)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def calculate_roll_from_accelerometer(accel_x, accel_y, accel_z):
    """Calculate roll angle from accelerometer data using gravity"""
    # Roll is rotation around forward axis (x-axis in camera frame)
    # Using atan2 with y and z components of gravity
    # When camera is level, gravity points down in z-axis
    # When camera rolls, y and z components change
    roll = math.atan2(accel_y, accel_z)
    return roll

def draw_grid(frame, grid_size=56, rotation_angle=0.0):
    """Draw 56x56 pixel grid overlay on frame with optional rotation"""
    h, w = frame.shape[:2]
    
    # Create a black overlay for the grid
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw vertical lines (every 56 pixels)
    for x in range(0, w, grid_size):
        cv2.line(overlay, (x, 0), (x, h), color=(0, 255, 0), thickness=1)
        cv2.line(mask, (x, 0), (x, h), color=255, thickness=1)
    
    # Draw horizontal lines (every 56 pixels)
    for y in range(0, h, grid_size):
        cv2.line(overlay, (0, y), (w, y), color=(0, 255, 0), thickness=1)
        cv2.line(mask, (0, y), (w, y), color=255, thickness=1)
    
    # Add grid labels (shows foot measurements)
    for i, x in enumerate(range(0, w, grid_size)):
        cv2.putText(overlay, f"{i}ft", (x + 2, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(mask, f"{i}ft", (x + 2, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    
    for i, y in enumerate(range(0, h, grid_size)):
        cv2.putText(overlay, f"{i}ft", (2, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(mask, f"{i}ft", (2, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    
    # Apply rotation if needed
    if abs(rotation_angle) > 0.01:  # Only rotate if angle is significant
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, math.degrees(rotation_angle), 1.0)
        # Rotate overlay and mask
        overlay = cv2.warpAffine(overlay, rotation_matrix, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, rotation_matrix, (w, h), 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Blend overlay onto frame where mask is non-zero
    mask_bool = mask > 0
    if np.any(mask_bool):
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        frame[mask_bool] = blended[mask_bool]
    
    # Add grid info overlay (always on top, no rotation)
    grid_info = f"Grid: {grid_size}px = 1ft | Resolution: {w}x{h}"
    if abs(rotation_angle) > 0.01:
        grid_info += f" | Rotation: {math.degrees(rotation_angle):.1f}°"
    cv2.putText(frame, grid_info, (10, h - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def draw_crosshair(frame, center_x, center_y, grid_size=56):
    """Draw crosshair at specified position for distance measurement"""
    h, w = frame.shape[:2]
    
    # Draw crosshair
    cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
    
    # Calculate grid position
    grid_x = center_x // grid_size
    grid_y = center_y // grid_size
    
    # Calculate distance from center in feet
    distance_x = center_x / grid_size
    distance_y = center_y / grid_size
    
    # Display position info
    pos_info = f"Grid: ({grid_x}, {grid_y}) | Distance: ({distance_x:.1f}ft, {distance_y:.1f}ft)"
    cv2.putText(frame, pos_info, (center_x + 15, center_y - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    return frame

def main():
    """Main camera loop with grid overlay using OAK-D S-2"""
    grid_size = 56  # 56 pixels = 1 foot
    
    print("OAK-D S-2 Grid Test Camera System")
    print("==================================")
    print("Controls:")
    print("- Mouse click: Place crosshair for distance measurement")
    print("- 'c': Clear crosshair")
    print("- 'r': Reset crosshair to center")
    print("- 's': Save current frame")
    print("- 'q': Quit")
    print()
    
    try:
        print("OAK-D S-2 Grid Test Camera started. Press 'q' to quit.")
        print("Mouse click to place crosshair for distance measurement.")
        print("Controls: 'c'=clear crosshair, 'r'=reset to center, 's'=save frame")
        
        # Mouse callback for crosshair placement
        crosshair_pos = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal crosshair_pos
            if event == cv2.EVENT_LBUTTONDOWN:
                crosshair_pos = (x, y)
                grid_x = x // grid_size
                grid_y = y // grid_size
                distance_x = x / grid_size
                distance_y = y / grid_size
                print(f"Crosshair placed at: ({x}, {y}) - Grid: ({grid_x}, {grid_y}) - Distance: ({distance_x:.1f}ft, {distance_y:.1f}ft)")
        
        cv2.namedWindow('OAK-D S-2 Grid Test Camera')
        cv2.setMouseCallback('OAK-D S-2 Grid Test Camera', mouse_callback)
        
        frame_count = 0
        
        # Use the simplest possible connection (following working pattern)
        device = dai.Device()
        
        # Check IMU availability
        imu_type = device.getConnectedIMU()
        imu_firmware_version = device.getIMUFirmwareVersion()
        print(f"IMU type: {imu_type}, firmware version: {imu_firmware_version}")
        
        use_imu = False
        if imu_type == "BNO086":
            use_imu = True
            print("Rotation vector supported - grid will rotate with camera movement!")
        elif imu_type == "BMI270":
            use_imu = True
            print("BMI270 detected - using accelerometer/gyroscope for orientation tracking!")
        else:
            print(f"Warning: IMU type {imu_type} not fully supported. Grid will not rotate.")
            print("Grid overlay will work but won't respond to camera movement.")
        
        with device:
            print("OAK-D S-2 device connected successfully!")
            
            # Create pipeline
            pipeline = dai.Pipeline()
            
            # Create camera node
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            
            # Create IMU node if supported
            rotation_angle = 0.0
            if use_imu:
                imu = pipeline.create(dai.node.IMU)
                sync = pipeline.create(dai.node.Sync)
                xout_imu = pipeline.create(dai.node.XLinkOut)
                xout_imu.setStreamName("imu")
                
                # Enable appropriate sensors based on IMU type
                if imu_type == "BNO086":
                    # Enable rotation vector sensor (provides quaternion directly)
                    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 120)
                elif imu_type == "BMI270":
                    # Enable accelerometer and gyroscope for BMI270
                    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
                    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
                
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)
                
                # Sync video and IMU data
                sync.setSyncThreshold(timedelta(milliseconds=10))
                sync.setSyncAttempts(-1)
                
                xout_sync = pipeline.create(dai.node.XLinkOut)
                xout_sync.setStreamName("sync")
                
                cam_rgb.video.link(sync.inputs["video"])
                imu.out.link(sync.inputs["imu"])
                sync.out.link(xout_sync.input)
            else:
                # Standard video output without IMU
                xout_video = pipeline.create(dai.node.XLinkOut)
                xout_video.setStreamName("video")
                cam_rgb.video.link(xout_video.input)
            
            # Start pipeline
            device.startPipeline(pipeline)
            
            # Get output queues
            if use_imu:
                sync_queue = device.getOutputQueue(name="sync", maxSize=4, blocking=False)
                # Store initial rotation as reference
                initial_rotation = None
                current_roll = 0.0  # Current raw roll value
            else:
                video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
            
            while True:
                # Get frame and IMU data (if available)
                if use_imu:
                    sync_data = sync_queue.get()
                    video_msg = sync_data["video"]
                    imu_msg = sync_data["imu"]
                    frame = video_msg.getCvFrame()
                    
                    # Get latest IMU data
                    rotation_angle = 0.0
                    if imu_msg.packets:
                        latest_packet = imu_msg.packets[-1]
                        
                        if imu_type == "BNO086":
                            # Use rotation vector (quaternion) for BNO086
                            latest_rv = latest_packet.rotationVector
                            q_i, q_j, q_k, q_real = latest_rv.i, latest_rv.j, latest_rv.k, latest_rv.real
                            # Convert quaternion to Euler angles
                            roll, pitch, yaw = quaternion_to_euler(q_i, q_j, q_k, q_real)
                            current_roll = roll
                            
                        elif imu_type == "BMI270":
                            # Use accelerometer to calculate roll for BMI270
                            # Process all packets to find accelerometer data
                            found_accel = False
                            for packet in imu_msg.packets:
                                if hasattr(packet, 'acceleroMeter') and packet.acceleroMeter is not None:
                                    accel = packet.acceleroMeter
                                    roll = calculate_roll_from_accelerometer(accel.x, accel.y, accel.z)
                                    current_roll = roll
                                    found_accel = True
                                    break
                            
                            if not found_accel:
                                # No accelerometer data found - use previous value if available
                                # current_roll already has previous value
                                pass
                        
                        # Store initial rotation as reference (zero point)
                        if initial_rotation is None:
                            initial_rotation = current_roll
                            rotation_angle = 0.0
                        else:
                            # Calculate rotation relative to initial position
                            rotation_angle = current_roll - initial_rotation
                else:
                    in_video = video_queue.get()
                    frame = in_video.getCvFrame()
                    rotation_angle = 0.0
                
                # Draw grid overlay with rotation
                frame = draw_grid(frame, grid_size, rotation_angle)
                
                # Draw crosshair if placed
                if crosshair_pos:
                    frame = draw_crosshair(frame, crosshair_pos[0], crosshair_pos[1], grid_size)
                
                # Add frame counter and camera info
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "OAK-D S-2 Camera (IMU Enabled)" if use_imu else "OAK-D S-2 Camera (Simplified)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display IMU info if available
                if use_imu and abs(rotation_angle) > 0.01:
                    imu_info = f"Camera Roll: {math.degrees(rotation_angle):.1f}°"
                    cv2.putText(frame, imu_info, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display frame
                cv2.imshow('OAK-D S-2 Grid Test Camera', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    crosshair_pos = None  # Clear crosshair
                    print("Crosshair cleared")
                elif key == ord('r'):
                    # Reset to center
                    h, w = frame.shape[:2]
                    crosshair_pos = (w//2, h//2)
                    print(f"Crosshair reset to center: ({w//2}, {h//2})")
                elif key == ord('s'):
                    # Save current frame
                    filename = f"oak_grid_test_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as: {filename}")
                
                frame_count += 1
                        
    except Exception as e:
        print(f"Error with OAK-D S-2 camera: {e}")
        print("Make sure:")
        print("1. OAK-D S-2 is connected via USB")
        print("2. DepthAI is properly installed")
        print("3. Camera is not being used by another application")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()