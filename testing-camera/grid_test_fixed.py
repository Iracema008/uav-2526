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

def calculate_pitch_from_accelerometer(accel_x, accel_y, accel_z):
    """Calculate pitch angle from accelerometer data using gravity"""
    # Pitch is rotation around side-to-side axis (y-axis in camera frame)
    # Using atan2 with x component and magnitude of y/z
    # When camera is level, gravity points down in z-axis
    # When camera pitches up/down, x component changes
    pitch = math.atan2(-accel_x, math.sqrt(accel_y * accel_y + accel_z * accel_z))
    return pitch

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

def draw_crosshair(frame, center_x, center_y, grid_size=56, distance_mm=None):
    """Draw crosshair at specified position with real depth-based distance measurement"""
    h, w = frame.shape[:2]
    
    # Draw crosshair
    cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
    
    # Display distance info if available
    if distance_mm is not None and distance_mm > 0:
        distance_ft = distance_mm / 304.8  # Convert mm to feet
        distance_m = distance_mm / 1000.0  # Convert mm to meters
        pos_info = f"Distance: {distance_ft:.2f}ft ({distance_m:.2f}m)"
        cv2.putText(frame, pos_info, (center_x + 15, center_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        pos_info = "No depth data"
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
    print("- 'z': Zero/reset yaw (BMI270 only)")
    print("- 'q': Quit")
    print()
    
    try:
        print("OAK-D S-2 Grid Test Camera started. Press 'q' to quit.")
        print("Mouse click to place crosshair for distance measurement.")
        print("Controls: 'c'=clear crosshair, 'r'=reset to center, 's'=save frame, 'z'=reset yaw")
        
        # Mouse callback for crosshair placement - will be set up after device initialization
        crosshair_pos = None
        
        cv2.namedWindow('OAK-D S-2 Grid Test Camera')
        
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
            
            # Create stereo depth nodes for distance measurement
            monoLeft = pipeline.create(dai.node.MonoCamera)
            monoRight = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)
            xoutDepth = pipeline.create(dai.node.XLinkOut)
            
            # Configure mono cameras
            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Left camera
            monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Right camera
            
            # Configure stereo depth
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)  # Better accuracy
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to RGB camera
            
            xoutDepth.setStreamName("depth")
            
            # Link mono cameras to stereo depth
            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)
            stereo.depth.link(xoutDepth.input)
            
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
                current_roll = 0.0  # Current raw rotation value
                
                # Initialize BMI270 gyroscope integration variables for yaw calculation
                if imu_type == "BMI270":
                    integrated_yaw = 0.0  # Cumulative yaw angle (radians)
                    last_gyro_ts = None  # Last gyroscope timestamp for integration
                    # Drift correction variables
                    gyro_bias_z = 0.0  # Gyroscope bias on z-axis (rad/s)
                    gyro_bias_samples = []  # Samples for bias calculation
                    bias_calibration_samples = 100  # Number of samples to collect for bias (~2s at 50Hz)
                    bias_calibrated = False  # Whether bias has been calibrated
                    zero_rate_threshold = 0.05  # rad/s threshold to consider stationary (~2.9°/s)
            else:
                video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
            
            # Add depth queue for stereo depth measurements
            depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            current_depth_frame = None  # Store current depth frame for click measurements
            
            # Define mouse callback here so it can access current_depth_frame
            def mouse_callback(event, x, y, flags, param):
                nonlocal crosshair_pos, current_depth_frame
                if event == cv2.EVENT_LBUTTONDOWN:
                    crosshair_pos = (x, y)
                    
                    # Get depth at clicked point
                    if current_depth_frame is not None:
                        if 0 <= y < current_depth_frame.shape[0] and 0 <= x < current_depth_frame.shape[1]:
                            depth_value = current_depth_frame[y, x]
                            if depth_value > 0:
                                distance_ft = depth_value / 304.8
                                distance_m = depth_value / 1000.0
                                print(f"Crosshair at ({x}, {y}) - Distance: {distance_ft:.2f}ft ({distance_m:.2f}m)")
                            else:
                                print(f"Crosshair at ({x}, {y}) - No depth data at this point")
                        else:
                            print(f"Crosshair at ({x}, {y}) - Coordinates out of bounds")
                    else:
                        print(f"Crosshair at ({x}, {y}) - Depth frame not available yet")
            
            cv2.setMouseCallback('OAK-D S-2 Grid Test Camera', mouse_callback)
            
            while True:
                # Get depth frame (non-blocking)
                depth_packet = depth_queue.tryGet()
                if depth_packet is not None:
                    depth_frame = depth_packet.getFrame()  # Depth values in millimeters
                    current_depth_frame = depth_frame
                
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
                            current_roll = yaw  # Use yaw for grid rotation
                            
                        elif imu_type == "BMI270":
                            # ============================================================
                            # DRIFT-CORRECTED YAW INTEGRATION
                            # Uses gyroscope z-axis integration with bias correction and
                            # zero-rate detection to prevent drift accumulation over time
                            # ============================================================
                            
                            found_gyro = False
                            found_accel = False
                            accel_magnitude = 0.0
                            
                            # Process all packets to get both accelerometer and gyroscope data
                            for packet in imu_msg.packets:
                                # Process accelerometer to detect if camera is stationary
                                if hasattr(packet, 'acceleroMeter') and packet.acceleroMeter is not None:
                                    accel = packet.acceleroMeter
                                    # Calculate total acceleration magnitude
                                    # Should be ~9.8 m/s² when stationary (gravity only)
                                    accel_magnitude = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
                                    found_accel = True
                                
                                # Process gyroscope for yaw integration
                                if hasattr(packet, 'gyroscope') and packet.gyroscope is not None:
                                    gyro = packet.gyroscope
                                    gyro_ts = gyro.getTimestampDevice()
                                    
                                    # STEP 1: Calibrate gyroscope bias during initial period
                                    # This measures the sensor's offset when stationary
                                    if not bias_calibrated:
                                        # Detect if camera is stationary:
                                        # - Gyro reading near zero (not rotating)
                                        # - Acceleration matches gravity (~9.8 m/s²)
                                        is_stationary = (abs(gyro.z) < zero_rate_threshold and 
                                                        found_accel and 
                                                        9.0 < accel_magnitude < 10.5)
                                        
                                        if is_stationary:
                                            gyro_bias_samples.append(gyro.z)
                                            if len(gyro_bias_samples) >= bias_calibration_samples:
                                                # Calculate average bias (offset from true zero)
                                                gyro_bias_z = sum(gyro_bias_samples) / len(gyro_bias_samples)
                                                bias_calibrated = True
                                                print(f"✓ Gyroscope bias calibrated: {math.degrees(gyro_bias_z):.3f}°/s")
                                    else:
                                        # STEP 2: Apply bias correction to gyroscope reading
                                        # Subtract the measured offset to get true angular velocity
                                        gyro_z_corrected = gyro.z - gyro_bias_z
                                        
                                        # STEP 3: Detect if currently stationary (zero-rate detection)
                                        # When stationary, gyroscope should read zero (after bias correction)
                                        is_currently_stationary = (abs(gyro_z_corrected) < zero_rate_threshold)
                                        
                                        if last_gyro_ts is not None:
                                            dt = (gyro_ts - last_gyro_ts).total_seconds()
                                            
                                            # STEP 4: Only integrate if there's actual rotation
                                            # Prevents integrating noise/drift when camera is still
                                            if not is_currently_stationary:
                                                # Integrate: angle = angular_velocity * time_delta
                                                # This accumulates rotation over time
                                                integrated_yaw += gyro_z_corrected * dt
                                            # If stationary, don't integrate (prevents drift accumulation)
                                        
                                        last_gyro_ts = gyro_ts
                                        current_roll = integrated_yaw  # Use integrated yaw for grid rotation
                                        found_gyro = True
                                        break
                            
                            if not found_gyro:
                                # No gyroscope data found - keep previous integrated_yaw value
                                # current_roll already has previous value from integrated_yaw
                                pass
                            
                            # ============================================================
                            # OLD CODE (COMMENTED OUT - Basic integration without drift correction)
                            # ============================================================
                            # found_gyro = False
                            # for packet in imu_msg.packets:
                            #     if hasattr(packet, 'gyroscope') and packet.gyroscope is not None:
                            #         gyro = packet.gyroscope
                            #         gyro_ts = gyro.getTimestampDevice()
                            #         
                            #         if last_gyro_ts is not None:
                            #             # Calculate time delta in seconds
                            #             dt = (gyro_ts - last_gyro_ts).total_seconds()
                            #             # Integrate angular velocity: angle = velocity * time
                            #             # gyro.z is angular velocity in rad/s around z-axis (yaw)
                            #             integrated_yaw += gyro.z * dt
                            #         
                            #         last_gyro_ts = gyro_ts
                            #         current_roll = integrated_yaw  # Use integrated yaw for grid rotation
                            #         found_gyro = True
                            #         break
                            # 
                            # if not found_gyro:
                            #     # No gyroscope data found - keep previous integrated_yaw value
                            #     pass
                            # ============================================================
                        
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
                    # Get depth at crosshair position
                    cx, cy = crosshair_pos
                    distance_at_click = None
                    
                    if current_depth_frame is not None:
                        if 0 <= cy < current_depth_frame.shape[0] and 0 <= cx < current_depth_frame.shape[1]:
                            depth_val = current_depth_frame[cy, cx]
                            if depth_val > 0:  # Valid depth
                                distance_at_click = depth_val
                    
                    frame = draw_crosshair(frame, cx, cy, grid_size, distance_at_click)
                
                # Add frame counter and camera info
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "OAK-D S-2 Camera (IMU Enabled)" if use_imu else "OAK-D S-2 Camera (Simplified)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display IMU info if available
                if use_imu and abs(rotation_angle) > 0.01:
                    imu_info = f"Camera Yaw: {math.degrees(rotation_angle):.1f}°"
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
                elif key == ord('z'):
                    # Zero/reset yaw integration (BMI270 only - useful for recalibration)
                    if use_imu and imu_type == "BMI270":
                        integrated_yaw = 0.0
                        if initial_rotation is not None:
                            initial_rotation = 0.0
                        print("Yaw reset to zero")
                    else:
                        print("Yaw reset only available for BMI270 IMU")
                
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