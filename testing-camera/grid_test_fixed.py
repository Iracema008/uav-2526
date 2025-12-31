#!/usr/bin/env python3
"""
Simplified OAK-D S-2 Grid Test Camera - No Class Encapsulation
"""

import cv2
import depthai as dai
import numpy as np
from datetime import timedelta
import math
import os
import time

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

def create_pipeline(use_imu, imu_type, depth_enabled=True, use_rgb_output=False):
    """Create and configure the depthai pipeline with optional components"""
    pipeline = dai.Pipeline()
    
    # Create RGB camera node (still needed for depth alignment even if not output)
    cam_rgb = None
    if use_rgb_output or depth_enabled:  # Need RGB for depth alignment or if using RGB output
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    # Create mono cameras (always create when depth is enabled OR when not using RGB output)
    monoLeft = None
    monoRight = None
    
    if depth_enabled or not use_rgb_output:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        
        # Configure mono cameras
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Left camera
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Right camera
    
    # Create stereo depth nodes conditionally - only when depth_enabled
    if depth_enabled:
        stereo = pipeline.create(dai.node.StereoDepth)
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        
        # Configure stereo depth - use DEFAULT for Pi5 compatibility
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)  # Better accuracy
        # Reduce processing load on Pi5
        stereo.setSubpixel(False)  # Disable subpixel for better performance
        if cam_rgb is not None:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to RGB camera
        else:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)  # Align to left mono if no RGB
        
        xoutDepth.setStreamName("depth")
        
        # Link mono cameras to stereo depth
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(xoutDepth.input)
        
        # If not using RGB output, use stereo's rectified outputs for display
        if not use_rgb_output:
            xoutMonoLeft = pipeline.create(dai.node.XLinkOut)
            xoutMonoRight = pipeline.create(dai.node.XLinkOut)
            xoutMonoLeft.setStreamName("monoLeft")
            xoutMonoRight.setStreamName("monoRight")
            stereo.rectifiedLeft.link(xoutMonoLeft.input)
            stereo.rectifiedRight.link(xoutMonoRight.input)
    
    # Create mono camera outputs if not using RGB and depth is disabled
    elif not use_rgb_output and monoLeft is not None:
        # When depth is disabled, link mono cameras directly to output
        xoutMonoLeft = pipeline.create(dai.node.XLinkOut)
        xoutMonoRight = pipeline.create(dai.node.XLinkOut)
        xoutMonoLeft.setStreamName("monoLeft")
        xoutMonoRight.setStreamName("monoRight")
        monoLeft.out.link(xoutMonoLeft.input)
        monoRight.out.link(xoutMonoRight.input)
    
    # Create IMU node if supported
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
        
        # Link appropriate video source to sync
        if use_rgb_output and cam_rgb is not None:
            cam_rgb.video.link(sync.inputs["video"])
        elif monoLeft is not None:
            # Sync with mono left camera
            monoLeft.out.link(sync.inputs["video"])
        else:
            raise RuntimeError("No video source available for IMU sync")
        
        imu.out.link(sync.inputs["imu"])
        sync.out.link(xout_sync.input)
    else:
        # Standard video output without IMU
        if use_rgb_output and cam_rgb is not None:
            xout_video = pipeline.create(dai.node.XLinkOut)
            xout_video.setStreamName("video")
            cam_rgb.video.link(xout_video.input)
        # Note: When not using RGB, we use monoLeft_queue/monoRight_queue instead of video_queue
        # These are already created above when depth_enabled or not use_rgb_output
    
    return pipeline

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
    print("- 'd': Toggle depth on/off (restarts pipeline, improves performance when off)")
    print("- 'm': Toggle RGB/Mono camera output (restarts pipeline)")
    print("- 'q': Quit")
    print()
    
    try:
        # Fix Qt/Wayland issues on Raspberry Pi
        if 'QT_QPA_PLATFORM' not in os.environ:
            # Try xcb first (for X11), fall back to offscreen if no display
            try:
                os.environ['QT_QPA_PLATFORM'] = 'xcb'
            except:
                os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        print("OAK-D S-2 Grid Test Camera started. Press 'q' to quit.")
        print("Mouse click to place crosshair for distance measurement.")
        print("Controls: 'c'=clear crosshair, 'r'=reset to center, 's'=save frame, 'd'=toggle depth, 'm'=toggle RGB/Mono, 'z'=reset yaw")
        
        # Initialize depth enabled state (start with True for depth support)
        depth_enabled = True
        # Initialize RGB output state (False = use mono cameras, True = use RGB camera)
        use_rgb_output = False
        
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
            print("Note: Keep camera STILL for first 2 seconds to calibrate gyroscope bias!")
        else:
            print(f"Warning: IMU type {imu_type} not fully supported. Grid will not rotate.")
            print("Grid overlay will work but won't respond to camera movement.")
        
        # Initialize IMU state variables (preserved across pipeline restarts)
        initial_rotation = None
        current_roll = 0.0
        integrated_yaw = 0.0
        last_gyro_ts = None
        gyro_bias_z = 0.0
        gyro_bias_samples = []
        bias_calibration_samples = 100
        bias_calibrated = False
        zero_rate_threshold = 0.05
        
        # Main loop with pipeline restart capability
        # We need to reconnect device when restarting pipeline
        while True:
            # Close previous connection if exists (will happen on second iteration)
            try:
                device.close()
            except:
                pass
            
            # Reconnect device
            device = dai.Device()
            
            with device:
                print("OAK-D S-2 device connected successfully!")
                if depth_enabled:
                    print("Depth: ENABLED (stereo processing active)")
                else:
                    print("Depth: DISABLED (better performance, no distance measurement)")
                if use_rgb_output:
                    print("Camera: RGB (Color)")
                else:
                    print("Camera: Mono (Grayscale)")
                
                # Create pipeline with current depth_enabled and use_rgb_output settings
                pipeline = create_pipeline(use_imu, imu_type, depth_enabled, use_rgb_output)
                
                # Start pipeline
                device.startPipeline(pipeline)
                
                # Small delay to let streams initialize (helps prevent XLink errors on Pi)
                time.sleep(0.5)
                
                # Get output queues based on current configuration
                sync_queue = None
                video_queue = None
                monoLeft_queue = None
                monoRight_queue = None
                
                if use_imu:
                    sync_queue = device.getOutputQueue(name="sync", maxSize=4, blocking=False)
                    # Also get mono camera queues if not using RGB (for side-by-side display)
                    if not use_rgb_output:
                        monoLeft_queue = device.getOutputQueue(name="monoLeft", maxSize=4, blocking=False)
                        monoRight_queue = device.getOutputQueue(name="monoRight", maxSize=4, blocking=False)
                else:
                    # When not using IMU, use video queue if RGB, otherwise use mono queues
                    if use_rgb_output:
                        video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                    else:
                        monoLeft_queue = device.getOutputQueue(name="monoLeft", maxSize=4, blocking=False)
                        monoRight_queue = device.getOutputQueue(name="monoRight", maxSize=4, blocking=False)
                
                # Get depth queue only if depth is enabled
                depth_queue = None
                current_depth_frame = None
                if depth_enabled:
                    try:
                        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                    except Exception as e:
                        print(f"Warning: Could not create depth queue: {e}")
                        depth_queue = None
                        depth_enabled = False  # Disable depth if queue creation fails
                
                # Define mouse callback here so it can access current_depth_frame
                def mouse_callback(event, x, y, flags, param):
                    nonlocal crosshair_pos, current_depth_frame, depth_enabled
                    if event == cv2.EVENT_LBUTTONDOWN:
                        crosshair_pos = (x, y)
                        
                        # Get depth at clicked point
                        if not depth_enabled:
                            print(f"Crosshair at ({x}, {y}) - Depth disabled (press 'd' to enable)")
                        elif current_depth_frame is not None:
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
                
                restart_pipeline = False
                consecutive_errors = 0
                max_consecutive_errors = 10
                
                while True:
                    # Small delay to reduce CPU load on Pi5
                    time.sleep(0.01)  # 10ms delay
                    
                    # Get depth frame (non-blocking) - only when enabled
                    if depth_enabled and depth_queue is not None:
                        try:
                            depth_packet = depth_queue.tryGet()
                            if depth_packet is not None:
                                depth_frame = depth_packet.getFrame()  # Depth values in millimeters
                                current_depth_frame = depth_frame
                        except Exception as e:
                            # Handle XLink errors gracefully (e.g., during pipeline restart)
                            error_str = str(e).lower()
                            if "x_link_error" in error_str or "couldn't read data" in error_str:
                                # Silently handle XLink errors during restart
                                current_depth_frame = None
                            else:
                                print(f"Warning: Error reading depth frame: {e}")
                            # If persistent errors, disable depth to prevent crashes
                            if "x_link_error" in error_str:
                                depth_queue = None
                                depth_enabled = False
                                print("Depth disabled due to communication error")
                    
                    # Get frame and IMU data (if available) - USE NON-BLOCKING
                    if use_imu:
                        try:
                            sync_data = sync_queue.tryGet()
                            if sync_data is None:
                                # No sync data yet, skip this frame but continue processing
                                # This prevents blocking when depth processing is busy
                                continue
                            
                            video_msg = sync_data["video"]
                            imu_msg = sync_data["imu"]
                            frame = video_msg.getCvFrame()
                        except Exception as e:
                            # Handle XLink errors gracefully
                            error_str = str(e).lower()
                            if "x_link_error" in error_str or "couldn't read data" in error_str or "communication exception" in error_str:
                                consecutive_errors += 1
                                if consecutive_errors >= max_consecutive_errors:
                                    print("Too many consecutive sync errors, restarting pipeline...")
                                    restart_pipeline = True
                                    break
                                # Skip this frame and continue
                                time.sleep(0.1)  # Wait a bit before retrying
                                continue
                            else:
                                print(f"Warning: Error reading sync data: {e}")
                                consecutive_errors += 1
                                if consecutive_errors >= max_consecutive_errors:
                                    restart_pipeline = True
                                    break
                                continue
                        
                        # Reset error counter on successful read
                        consecutive_errors = 0
                        
                        # Convert mono to BGR for display if using mono cameras
                        if not use_rgb_output:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            
                            # Show both mono cameras side-by-side
                            if monoRight_queue is not None:
                                try:
                                    mono_right_packet = monoRight_queue.tryGet()
                                    if mono_right_packet is not None:
                                        mono_right_frame = mono_right_packet.getCvFrame()
                                        mono_right_frame = cv2.cvtColor(mono_right_frame, cv2.COLOR_GRAY2BGR)
                                        # Combine frames side-by-side
                                        frame = np.hstack([frame, mono_right_frame])
                                except Exception as e:
                                    # Silently skip right camera if error occurs
                                    pass
                        
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
                                            
                                            # FIX: Initialize timestamp and set current_roll even during calibration
                                            if last_gyro_ts is None:
                                                last_gyro_ts = gyro_ts
                                            current_roll = integrated_yaw  # Update even during calibration
                                            found_gyro = True
                                        else:
                                            # STEP 2: Apply bias correction to gyroscope reading
                                            # Subtract the measured offset to get true angular velocity
                                            gyro_z_corrected = gyro.z - gyro_bias_z
                                            
                                            # STEP 3: Detect if currently stationary (zero-rate detection)
                                            # When stationary, gyroscope should read zero (after bias correction)
                                            is_currently_stationary = (abs(gyro_z_corrected) < zero_rate_threshold)
                                            
                                            # FIX: Initialize timestamp on first reading if needed
                                            if last_gyro_ts is None:
                                                last_gyro_ts = gyro_ts
                                                # Don't integrate on first reading, just initialize
                                                current_roll = integrated_yaw
                                            else:
                                                dt = (gyro_ts - last_gyro_ts).total_seconds()
                                                
                                                # STEP 4: Only integrate if there's actual rotation
                                                # Prevents integrating noise/drift when camera is still
                                                if not is_currently_stationary:
                                                    # Integrate: angle = angular_velocity * time_delta
                                                    # This accumulates rotation over time
                                                    integrated_yaw += gyro_z_corrected * dt
                                            
                                            last_gyro_ts = gyro_ts
                                            current_roll = integrated_yaw  # Use integrated yaw for grid rotation
                                            found_gyro = True
                                
                                # FIX: Ensure current_roll is updated even if no gyro found in this packet
                                if not found_gyro:
                                    # No gyroscope data found - keep previous integrated_yaw value
                                    current_roll = integrated_yaw  # Explicitly update from last known value
                            
                            # Store initial rotation as reference (zero point)
                            # FIX: Only set initial_rotation if we actually have valid data
                            if initial_rotation is None and abs(current_roll) < 10.0:  # Sanity check: avoid extreme values
                                initial_rotation = current_roll
                                rotation_angle = 0.0
                            elif initial_rotation is not None:
                                # Calculate rotation relative to initial position
                                rotation_angle = current_roll - initial_rotation
                            else:
                                # Still initializing
                                rotation_angle = 0.0
                    else:
                        try:
                            if use_rgb_output and video_queue is not None:
                                in_video = video_queue.get()
                                frame = in_video.getCvFrame()
                            elif monoLeft_queue is not None:
                                # Get mono left frame
                                in_video = monoLeft_queue.tryGet()
                                if in_video is not None:
                                    frame = in_video.getCvFrame()
                                    # Convert grayscale to BGR for display
                                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                    
                                    # Optionally show both mono cameras side-by-side
                                    if monoRight_queue is not None:
                                        try:
                                            mono_right_packet = monoRight_queue.tryGet()
                                            if mono_right_packet is not None:
                                                mono_right_frame = mono_right_packet.getCvFrame()
                                                mono_right_frame = cv2.cvtColor(mono_right_frame, cv2.COLOR_GRAY2BGR)
                                                # Combine frames side-by-side
                                                frame = np.hstack([frame, mono_right_frame])
                                        except Exception as e:
                                            # Silently skip right camera if error occurs
                                            pass
                                else:
                                    continue
                            else:
                                continue
                        except Exception as e:
                            # Handle XLink errors gracefully
                            error_str = str(e).lower()
                            if "x_link_error" in error_str or "couldn't read data" in error_str or "communication exception" in error_str:
                                consecutive_errors += 1
                                if consecutive_errors >= max_consecutive_errors:
                                    print("Too many consecutive video errors, restarting pipeline...")
                                    restart_pipeline = True
                                    break
                                # Skip this frame and continue
                                time.sleep(0.1)  # Wait a bit before retrying
                                continue
                            else:
                                print(f"Warning: Error reading video frame: {e}")
                                consecutive_errors += 1
                                if consecutive_errors >= max_consecutive_errors:
                                    restart_pipeline = True
                                    break
                                continue
                        
                        # Reset error counter on successful read
                        consecutive_errors = 0
                        rotation_angle = 0.0
                    
                    # Draw grid overlay with rotation
                    frame = draw_grid(frame, grid_size, rotation_angle)
                    
                    # Draw crosshair if placed
                    if crosshair_pos:
                        # Get depth at crosshair position (only if depth is enabled)
                        cx, cy = crosshair_pos
                        distance_at_click = None
                        
                        if depth_enabled and current_depth_frame is not None:
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
                    
                    # Display depth status and camera type
                    h, w = frame.shape[:2]
                    depth_status = "Depth: ON" if depth_enabled else "Depth: OFF"
                    depth_color = (0, 255, 0) if depth_enabled else (0, 0, 255)
                    cv2.putText(frame, depth_status, (10, h - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, depth_color, 1)
                    camera_type = "RGB" if use_rgb_output else "Mono"
                    cv2.putText(frame, f"Camera: {camera_type}", (10, h - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display frame
                    cv2.imshow('OAK-D S-2 Grid Test Camera', frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        restart_pipeline = False
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
                    elif key == ord('d'):
                        # Toggle depth processing on/off - requires pipeline restart
                        depth_enabled = not depth_enabled
                        restart_pipeline = True
                        # Clear depth queue to prevent XLink errors
                        if depth_queue is not None:
                            try:
                                # Drain any remaining packets
                                while depth_queue.tryGet() is not None:
                                    pass
                            except:
                                pass
                            depth_queue = None
                        current_depth_frame = None  # Clear depth frame
                        if depth_enabled:
                            print("Depth measurement ENABLED - Restarting pipeline...")
                        else:
                            print("Depth measurement DISABLED - Restarting pipeline for better performance...")
                        break  # Exit inner loop to restart pipeline
                    elif key == ord('m'):
                        # Toggle RGB/Mono camera output - requires pipeline restart
                        use_rgb_output = not use_rgb_output
                        restart_pipeline = True
                        if use_rgb_output:
                            print("Switching to RGB camera - Restarting pipeline...")
                        else:
                            print("Switching to Mono cameras - Restarting pipeline...")
                        break  # Exit inner loop to restart pipeline
                    
                    frame_count += 1
                
                # Exit outer loop if not restarting
                if not restart_pipeline:
                    break
                        
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
