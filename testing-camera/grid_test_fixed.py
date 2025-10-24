#!/usr/bin/env python3
"""
OAK-D S-2 Grid Test - Fixed version using correct DepthAI API
"""

import cv2
import depthai as dai
import numpy as np

class GridTestCamera:
    def __init__(self, grid_size=56):
        self.grid_size = grid_size  # 56 pixels = 1 foot
        self.pipeline = None
        self.device = None
        
    def setup_oak_camera(self):
        """Setup OAK-D S-2 camera using correct DepthAI API"""
        try:
            self.pipeline = dai.Pipeline()
            
            # Create camera node using the new Camera node (not deprecated)
            cam_rgb = self.pipeline.create(dai.node.Camera)
            cam_rgb.build(dai.CameraBoardSocket.CAM_A)  # Use build method with CAM_A
            
            # Create output stream using SPIOut (modern approach)
            xout_video = self.pipeline.create(dai.node.SPIOut)
            xout_video.setStreamName('video')
            cam_rgb.raw.link(xout_video.input)
            
            print("OAK-D S-2 camera pipeline created successfully")
            return self.pipeline
            
        except Exception as e:
            print(f"OAK-D S-2 camera setup failed: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure OAK-D S-2 is connected via USB")
            print("2. Check if DepthAI is installed: pip install depthai")
            print("3. Verify camera is not being used by another application")
            print("4. Try unplugging and reconnecting the camera")
            print("5. Check USB cable and power supply")
            return None
    
    def draw_grid(self, frame):
        """Draw 56x56 pixel grid overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw vertical lines (every 56 pixels)
        for x in range(0, w, self.grid_size):
            cv2.line(frame, (x, 0), (x, h), color=(0, 255, 0), thickness=1)
        
        # Draw horizontal lines (every 56 pixels)
        for y in range(0, h, self.grid_size):
            cv2.line(frame, (0, y), (w, y), color=(0, 255, 0), thickness=1)
        
        # Add grid labels (shows foot measurements)
        for i, x in enumerate(range(0, w, self.grid_size)):
            cv2.putText(frame, f"{i}ft", (x + 2, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        for i, y in enumerate(range(0, h, self.grid_size)):
            cv2.putText(frame, f"{i}ft", (2, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add grid info overlay
        grid_info = f"Grid: {self.grid_size}px = 1ft | Resolution: {w}x{h}"
        cv2.putText(frame, grid_info, (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_crosshair(self, frame, center_x, center_y):
        """Draw crosshair at specified position for distance measurement"""
        h, w = frame.shape[:2]
        
        # Draw crosshair
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
        
        # Calculate grid position
        grid_x = center_x // self.grid_size
        grid_y = center_y // self.grid_size
        
        # Calculate distance from center in feet
        distance_x = center_x / self.grid_size
        distance_y = center_y / self.grid_size
        
        # Display position info
        pos_info = f"Grid: ({grid_x}, {grid_y}) | Distance: ({distance_x:.1f}ft, {distance_y:.1f}ft)"
        cv2.putText(frame, pos_info, (center_x + 15, center_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return frame
    
    def run_camera(self):
        """Main camera loop with grid overlay using OAK-D S-2"""
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
                    grid_x = x // self.grid_size
                    grid_y = y // self.grid_size
                    distance_x = x / self.grid_size
                    distance_y = y / self.grid_size
                    print(f"Crosshair placed at: ({x}, {y}) - Grid: ({grid_x}, {grid_y}) - Distance: ({distance_x:.1f}ft, {distance_y:.1f}ft)")
            
            cv2.namedWindow('OAK-D S-2 Grid Test Camera')
            cv2.setMouseCallback('OAK-D S-2 Grid Test Camera', mouse_callback)
            
            frame_count = 0
            
            # Use DepthAI device with the pipeline (modern API)
            with dai.Device() as device:
                device.startPipeline(self.pipeline)
                print("OAK-D S-2 device connected successfully!")
                
                # Get the video output from the camera
                video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                
                while True:
                    # Get frame from OAK-D S-2
                    in_video = video_queue.get()
                    frame = in_video.getCvFrame()
                    
                    # Draw grid overlay
                    frame = self.draw_grid(frame)
                    
                    # Draw crosshair if placed
                    if crosshair_pos:
                        frame = self.draw_crosshair(frame, crosshair_pos[0], crosshair_pos[1])
                    
                    # Add frame counter and camera info
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, "OAK-D S-2 Camera", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
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
            print("2. Camera is not being used by another application")
            print("3. Try unplugging and reconnecting the camera")
        finally:
            cv2.destroyAllWindows()

def main():
    # Initialize the grid test camera
    camera = GridTestCamera(grid_size=56)  # 56 pixels = 1 foot
    
    print("OAK-D S-2 Grid Test Camera System")
    print("=================================")
    print("Controls:")
    print("- Mouse click: Place crosshair for distance measurement")
    print("- 'c': Clear crosshair")
    print("- 'r': Reset crosshair to center")
    print("- 's': Save current frame")
    print("- 'q': Quit")
    print()
    
    # Setup OAK-D S-2 camera
    if camera.setup_oak_camera():
        camera.run_camera()
    else:
        print("Failed to setup OAK-D S-2 camera pipeline")

if __name__ == "__main__":
    main()


