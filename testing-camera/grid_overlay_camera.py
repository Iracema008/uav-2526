import cv2
import depthai as dai
import numpy as np

class GridOverlayCamera:
    def __init__(self, grid_size=56):
        self.grid_size = grid_size  # 56 pixels = 1 foot
        self.pipeline = None
        self.device = None
        
    def setup_pipeline(self, yolo_model_path='yolo_model.blob'):
        """Setup DepthAI pipeline with YOLO detection"""
        self.pipeline = dai.Pipeline()
        
        # Create camera node
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)  # Adjust based on your camera
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        
        # Create YOLO detection network
        detection_nn = self.pipeline.create(dai.node.YoloDetectionNetwork)
        detection_nn.setBlobPath(yolo_model_path)
        detection_nn.setConfidenceThreshold(0.5)
        detection_nn.setNumClasses(80)  # COCO dataset classes
        detection_nn.setCoordinateSize(4)
        detection_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
        detection_nn.setAnchorMasks({"side80": [0, 1, 2], "side40": [3, 4, 5], "side20": [6, 7, 8]})
        detection_nn.setIouThreshold(0.5)
        
        # Link camera to detection network
        cam_rgb.preview.link(detection_nn.input)
        
        # Create output streams
        xout_video = self.pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName('video')
        detection_nn.passthrough.link(xout_video.input)
        
        xout_nn = self.pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName('detections')
        detection_nn.out.link(xout_nn.input)
        
        return self.pipeline
    
    def draw_grid(self, frame):
        """Draw 56x56 pixel grid overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw vertical lines (every 56 pixels)
        for x in range(0, w, self.grid_size):
            cv2.line(frame, (x, 0), (x, h), color=(0, 255, 0), thickness=1)
        
        # Draw horizontal lines (every 56 pixels)
        for y in range(0, h, self.grid_size):
            cv2.line(frame, (0, y), (w, y), color=(0, 255, 0), thickness=1)
        
        # Add grid labels (optional - shows foot measurements)
        for i, x in enumerate(range(0, w, self.grid_size)):
            cv2.putText(frame, f"{i}ft", (x + 2, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        for i, y in enumerate(range(0, h, self.grid_size)):
            cv2.putText(frame, f"{i}ft", (2, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def draw_detections(self, frame, detections):
        """Draw YOLO detections on frame"""
        for detection in detections:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(detection.xmin * frame.shape[1])
            y1 = int(detection.ymin * frame.shape[0])
            x2 = int(detection.xmax * frame.shape[1])
            y2 = int(detection.ymax * frame.shape[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label and confidence
            label = f"Class {detection.label}: {detection.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate which grid cell the object is in
            grid_x = x1 // self.grid_size
            grid_y = y1 // self.grid_size
            cv2.putText(frame, f"Grid: ({grid_x}, {grid_y})", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame
    
    def run_camera(self):
        """Main camera loop with grid overlay and YOLO detection"""
        try:
            with dai.Device(self.pipeline) as device:
                video_queue = device.getOutputQueue(name='video', maxSize=4, blocking=False)
                detections_queue = device.getOutputQueue(name='detections', maxSize=4, blocking=False)
                
                print("Camera started. Press 'q' to quit.")
                
                while True:
                    # Get frame
                    video_frame = video_queue.get().getCvFrame()
                    
                    # Get detections
                    detections = []
                    if not detections_queue.empty():
                        detections = detections_queue.get().detections
                    
                    # Draw grid overlay
                    video_frame = self.draw_grid(video_frame)
                    
                    # Draw YOLO detections
                    if detections:
                        video_frame = self.draw_detections(video_frame, detections)
                    
                    # Display frame
                    cv2.imshow('Grid Overlay Camera', video_frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cv2.destroyAllWindows()

def main():
    # Initialize the grid overlay camera
    camera = GridOverlayCamera(grid_size=56)  # 56 pixels = 1 foot
    
    # Setup pipeline (you'll need to provide the path to your YOLO model)
    pipeline = camera.setup_pipeline('path_to_your_yolo_model.blob')
    
    # Run the camera
    camera.run_camera()

if __name__ == "__main__":
    main()
