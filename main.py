#import tensorflow as tf
#import akida_model as akida
import cv2

from vison.videoCapture import VideoCapture
from vison.log import get_logger
from vison.json_utils import read_json
from types import SimpleNamespace
from ultralytics import YOLO


logger = get_logger(__name__)

class Uav:
    """Main autonomous UAV class.

    Attr:
        conf: App configurations
        video_capture: Video capture class that provides frames for object detection
        camera_coordinate_transformer: 2d camera coordinate to 3d world coordinate transformer
        detector: Target identifier/detector
        fps_tracker: Tracks the fps of image processing
    """

    def __init__(self, conf: SimpleNamespace) -> None:
        """Constructor for AutoUav.

        Args:
            conf: App configuration
        """
        self.conf = conf
        self.video_capture: VideoCapture = VideoCapture(conf.video)
        #self.camera_coordinate_transformer: CameraCoordinateTransformer = (
        #    CameraCoordinateTransformer(conf.video)
        #)
        #self.detector: Detector = DetectorManager(conf.detector).get_detector()
        #self.fps_tracker: FPSTracker = FPSTracker()
        self.use_depthai = getattr(conf, "use_depthai", False)
        self.correct_marker = False
        self.marker_detected_before = False


    def clean_up(self) -> None:
        """Cleanup for AutoUav."""
        logger.info("Cleaning up")
        self.video_capture.stop()

    
    '''
    def run(self) -> None:
        """Runs the main logic."""
        logger.info("AutoUav starting up...")
        self.video_capture.start()

        while True:
            # get frame from the video capture
            frame = self.video_capture._capture_frames()
            #changed from read() to capture_frames()

            if frame is None:
                logger.warning("Recieved Empty Frame")
                cv2.waitKey(1)

            corners, ids, _ = self.detector.detect(frame, True)
            self.check_ids(frame, ids)

            # update fps tracker
            self.fps_tracker.update()
            self.fps_tracker.put_fps_on_frame(frame)

            # if no corners are detected, show the frame and continue to next frame
            if not corners:
                if self.conf.video.show_video:
                    cv2.imshow("Video Capture", frame)
                    #wait for user to press any key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # get center of corners - WE ARE ASSUMING ONLY ONE ARUCO AT A TIME RIGHT NOW
            # TODO: accomodate for scenarios of multiple aruco markers
            center_x: float = (
                corners[0][0][0][0] + (corners[0][0][2][0] - corners[0][0][0][0]) / 2
            )
            center_y: float = (
                corners[0][0][0][1] + (corners[0][0][2][1] - corners[0][0][0][1]) / 2
            )

            # After detection, we will translate into commands that can be sent to the PIXHAWK
            # TODO: Implement CameraCoordinateTransformer, FieldCoordinateMapper,
            # CommandGenerator
            x, y, z = self.camera_coordinate_transformer.transform((center_x, center_y), 10)

            # draw an arrow from center of frame to the center of the aruco marker
            cv2.arrowedLine(
                frame,
                tuple(map(int, self.camera_coordinate_transformer.camera_center)),
                tuple(map(int, (center_x, center_y))),
                (255, 0, 0),
                2,
            )
            relative_coordinates_text = [
                f"{'FORWARD' if y > 0 else 'BACK'}: {abs(y)} meters",
                f"{'RIGHT' if x > 0 else 'LEFT'}: {abs(x)} meters",
                f"{'BELOW'}: {abs(z)} meters",
            ]

            #font
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)  # Black text
            line_thickness = 2
            line_spacing = 20  # Spacing between lines

            # starting point for text (just offset a bit from the endpoint)
            text_start_x, text_start_y = int(center_x) + 10, int(center_y) - 10

            # draw each line of text
            for i, line in enumerate(relative_coordinates_text):
                y_offset = text_start_y + i * line_spacing
                cv2.putText(
                    frame,
                    line,
                    (text_start_x, y_offset),
                    font,
                    font_scale,
                    font_color,
                    line_thickness,
                )

            # show text of centering success
            if abs(x) < self.conf.centering_epsilon and abs(y) < self.conf.centering_epsilon:
                cv2.putText(
                    frame,
                    "CENTERING SUCCESS",
                    (40, 100),
                    font,
                    2,
                    (0, 255, 0),
                    2,
                    20,
                )

                # show video frame
                if self.conf.video.show_video:
                    self.detector.draw_detections(frame, corners, ids)
                    cv2.imshow("Video Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.clean_up()
        logger.info("Shutting down gracefully")




    '''
if __name__ == "__main__":
    config: SimpleNamespace = read_json("config.json")

    auto_uav = Uav(config)
    auto_uav.run()