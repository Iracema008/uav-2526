""" Initizalizes the OAK SD-2 Pipeline w/SpectacularAI VIO """

import depthai as dai
import spectacularAI
from dataclasses import dataclass
from typing import Optional, Tuple

'''
WE ARE USING VERSION 3, here are some things to remember:
   1.  Recall that ".mono/.rgb" camera has been switched to just one unified node ".Camera"
   2.  With SpectacularAI: let it build/own the VIO pipeline (the vstereo & IMU), then we can add outputs if needed
'''

@dataclass(frozen=True)
class HoldOakValues:
   mono_frame_size = (1280,720)
   mono_fps = 30.0

   enable_stereo_depth = True
   mono_frame_size= (1280, 720)
   mono_fps= 30.0

   output_depth= True
   output_disparity = False

   lr_disparity_check = False
   extend_disparity= False
   # maybe turn this to True?
   measure_subpixel = False
   # Median filtering is disabled when subpixel mode is set to 4 or 5 bits.
   median: dai.StereoDepthProperties.MedianFilter = (
      dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
   )

   output_rectified: bool = False
   # max_q_size: int = 4
   depth_stream_name = "depth"
   disparity_stream_name = "disparity"
   rectified_left_stream= "rectified_left"
   rectified_right_stream = "rectified_right"


def set_mono_resolution(camera_node: dai.node.Camera, size: Tuple[int, int]) -> None:
   w, h = size
   if (w, h) == (1280, 720):
      camera_node.setSensorResolution(dai.CameraProperties.SensorResolution.THE_720_P)

   elif (w, h) == (640, 400):
      camera_node.setSensorResolution(dai.CameraProperties.SensorResolution.THE_400_P)
   elif (w, h) == (640, 480):
      camera_node.setSensorResolution(dai.CameraProperties.SensorResolution.THE_480_P)
   else:
      camera_node.setSensorResolution(dai.CameraProperties.SensorResolution.THE_720_P)


def create_pipe():
   pipeline = dai.Pipeline()
   st = HoldOakValues()
   vio_config = spectacularAI.depthai.Configuration()
   # vio_config.useImu = True
   # vio_config.internalParameters

   vio_pipeline = spectacularAI.depthai.Pipeline(pipeline, vio_config)

   if st.enable_stereo_depth:
      if hasattr(vio_pipeline, "monoLeft"):
         try:
            vio_pipeline.monoLeft.setFps(st.mono_fps)
            set_mono_resolution(vio_pipeline.monoLeft, st.mono_frame_size)
         except Exception:
            pass

      if hasattr(vio_pipeline, "monoRight"):
         try:
            vio_pipeline.monoRight.setFps(st.mono_fps)
            set_mono_resolution(vio_pipeline.monoRight, st.mono_frame_size)
         except Exception:
            pass


   stereo_node = None

   for attr in ("stereoDepth", "stereo", "stereo_node"):
      if hasattr(vio_pipeline, attr):
         stereo_node = getattr(vio_pipeline, attr)
         break

   # If we can access a stereo node, attach outputs
   if stereo_node is not None:
      try:
         stereo_node.setLeftRightCheck(st.lr_disparity_check)
         stereo_node.setExtendedDisparity(st.extend_disparity)
         stereo_node.setSubpixel(st.measure_subpixel)
         stereo_node.initialConfig.setMedianFilter(st.median)
      except Exception:
         pass

      if st.output_depth and hasattr(stereo_node, "depth"):
         xout_depth = pipeline.create(dai.node.XLinkOut)
         xout_depth.setStreamName(st.depth_stream_name)
         stereo_node.depth.link(xout_depth.input)

      if st.output_disparity and hasattr(stereo_node, "disparity"):
         xout_disp = pipeline.create(dai.node.XLinkOut)
         xout_disp.setStreamName(st.disparity_stream_name)
         stereo_node.disparity.link(xout_disp.input)

      if st.output_rectified:
         if hasattr(stereo_node, "rectifiedLeft"):
            xout_left = pipeline.create(dai.node.XLinkOut)
            xout_left.setStreamName(st.rectified_left_stream)
            stereo_node.rectifiedLeft.link(xout_left.input)

         if hasattr(stereo_node, "rectifiedRight"):
            xout_right = pipeline.create(dai.node.XLinkOut)
            xout_right.setStreamName(st.rectified_right_stream)
            stereo_node.rectifiedRight.link(xout_right.input)

   # queues are created on host side (device.py)
   return pipeline, vio_pipeline, st


def main():
   pass
