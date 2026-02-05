"""Initizalizes the OAK SD-2 Pipeline"""

import depthai as dai
from dataclasses import dataclass

'''
WE ARE USING VERSION 3, here are some things to remember:
   1.  Recall that ".mono/.rgb" camera has been switched to just one unified node ".Camera"

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
   #align_to_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_B
   output_rectified = False

   # wueues are created on host side)
   # max_q_size: int = 4
   depth_stream_name = "depth"
   disparity_stream_name = "disparity"
   rectified_left_stream= "rectified_left"
   rectified_right_stream = "rectified_right"


def create_pipe():
   pipeline = dai.Pipeline()
   st = HoldOakValues()
   #stereo_enabled = st.enable_stereo_depth

   if st.enable_stereo_depth:
      mono_left = pipeline.create(dai.node.Camera)
      mono_left.build(dai.CameraBoardSocket.LEFT)
      mono_left.setFps(st.mono_fps)

      mono_right = pipeline.create(dai.node.Camera)
      mono_right.build(dai.CameraBoardSocket.RIGHT)
      mono_right.setFps(st.mono_fps)

      # Our stereo node
      cam_stereo = pipeline.create(dai.node.StereoDepth)
      cam_stereo.setLeftRightCheck(st.lr_disparity_check)
      cam_stereo.setExtendedDisparity(st.extend_disparity)
      
      cam_stereo.initialConfig.setMedianFilter(st.median)
      cam_stereo.setSubpixel(st.measure_subpixel)

      mono_left_out = mono_left.requestFullResolutionOutput()
      mono_right_out = mono_right.requestFullResolutionOutput()
      # You can use debugDispLrCheckIt1 and debugDispLrCheckIt2
      # debug outputs for debugging/fine-tuning purposes

      mono_left_out.link(cam_stereo.left)
      mono_right_out.link(cam_stereo.right)

      # If we want the depth data, output stream
      # This ONLY creates our "usbs' output stream" to route depth frames from OAK to Pi
      if st.output_depth:
         xout_depth = pipeline.create(dai.node.XLinkOut)
         xout_depth.setStreamName(st.depth_stream_name)
         cam_stereo.depth.link(xout_depth.input)

      if st.output_disparity:
         xout_disparity = pipeline.create(dai.node.XLinkOut)
         xout_disparity.setStreamName(st.disparity_stream_name)
         cam_stereo.disparity.link(xout_disparity.input)

      if st.output_rectified:
         xout_left = pipeline.create(dai.node.XLinkOut)
         xout_left.setStreamName(st.rectified_left_stream)
         cam_stereo.rectifiedLeft.link(xout_left.input)

         xout_right = pipeline.create(dai.node.XLinkOut)
         xout_right.setStreamName(st.rectified_right_stream)
         cam_stereo.rectifiedRight.link(xout_right.input)


      return pipeline

def main():
   pipeline = create_pipe()
   pass

if __name__ == "__main__":
   main()

