'''Initizalizes the OAK SD-2 Pipeline'''

import depthai as dai
from dataclasses import dataclass


@dataclass(frozen=True)
class HoldOakValues:
    #rgb_frame_size = (1280,720)
    #rgp_fps = 30.0

    enable_stereo_depth = True
    mono_frame_size= (1280, 720)
    mono_fps= 30.0

    output_depth= True
    output_disparity = False

    lr_disparity_check = False
    extend_disparity= False
    measure_subpixel = False

    # Median filtering is disabled when subpixel mode is set to 4 or 5 bits.
    median: dai.StereoDepthProperties.MedianFilter = (
       dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    )
    align_to_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_B
    output_rectified: bool = False

    # wueues are created on host side)
    # max_q_size: int = 4