import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

@dataclass
class SegmenterConfig: # we're doing green and white since the football field will be green with white lines
    green_lo: Tuple[int, int, int] = (35, 40, 40)
    green_hi: Tuple[int, int, int] = (90, 255, 255)

    white_lo: Tuple[int, int, int] = (0, 0, 180)
    white_hi: Tuple[int, int, int] = (179, 60, 255)

    g_kernel: int = 5
    obstacle_kernel: int = 7
    bg_open_iter: int = 1
    bg_close_iter: int = 2
    ob_open_iter: int = 1
    ob_close_iter: int = 2