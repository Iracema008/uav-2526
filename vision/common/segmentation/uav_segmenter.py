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

    bg_kernel: int = 5
    obstacle_kernel: int = 7
    bg_open_iter: int = 1
    bg_close_iter: int = 2
    ob_open_iter: int = 1
    ob_close_iter: int = 2

    grid_h: int = 40
    grid_w: int = 40
    occ_thresh: float = 0.03

    process_interval_sec: float = 0.5 # this is the "still snap every .5 secs"

    def _odd(k: int) -> int:
        return k if k % 2 == 1 else k + 1


class FieldObstacleSegmenter: