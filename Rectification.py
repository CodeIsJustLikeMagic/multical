from dataclasses import dataclass


import multical
from multical.config.arguments import run_with
from multiprocessing import cpu_count
from typing import Union

from multical.app.boards import Boards
from multical.app.calibrate import Calibrate
from multical.app.intrinsic import Intrinsic
from multical.app.vis import Vis

import cv2
def rectify():
    #cameraMatrix1 =
    cv2.stereoRectify()
    pass


if __name__ == '__main__':
    # import required module
    rectify()
