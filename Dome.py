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


@dataclass
class Multical:
    """multical - multi camera calibration
    - calibrate: multi-camera calibration
    - intrinsic: calibrate separate intrinsic parameters
    - boards: generate/visualize board images, test detections
    - vis: visualize results of a calibration
    """
    command: Union[Calibrate, Intrinsic, Boards, Vis]

    def execute(self):
        return self.command.execute()


def cli():
    run_with(Multical)

def pre_process():
    import os

    # assign directory
    directory = 'calibrationImages'

    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            camindex = (f.split("cam")[-1]).split("frame")[0]
            frameindex = (f.split("frame")[-1]).split(".png")[0]
            image = cv2.imread(f)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE, image)
            cv2.imwrite("cam"+camindex+"\\"+"image"+frameindex+".png", image)


if __name__ == '__main__':
    # import required module
    import os
    print(os.getcwd())
    cli()
