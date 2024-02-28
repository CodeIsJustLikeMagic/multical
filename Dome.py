from dataclasses import dataclass


import multical
from multical.config.arguments import run_with
from multiprocessing import cpu_count
from typing import Union

from multical.app.boards import Boards
from multical.app.calibrate import Calibrate
from multical.app.intrinsic import Intrinsic
from multical.app.rectify import Rectify
from multical.app.vis import Vis

from multical import workspace

import cv2


@dataclass
class Multical:
    """multical - multi camera calibration
    - calibrate: multi-camera calibration
    - intrinsic: calibrate separate intrinsic parameters
    - boards: generate/visualize board images, test detections
    - vis: visualize results of a calibration
    - rectify: rectify images after a calibration
    """
    command: Union[Calibrate, Intrinsic, Boards, Vis, Rectify]

    def execute(self):
        return self.command.execute()


def cli():
    run_with(Multical)

def pre_process():
    import os

    # assign directory
    in_directory = 'calibrationImages'
    out_directory = 'fixedCalibrationImages'
    rotate = True
    # iterate over files in
    # that directory
    for filename in os.listdir(in_directory):
        f = os.path.join(in_directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            camindex = (f.split("cam")[-1]).split("frame")[0]
            frameindex = (f.split("frame")[-1]).split(".png")[0]
            image = cv2.imread(f)
            if rotate:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE, image)
            cv2.imwrite(out_directory+"cam"+camindex+"\\"+"image"+frameindex+".png", image)


if __name__ == '__main__':
    #calibrate --cameras cam0 cam1 --boards charuco_small.yaml
    #vis --workspace_file calibration.pkl
    #rectify
    #rectify --cameras cam0 cam1 --output_path rectified_images --limit_images 3 --boards charuco_small.yaml
    cli()

    #calibrate --image_path ./captureImages_tricams --cameras cam0 cam1 cam2 --boards charuco_small.yaml
    #vis --workspace_file ./captureImages_tricams/calibration.pkl
    #rectify --cameras cam0 cam1 cam2 --output_path rectified_images --limit_images 3 --boards charuco_small.yaml --image_path ./captureImages_tricams


    #boards --boards ./captureSimulation/charuco_Dome.yaml --detect ./captureSimulation/cam0/image0.png
    #calibrate --image_path ./captureSimulation --cameras 0 1 2 3 4 --boards ./captureSimulation/charuco_Dome.yaml
