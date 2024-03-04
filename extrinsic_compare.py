import json

from multical.rectification.rectification import *

def refactor_calibration_poses():
    calibrationDataPath = "captureSimulation\calibration.json"
    multical_cameras = read_calibration_data(calibrationDataPath)

    camera_poses = [{"camera_id": cam.name, "R": cam.R.flatten().tolist(), "T": cam.T.flatten().tolist()} for cam in multical_cameras]

    #calibrationdataPath = "C:\Users\Janelle Pfeifer\Desktop\\2023_12_15_wave.json"

    with open("captureSimulation\calibration_.json", "w") as f:
        json.dump({"camera_poses": camera_poses},f)

refactor_calibration_poses()
