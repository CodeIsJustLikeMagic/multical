import json

with open("calibration.json") as f:
    data = json.load(f)

cameras_j = data['cameras']
print(cameras_j)

poses = data['camera_poses']

print(poses.keys())

cameras = [(camera, cameras_j[camera]['K'], cameras_j[camera]["dist"], poses[pose]["R"], poses[pose]["T"]) for camera,pose in zip(cameras_j.keys(), poses.keys())]
print(cameras[0])
from multical.workspace import Workspace
from structs.struct import struct, split_dict
import pathlib
import os

ws = Workspace.load(os.path.join(os.getcwd(), "calibration.pkl"))

_, calibs = split_dict(ws.calibrations)
calibration = calibs[0]
poses = calibration.camera_poses.pose_table
print(poses)
print("Cam0 intrinsic matrix: " , calibration.cameras[0].intrinsic)
import cv2

#cv2.initUndistortRectifyMap(calibration.cameras[0].intrinsic)

#https://stackoverflow.com/questions/18804182/rectifying-images-on-opencv-with-intrinsic-and-extrinsic-parameters-already-foun
