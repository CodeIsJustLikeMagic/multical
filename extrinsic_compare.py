import json

from multical.rectification.rectification import *

def read_from_workspace(path):
    from multical.workspace import Workspace
    from structs.struct import struct, split_dict
    import pathlib
    import os

    ws = Workspace.load(path)

    _, calibs = split_dict(ws.calibrations)
    calibration = calibs[0]
    camera_poses = calibration.camera_poses

    valid_camera_poses = {name: extrinsic for name,extrinsic, valid in zip(camera_poses.names, camera_poses.poses, camera_poses.valid)}
    camera_view_matrix_json = [{"camera_id": name,
                                "extrinsics": {"view_matrix": valid_camera_poses[name].flatten().tolist()},
                                "intrinsics": {"camera_matrix": camera.intrinsic.flatten().tolist(),
                                                "resolution": camera.image_size,
                                               "distortion_coefficients": camera.dist.flatten().tolist()}}
                         for name, camera
                         in zip(calibration.cameras.names, calibration.cameras.param_objects)]

    board_poses = calibration.board_poses
    board_poses_json = [{"board_name": name, "extrinsics": {"view_matrix": extrinsicMatrix.flatten().tolist()}}
                        for name, extrinsicMatrix, valid
                        in zip(board_poses.names, board_poses.poses, board_poses.valid)
                        if valid]
    from multical import tables
    board_poses_2 = tables.expand_boards(calibration.pose_estimates)
    board_poses_2 = [poses for poses in board_poses_2._sequence()] # gets 4 poses per board

    camera_poses_2 = tables.inverse(calibration.pose_estimates.camera)

    with open("./captureSimulation/calibration_.json", "w") as f:
        json.dump({"meta": "camera and board poses relative to camera 0",
                   "cameras": camera_view_matrix_json,
                   "boards": board_poses_json}, f)


calibrationDataPath = "./captureSimulation/calibration.pkl"
read_from_workspace(calibrationDataPath)


