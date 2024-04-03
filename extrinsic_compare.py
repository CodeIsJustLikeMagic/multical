import json

from multical.rectification.rectification import *
from multical import tables
from multical.workspace import Workspace
from structs.struct import struct, split_dict
import pathlib
import os


def read_from_workspace(path):
    ws = Workspace.load(path)
    _, calibs = split_dict(ws.calibrations)
    calibration = calibs[1]  # <- use the calibrated positions (1) or the initialization positions (0)

    def camera_parameter_view_cam_params():
        # data displayed in vis camera_param view:
        poses = calibration.camera_poses.pose_table
        poses = tables.inverse(poses)

        # adjust for master camera
        masterindex = 0
        inv = np.linalg.inv(poses.poses[masterindex])
        final = poses._extend(poses=poses.poses @ np.expand_dims(inv, 0))
        return final

    def calibrationJson_cam_prams(): # outputs just like Json file (opencv coordinate system)
        poses = calibration.camera_poses.pose_table
        inv = np.linalg.inv(poses.poses[0])
        finale = poses._extend(poses=poses.poses @ np.expand_dims(inv, 0))
        # looks similar to display, values by column major and with -t1
        return finale

    def camera_mesh_set():  # MovingBoard
        camera_poses = tables.inverse(calibration.pose_estimates.camera)
        return camera_poses

    def board_mesh_set():  # MovingBoard
        board_poses = tables.expand_boards(calibration.pose_estimates)
        return board_poses
        # board_poses = board_poses._sequence()

    # print("cam parameter viewer", camera_parameter_view_cam_params().poses[6])
    # print("calibrationJson", calibrationJson_cam_prams().poses[6])
    # print("camera mesh set position", camera_mesh_set().poses[6])
    # print("board mesh set position", board_mesh_set().poses[3])
    np.set_printoptions(suppress=True, precision=4)
    names = calibration.camera_poses.names
    camera_poses = camera_mesh_set()#calibrationJson_cam_prams()  # calibration.camera_poses
    opencv_trans = np.array([[1, -1, -1, -1],
                             [-1, 1, -1, -1],
                             [-1, -1, 1, 1],
                             [1, 1, 1, 1]])
    poses = calibration.camera_poses.pose_table
    inv = np.linalg.inv(poses.poses[0])
    cam_poses = camera_poses.poses @ np.expand_dims(inv, 0)

    valid_camera_poses = {name: extrinsic for name, extrinsic, valid in
                          zip(names, cam_poses, camera_poses.valid)}
    camera_view_matrix_json = [{"camera_id": name,
                                "extrinsics": {"view_matrix": valid_camera_poses[name].flatten().tolist()},
                                "intrinsics": {"camera_matrix": camera.intrinsic.flatten().tolist(),
                                               "resolution": camera.image_size,
                                               "distortion_coefficients": camera.dist.flatten().tolist()}}
                               for name, camera
                               in zip(names, calibration.cameras.param_objects)]

    board_poses_2 = tables.expand_boards(calibration.pose_estimates)
    board_poses_2 = [poses for poses in
                     board_poses_2._sequence()]  # each item is a list of poses for each board in boards.yaml.
    # each item has a poses and a valid array. The valid array determines which of the poses is valid. If a board cannot be seen it doesn't have a valid pose.
    # each item (poselist) exists for each mutlicam frame (Each camera takes an image at the same time, together they form one multicam frame, if we want to call it that idk)
    pose_entry_multicamframe1 = board_poses_2[0]
    poses_for_each_board_multicamframe1 = pose_entry_multicamframe1.poses

    # transform by camera 0 position, idk if it is nessesary
    poses = calibration.camera_poses.pose_table
    opencv_trans = np.array([[1, -1, -1, -1],
                             [-1, 1, -1, -1],
                             [-1, -1, 1, 1],
                             [1, 1, 1, 1]])
    inv = np.linalg.inv(poses.poses[0])
    #poses_for_each_board_multicamframe1 = poses_for_each_board_multicamframe1 @ np.expand_dims(inv, 0)

    # adjust to opencv coordinate system. seems like we just need to switch a few signs

    #poses_for_each_board_multicamframe1 = np.multiply(poses_for_each_board_multicamframe1, opencv_trans)

    print("board poses multicamframe1", poses_for_each_board_multicamframe1, "board poses end")

    board_poses_json = [{"board_name": name, "extrinsics": {"view_matrix": extrinsicMatrix.flatten().tolist()}}
                        for name, extrinsicMatrix, valid
                        in zip(board_poses.names, poses_for_each_board_multicamframe1, board_poses.valid)
                        if valid]

    with open("./captureSimulation2/calibration_.json", "w") as f:
        json.dump({"meta": "camera and board poses relative to camera 0",
                   "cameras": camera_view_matrix_json,
                   "boards": board_poses_json}, f, indent=2)


calibrationDataPath = "./captureSimulation2/calibration.pkl"
read_from_workspace(calibrationDataPath)
