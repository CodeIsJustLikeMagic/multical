import json
from os import path
import numpy as np

from structs.struct import struct, to_dicts, transpose_lists
from multical.transform import matrix


def export_camera(camera):
    return struct(
        model=camera.model,
        image_size=camera.image_size,
        K=camera.intrinsic.tolist(),
        dist=camera.dist.tolist()
    )


def export_cameras(camera_names, cameras):
    return {k: export_camera(camera) for k, camera in zip(camera_names, cameras)}


def export_transform(pose):
    '''
    splits an extrinsicmatrix (position) into Rotationmatrix and Translationvector
    '''
    r, t = matrix.split(pose)
    return struct(R=r.tolist(), T=t.tolist())


def export_camera_poses(camera_names, camera_poses):
    '''compiles a dictionary of camera poses (RotationMatrix and TranslationVector)'''
    return {k: export_transform(pose)
            for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid)
            if valid}


def export_relative(camera_names, camera_poses, master):
    '''
  compiles a dictionary of camera poses (RotationMatrix and TranslationVector) where poses are relative to a master camera. Does not perform transfomration on the camera poses
  '''
    assert master in camera_names

    return {k if master == k else f"{k}_to_{master}": export_transform(pose)
            for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid)
            if valid}


def export_sequential(camera_names, camera_poses):
    transforms = {camera_names[0]: export_transform(np.eye(4))}
    poses = camera_poses.poses

    for i in range(1, len(camera_names)):
        k = f"{camera_names[i]}_to_{camera_names[i - 1]}"
        transforms[k] = export_transform(
            poses[i] @ np.linalg.inv(poses[i - 1]))  # transforms pose acording to previous cameras position

    return transforms


def export_poses(pose_table, names=None):
    names = names or [str(i) for i in range(pose_table._size[0])]

    return {i: t.poses.tolist() for i, t in zip(names, pose_table._sequence())
            if t.valid}


def export_images(camera_names, filenames):
    return struct(
        rgb=[{camera: image for image, camera in zip(images, camera_names)}
             for images in filenames]
    )


def export_single(filename, cameras, camera_names, filenames):
    filenames = transpose_lists(filenames)
    data = struct(
        cameras=export_cameras(camera_names, cameras),
        image_sets=export_images(camera_names, filenames)
    )

    with open(filename, 'w') as outfile:
        json.dump(to_dicts(data), outfile, indent=2)


def export(filename, calib, names, filenames, master=None):
    data = export_json(calib, names, filenames, master=master)

    with open(filename, 'w') as outfile:
        json.dump(to_dicts(data), outfile, indent=2)


def export_json(calib, names, filenames, master=None):
    if master is not None:
        calib = calib.with_master(master)

    camera_poses = calib.camera_poses.pose_table
    filenames = transpose_lists(filenames)

    data = struct(
        cameras=export_cameras(names.camera, calib.cameras),
        # camera_poses = export_sequential(names.camera, camera_poses),
        camera_poses=export_camera_poses(names.camera, camera_poses) \
            if master is None else export_relative(names.camera, camera_poses, master),
        image_sets=export_images(names.camera, filenames)

    )

    return data


def export_json_domeformat(calib, names, filenames, master=None):
    if master is not None:
        calib = calib.with_master(master)  # transforms camera poses

    camera_poses = calib.camera_poses.pose_table  # calibration.camera_poses
    camera_poses = {name: extrinsic for name, extrinsic in
                          zip(names.camera, camera_poses.poses)}
    camera_view_matrix_json = [{"camera_id": name,
                                "extrinsics": {"view_matrix": camera_poses[name].flatten().tolist()},
                                "intrinsics": {"camera_matrix": camera.intrinsic.flatten().tolist(),
                                               "resolution": camera.image_size,
                                               "distortion_coefficients": camera.dist.flatten().tolist()}}
                               for name, camera
                               in zip(names.camera, calib.cameras)]

    master_info = "" if master is None else "Camera poses relative to camera " + master + ". "

    return {"meta": master_info + "Extrinsics use OpenCV coordinate system (left-hand, x left, z forward, y down)",
            "cameras": camera_view_matrix_json}


def export_camera_and_boards(calib, names, master=None):
    from multical import tables
    if master is not None:
        calib = calib.with_master(master)

    camera_poses = tables.inverse(calib.pose_estimates.camera)
    board_poses = tables.expand_boards(calib.pose_estimates)  # set of board poses for each image pair
    board_poses = struct(poses=board_poses.poses[0], valid=board_poses.valid[0])  # board poses for first image pair
    camnames = names.camera
    boardnames = names.board
    if camnames is None:
        camnames = [str(indx) for indx in range(len(camera_poses.poses))]
    if boardnames is None:
        boardnames = [str(indx) for indx in range(len(board_poses.poses))]

    valid_camera_poses = {name: extrinsic for name, extrinsic in
                          zip(camnames, camera_poses.poses)}
    camera_view_matrix_json = [{"camera_id": name,
                                "extrinsics": {"view_matrix": valid_camera_poses[name].flatten().tolist()},
                                "intrinsics": {"camera_matrix": camera.intrinsic.flatten().tolist(),
                                               "resolution": camera.image_size,
                                               "distortion_coefficients": camera.dist.flatten().tolist()}}
                               for name, camera, valid
                               in zip(camnames, calib.cameras, camera_poses.valid) if valid]

    board_poses_json = [{"board_name": name, "extrinsics": {"view_matrix": extrinsicMatrix.flatten().tolist()}}
                        for name, extrinsicMatrix, valid
                        in zip(boardnames, board_poses.poses, board_poses.valid)
                        if valid]

    return {"meta": "camera and board poses relative to camera 0",
            "cameras": camera_view_matrix_json,
            "boards": board_poses_json}
