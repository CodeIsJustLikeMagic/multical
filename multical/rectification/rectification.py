import cv2
import json
from collections import namedtuple
import numpy as np
CameraParameters = namedtuple('Camera', 'name intrinsic distCoeffs R T resolution')

class Rectification:
    """
    This class performs image rectification given a Pair of CameraParameters.
    """
    def __init__(self, camBase: CameraParameters, camMatch: CameraParameters):
        self.camBase = camBase
        self.camMatch = camMatch
        R1, R2, P1, P2, Q, Roi1, Roi2 = cv2.stereoRectify(camBase.intrinsic, camBase.distCoeffs, camMatch.intrinsic,
                                                          camMatch.distCoeffs,
                                                          camBase.resolution, camMatch.R, camMatch.T,
                                                          alpha=1)  # , flags = cv2.CALIB_ZERO_DISPARITY)

        self.map1_base, self.map2_base = cv2.initUndistortRectifyMap(camBase.intrinsic, camBase.distCoeffs, R1, P1,
                                                                     camBase.resolution, cv2.CV_32FC1)

        self.map1_match, self.map2_match = cv2.initUndistortRectifyMap(camMatch.intrinsic, camMatch.distCoeffs, R2, P2,
                                                                       camMatch.resolution, cv2.CV_32FC1)

    def apply(self, frame_base, frame_match):
        """
        Performs image rectification and undistortion on a pair of images
        """
        rectify_base = cv2.remap(frame_base, self.map1_base, self.map2_base, cv2.INTER_NEAREST)
        rectify_match = cv2.remap(frame_match, self.map1_match, self.map2_match, cv2.INTER_NEAREST)
        return rectify_base, rectify_match

    def simple_undistort(self, frame_base, frame_match):
        """
        Performs only undistortion on a pair of images. Call apply instead to rectify and undistort.
        """
        new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camBase.intrinsic, self.camBase.distCoeffs,
                                                          self.camBase.resolution, 0)
        undistorted_base = cv2.undistort(frame_base, self.camBase.intrinsic, self.camBase.distCoeffs, new_cam_matrix)
        undistorted_match = cv2.undistort(frame_match, self.camMatch.intrinsic, self.camMatch.distCoeffs,
                                          new_cam_matrix)
        return undistorted_base, undistorted_match


# read calibration data from a source.
def read_calibration_data(calibration_json_path):
    with open(calibration_json_path) as f:
        data = json.load(f)

    cameras_j = data['cameras']
    poses = data['camera_poses']
    cameras = [CameraParameters(camera, np.array(cameras_j[camera]['K']),
                                np.array(cameras_j[camera]["dist"]), np.array(poses[pose]["R"]),
                                np.array(poses[pose]["T"]), np.array(cameras_j[camera]['image_size']))
               for camera, pose in zip(cameras_j.keys(), poses.keys())]
    return cameras


def read_calibration_data_domejson(calibration_json_path):
    from multical.transform import matrix
    with open(calibration_json_path) as f:
        data = json.load(f)

    cameras = [CameraParameters(camera["camera_id"], # name
                                np.array(camera["intrinsics"]["camera_matrix"]).reshape(3,3), # intrinsic matrix
                                np.array(camera["intrinsics"]["distortion_coefficients"]), # distortion coeffs
                                matrix.split(np.array(camera["extrinsics"]["view_matrix"]).reshape(4, 4))[0], # rotation matrix
                                matrix.split(np.array(camera["extrinsics"]["view_matrix"]).reshape(4, 4))[1], # translation vector
                                np.array(camera["intrinsics"]["resolution"])) # resolution
               for camera in data["cameras"]]
    return cameras

def read_calibration_data_pkl(calibration_pkl_path):
    from multical.workspace import Workspace
    from structs.struct import split_dict
    import numpy as np
    ws = Workspace.load(calibration_pkl_path)
    _, calibs = split_dict(ws.calibrations)
    calibration = calibs[1]
    cameras = calibration.cameras

    poses = calibration.camera_poses.pose_table
    inv = np.linalg.inv(poses.poses[0])
    camera_extrinsics = poses._extend(poses = poses.poses @ np.expand_dims(inv,0))

