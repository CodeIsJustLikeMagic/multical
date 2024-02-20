import json
from collections import namedtuple
import numpy as np
import cv2

Camera = namedtuple('Camera', 'name intrinsic distCoeffs R T resolution map1 map2')


def read_calibration_data():
    with open("calibration.json") as f:
        data = json.load(f)

    cameras_j = data['cameras']
    poses = data['camera_poses']
    cameras = [Camera(camera, np.array(cameras_j[camera]['K']),
                      np.array(cameras_j[camera]["dist"]), np.array(poses[pose]["R"]),
                      np.array(poses[pose]["T"]), np.array(cameras_j[camera]['image_size']), None, None)
               for camera, pose in zip(cameras_j.keys(), poses.keys())]
    return cameras


def read_from_workspace():
    from multical.workspace import Workspace
    from structs.struct import struct, split_dict
    import pathlib
    import os

    ws = Workspace.load(os.path.join(os.getcwd(), "calibration.pkl"))

    _, calibs = split_dict(ws.calibrations)
    calibration = calibs[0]
    poses = calibration.camera_poses.pose_table
    print(poses)
    print("Cam0 intrinsic matrix: ", calibration.cameras[0].intrinsic)


def camera_disparity_map(img_base, img_match):
    img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM.create(numDisparities=64, blockSize=7)
    stereo.setMinDisparity(0)
    stereo.setUniquenessRatio(10)
    stereo.setDisp12MaxDiff(1)
    stereo.setSpeckleWindowSize(10)
    stereo.setSpeckleRange(8)
    disparity = stereo.compute(img_base, img_match)
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return disparity


def camera_dispartiy_map2(imgL, imgR):
    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM.create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...', )
    h, w = imgL.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    # out_fn = 'out.ply'
    # write_ply(out_fn, out_points, out_colors)
    # print('%s saved' % out_fn)

    return (disp - min_disp) / num_disp


class Rectification:
    def __init__(self, camBase, camMatch):
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
        rectify_base = cv2.remap(frame_base, self.map1_base, self.map2_base, cv2.INTER_NEAREST)
        rectify_match = cv2.remap(frame_match, self.map1_match, self.map2_match, cv2.INTER_NEAREST)
        return rectify_base, rectify_match

    def simple_undistort(self, frame_base, frame_match):
        new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camBase.intrinsic, self.camBase.distCoeffs,
                                                          self.camBase.resolution, 0)
        undistorted_base = cv2.undistort(frame_base, self.camBase.intrinsic, self.camBase.distCoeffs, new_cam_matrix)
        undistorted_match = cv2.undistort(frame_match, self.camMatch.intrinsic, self.camMatch.distCoeffs,
                                          new_cam_matrix)
        return undistorted_base, undistorted_match


def plot_image_grid(images, ncols=None, cmap='gray'):
    import matplotlib.pyplot as plt
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()):
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)


cameras = read_calibration_data()

camBase = cameras[0]
camMatch = cameras[1]

rectification = Rectification(camBase, camMatch)

frameBase = cv2.imread("./cam0/image00.png")
frameMatch = cv2.imread("./cam1/image00.png")
rectifyBase, rectifyMatch = rectification.apply(frameBase, frameMatch)
undistortBase, undistortMatch = rectification.simple_undistort(frameBase, frameMatch)

smol = np.array(camBase.resolution / 6, dtype=int)


def debug_rectification_images():
    cv2.imshow("input frame base", cv2.resize(frameBase, smol))
    cv2.imshow("undistort base", cv2.resize(undistortBase, smol))
    cv2.imshow("undistort base diff", cv2.resize(frameBase - undistortBase, smol))
    cv2.imshow("rectify base", cv2.resize(rectifyBase, smol))
    cv2.imshow("rectify difference base", cv2.resize(frameBase - rectifyBase, smol))
    cv2.imshow("rectify undisrort diff base", cv2.resize(undistortBase - rectifyBase, smol))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plot_image_grid([frameBase, undistortBase, frameBase, rectifyBase, frameBase-rectifyBase, undistortBase-rectifyBase])

    cv2.imshow("input frame match", cv2.resize(frameMatch, smol))
    cv2.imshow("undistorted frame match", cv2.resize(rectifyMatch, smol))
    cv2.imshow("difference match", cv2.resize(frameMatch - rectifyMatch, smol))
    cv2.imshow("basic_undistort match", cv2.resize(undistortMatch, smol))

    cv2.imshow("basic_undistort match", cv2.resize(frameMatch - undistortMatch, smol))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#debug_rectification_images()

cv2.imshow("rectify base", cv2.resize(rectifyBase, smol))
cv2.imshow("rectify match", cv2.resize(rectifyMatch, smol))

# disparity_recitifed = cameraDisparityMap(rectifyBase, rectifyMatch)
disparity_recitifed = camera_disparity_map(rectifyBase, rectifyMatch)
cv2.imshow("disparityMap_rectified", cv2.resize(disparity_recitifed, smol))
disparity_original = camera_disparity_map(frameBase, frameMatch)
cv2.imshow("disparityMap_original", cv2.resize(disparity_original, smol))

cv2.waitKey(0)
cv2.destroyAllWindows()
