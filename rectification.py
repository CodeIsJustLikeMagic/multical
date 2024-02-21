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

    stereo = cv2.StereoSGBM.create(minDisparity=0,
                                   numDisparities=128,
                                   blockSize=21,
                                   disp12MaxDiff=0,
                                   uniquenessRatio=10,
                                   speckleWindowSize=0,
                                   speckleRange=0
                                   )
    stereo.setMinDisparity(0)
    stereo.setUniquenessRatio(10)
    stereo.setDisp12MaxDiff(1)
    stereo.setSpeckleWindowSize(10)
    stereo.setSpeckleRange(8)
    disparity = stereo.compute(img_base, img_match)
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imwrite("disparity calibrated_python.jpg", disparity)
    return disparity


def camera_dispartiy_map2(imgL, imgR):
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


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
    return img1, img2


def epipolarlines(img1, img2):
    sift = cv2.SIFT.create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    error = 0
    for j in range(len(pts1)):
        errorij = (abs(pts1[j][0] * lines2[j][0] +
                       pts1[j][1] * lines2[j][1] + lines2[j][2]) +
                   abs(pts2[j][0] * lines1[j][0] +
                       pts2[j][1] * lines1[j][1] + lines1[j][2]))

        error += errorij
    print(error / len(pts1))
    return img5, img3


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
cv2.imwrite("./cam0_rectified.png", rectifyBase)
cv2.imwrite("./cam1_rectified.png", rectifyMatch)

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


# debug_rectification_images()

cv2.imshow("rectify base", cv2.resize(rectifyBase, smol))
cv2.imshow("rectify match", cv2.resize(rectifyMatch, smol))

if True:
    leftlines, rightlines = epipolarlines(rectifyBase, rectifyMatch)
    cv2.imshow("epipolarlines base", cv2.resize(leftlines, smol))
    cv2.imshow("epipolarlines match", cv2.resize(rightlines, smol))

    baselines_orig, matchlintes_orig = epipolarlines(frameBase, frameMatch)
    cv2.imshow("epipolarlines base original", cv2.resize(baselines_orig, smol))
    cv2.imshow("epipolarlines match original", cv2.resize(matchlintes_orig, smol))

# disparity_recitifed = cameraDisparityMap(rectifyBase, rectifyMatch)
disparity_recitifed = camera_disparity_map(rectifyBase, rectifyMatch)
cv2.imshow("disparityMap_rectified", cv2.resize(disparity_recitifed, smol))
disparity_original = camera_disparity_map(frameBase, frameMatch)
cv2.imshow("disparityMap_original", cv2.resize(disparity_original, smol))
cv2.waitKey(0)
cv2.destroyAllWindows()

def nothing(x):
    pass


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)

cv2.createTrackbar('numDisparities', 'disp', 1, 128, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

while True:

    # Capturing and storing left and right camera images
    Left_nice = rectifyBase
    Right_nice = rectifyMatch

    # Proceed only if the frames have been captured
    if True:
        #imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        #imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        Left_nice = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
        Right_nice = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)


        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice, Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Displaying the disparity map
        cv2.imshow("disp", disparity)

        # Close window using esc key
        if cv2.waitKey(1) == 27:
            #break
            cv2.destroyAllWindows()
            exit()

    #else:
    #    CamL = cv2.VideoCapture(CamL_id)
    #    CamR = cv2.VideoCapture(CamR_id)


