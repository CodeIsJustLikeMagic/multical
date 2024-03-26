import cv2
import numpy as np
from multical.config import find_board_config
from multical.image import detect
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
    cv2.imwrite("../../disparity calibrated_python.jpg", disparity)


    # dispariy map
    if True:

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

        # Capturing and storing left and right camera images
        Left_nice = img_base
        Right_nice = img_match

        # imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        # imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
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
            # break
            cv2.destroyAllWindows()





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


def epipolarlines(img1, img2, pts1, pts2):
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

    def findAndMatchFeatures(img1, img2):
        w, h, _ = img1.shape
        smol = np.array([w / 4, h / 4], dtype=int)
        img1 = cv2.resize(img1, smol)
        img2 = cv2.resize(img2, smol)
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)
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
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                matchesMask[i] = [1, 0]
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        # Draw the keypoint matches between both pictures
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask[100:500],
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        keypoint_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[100:500], None, **draw_params)

        cv2.namedWindow("Keypoint matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Keypoint matches", 2000, 1000)
        cv2.imshow("Keypoint matches", keypoint_matches)

        cv2.waitKey(0)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]


    indc = np.array(range(0, len(pts1)), dtype=int)
    sample_inds = np.random.choice(indc, (int)(len(pts1)/3), replace=False)
    print(sample_inds)
    pts1 = pts1[sample_inds]
    pts2 = pts2[sample_inds]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # print("yFx for each pair of points, should be 0 for all", np.transpose(pts2) * F * pts1)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    cv2.namedWindow("Epipolarlines", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Epipolarlines", 2000, 1000)
    cv2.imshow("Epipolarlines", cv2.hconcat([img5, img3]))

    error = 0
    for j in range(len(pts1)):
        errorij = (abs(pts1[j][0] * lines2[j][0] +
                       pts1[j][1] * lines2[j][1] + lines2[j][2]) +
                   abs(pts2[j][0] * lines1[j][0] +
                       pts2[j][1] * lines1[j][1] + lines1[j][2]))

        error += errorij
    print(error / len(pts1))
    return img5, img3

def ShowFeaturePairs(img1, img2, pts1, pts2):
    kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for i, pt in enumerate(pts1)]
    kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for i, pt in enumerate(pts2)]

    matches = [cv2.DMatch(i, i, _distance=10, _imgIdx=0) for i in range(len(pts1))]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0))
    featuremap = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    windowname = f"Feature Pairs (Average Difference in y direction {CalculateYDifferenceForPointMatches(pts1, pts2)})"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowname, 2000, 1000)
    cv2.imshow(windowname, featuremap)

def CalculateYDifferenceForPointMatches(pts1, pts2, verbos = True):
    assert len(pts1) == len(pts2)
    diff_per_point_pair = np.array([abs(pt1[1] - pt2[1]) for pt1, pt2 in zip(pts1, pts2)])
    avg = np.average(diff_per_point_pair)
    std = np.std(diff_per_point_pair)
    if verbos:
        print(f"Average Difference in y direction: {avg} pixels, std {std}")
    return avg


def CharucoBoardDetection(imgBase, imgMatch,  board_file, showCorners = False,):
    boards = find_board_config("", board_file=board_file).values()
    board_detection_base = detect.detect_image(imgBase, boards)
    board_detection_match = detect.detect_image(imgMatch, boards)

    if showCorners:
        import multical.display as display
        img_with_corners_base = display.draw_detections(imgBase, board_detection_base)
        img_with_corners_match = display.draw_detections(imgMatch, board_detection_match)
        cv2.namedWindow("corners base", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("corners base", 2000, 1000)
        cv2.imshow("corners base", cv2.hconcat([img_with_corners_base, img_with_corners_match]))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # find common corners
    board_detection_base = board_detection_base[0]
    board_detection_match = board_detection_match[0]
    common_corner_ids = [id for id in board_detection_base["ids"] if id in board_detection_match["ids"]]
    pts_base = np.array(
        [pt for pt, id in zip(board_detection_base["corners"], board_detection_base["ids"]) if id in common_corner_ids])
    pts_match = np.array(
        [pt for pt, id in zip(board_detection_match["corners"], board_detection_match["ids"]) if id in common_corner_ids])
    return pts_base, pts_match