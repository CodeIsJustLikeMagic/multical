import os.path

import numpy as np

from multical.rectification.rectification import *
from multical.rectification.verify_rectification import *
from multical.config import *
from multical.io.logging import setup_logging
import cv2


@dataclass
class Rectify:
    paths: PathOpts
    workspace_file: str = None
    debug: int = 0  # debug level, 0 is no debug, 1 is surface level debug, 2 is detailed debug

    def execute(self):
        rectify(self)


def show_image_stack(windowname, imagelist):
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowname, 2000, 1000)
    cv2.imshow(windowname, cv2.hconcat(imagelist))


def createOutputFileSystem(args):
    if args.paths.output_path == args.paths.image_path or args.paths.output_path is None:
        args.paths.output_path = args.paths.image_path + "_rectified"
    if not os.path.exists(args.paths.output_path):
        os.makedirs(args.paths.output_path)
    for c in args.paths.cameras:
        cam_path = os.path.join(args.paths.output_path, c)
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)


def rectify(
        args):  # rectify --image_path ./captureImages_tricams --cameras cam1 cam2 --calibration_json ./captureImages_tricams/calibration_dome.json
    """
    rectify images of two cameras. Adjusts the images so their epipolarlines align and are horizontal
    """

    print(f"Rectify for cameras: {args.paths.cameras}\n")  # --cameras cam0 cam1

    if len(args.paths.cameras) != 2:
        print(f"Please specify two cameras that are to be rectified with --cameras <cam1> <cam2>")
        return

    if args.workspace_file is not None:
        filename = args.workspace_file
        if path.isdir(filename):
            filename = path.join(filename, "calibration.pkl")
            print(f"Reading calibration file {filename}")
            cameraParameters = read_calibration_data_pkl(filename, args.paths.cameras[0])
            # read calibration data, with base camera as master (aka as origin of coordiante system for extrinsics)
    else:
        print(f"Reading calibration file {args.paths.calibration_json}")
        cameraParameters = read_calibration_data_domejson(args.paths.calibration_json)

    print(f"Found {len(cameraParameters)} CameraParameter entries")
    cameraParametersBase = [param for param in cameraParameters if param.name == args.paths.cameras[0]]
    cameraParametersMatch = [param for param in cameraParameters if param.name == args.paths.cameras[1]]



    # the cameras extrinsics should be transformed so that cameraBase is the origin of our coordinate system.
    # if we don't do this, the rectification is incorrect (returns large vertical feature errors)

    if len(cameraParametersBase) == 0:
        print(f"Error, could not find camera with name '{args.paths.cameras[0]}' in calibration_dome file. "
              f"Available camera names are:{[param.name for param in cameraParameters]}")
        return
    if len(cameraParametersMatch) == 0:
        print(f"Error, could not find camera with name '{args.paths.cameras[1]}' in calibration_dome file. "
              f"Available camera names are:{[param.name for param in cameraParameters]}")
        return
    cameraParametersBase = cameraParametersBase[0]  # select first camera that matches the name
    cameraParametersMatch = cameraParametersMatch[0]  # select first camera that matches the name

    np.set_printoptions(suppress=True, precision=4)
    print(f"Base {cameraParametersBase}")
    print(f"Match {cameraParametersMatch}")
    print("Extrinsic Paramaters (R-rotationmatrix and T-translationvector) are transformed so Base Camera is origin\n")

    image_path = os.path.expanduser(args.paths.image_path)
    print(f"Finding images in {image_path}")
    camera_images = find_camera_images(image_path, args.paths.cameras, args.paths.camera_pattern)
    image_names = camera_images.image_names
    camera_names = camera_images.cameras
    image_path = camera_images.image_path
    if len(image_names) == 0:
        print("Could not find any images :(")
        return

    rectification = Rectification(cameraParametersBase, cameraParametersMatch)  # Rectification class for camera pair

    createOutputFileSystem(args)  # only create output filesystem once we are sure that there are no obvious error

    print("\nApplyling rectification\n")
    verticalErrors = []
    for image_inx in range(len(image_names)):  # for each image index
        print(f"image pair {image_inx}/{len(image_names) - 1}")
        image_pair = [cv2.imread(os.path.join(image_path, camera_name, image_names[image_inx]), cv2.IMREAD_GRAYSCALE)
                      for camera_name in camera_names]

        # rectify image pairs
        # todo multical uses a pool to speed up image processing, we can probably use the same to speed this up
        rectifyBase, rectifyMatch = rectification.apply(image_pair[0], image_pair[1])
        cv2.imwrite(os.path.join(args.paths.output_path, camera_names[0], "image" + str(image_inx) + ".png"),
                    rectifyBase)
        cv2.imwrite(os.path.join(args.paths.output_path, camera_names[1], "image" + str(image_inx) + ".png"),
                    rectifyMatch)

        if args.paths.boards:  # if we are provided with a board to search in the image,
            # compute vertical error between charucocorners

            pts_base, pts_match = CharucoBoardDetection(rectifyBase, rectifyMatch, args.paths.boards)
            if (len(pts_base) == 0 or len(pts_match) == 0):
                print(
                    f"Could not find charuco board in image {image_inx} ({image_names[image_inx]}) for feature matching :(")
            else:
                verticalerror = CalculateYDifferenceForPointMatches(pts_base, pts_match, verbos=True)
                verticalErrors.append(verticalerror)
                if args.debug == 1:
                    ShowFeaturePairs(rectifyBase, rectifyMatch, pts_base, pts_match)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            # frameBase = image_pair[0]
            # frameMatch = image_pair[1]
            # pts_base_o, pts_match_o = CharucoBoardDetection(frameBase, frameMatch, args.paths.boards)
            # ShowFeaturePairs(frameBase, frameMatch, pts_base_o, pts_match_o)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        if args.debug == 2:
            def debug_rectification_images():
                frameBase = image_pair[0]
                frameMatch = image_pair[1]
                # show_image_stack("input", [frameBase, frameMatch])
                undistortBase, undistortMatch = rectification.simple_undistort(frameBase, frameMatch)
                # show_image_stack("undistorted", [undistortBase, undistortMatch])
                # show_image_stack("undistortion difference", [frameBase-undistortBase,frameMatch - undistortMatch])
                # show_image_stack("rectified", [rectifyBase, rectifyMatch])
                # show_image_stack("rectified Base", [np.abs(rectifyMatch-rectifyMatch_2)])
                # show_image_stack("rectify difference", [frameBase-rectifyBase, frameMatch - rectifyMatch])
                # show_image_stack("difference rectify and undistort", [undistortBase-rectifyBase, undistortMatch-rectifyMatch])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            debug_rectification_images()

            if False:
                leftlines, rightlines = epipolarlines(rectifyBase, rectifyMatch, pts_base, pts_match)
                # baselines_orig, matchlintes_orig = epipolarlines(frameBase, frameMatch)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print("\nRectification Finished! Output images written to " + args.paths.output_path+"\n")
    verticalErrors = np.array(verticalErrors)
    print(f"vertical errors mean: {np.mean(verticalErrors)}, std: {np.std(verticalErrors)}")
    print(f"    min: {np.min(verticalErrors)}, for image pair: {np.argmin(verticalErrors)}")
    print(f"    max: {np.max(verticalErrors)}, for image pair: {np.argmax(verticalErrors)}")


if __name__ == '__main__':
    run_with()
