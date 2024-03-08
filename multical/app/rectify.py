import os.path

from multical.rectification.rectification import *
from multical.rectification.verify_rectification import *
from multical.config import *
from multical.io.logging import setup_logging
import cv2


@dataclass
class Rectify:
    paths: PathOpts

    def execute(self):
        rectify(self)


def show_image_stack(windowname, imagelist):
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowname, 2000, 1000)
    cv2.imshow(windowname, cv2.hconcat(imagelist))


def rectify(args):
    """
    rectify images of two cameras. Adjusts the images so their epipolarlines align and are horizontal
    """

    debug = False
    print(f"Cameras: {args.paths.cameras}")  # --cameras cam0 cam1

    if len(args.paths.cameras) != 2:
        print(f"Please specify two cameras that are to be rectified with --cameras <cam1> <cam2>")
        return

    if args.paths.output_path == args.paths.image_path or args.paths.output_path is None:
        args.paths.output_path = args.paths.image_path + "_rectified"
    if not os.path.exists(args.paths.output_path):
        os.makedirs(args.paths.output_path)
    for c in args.paths.cameras:
        cam_path = os.path.join(args.paths.output_path, c)
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)

    print(f"Reading calibration file {args.paths.calibration_json}")
    cameraParameters = read_calibration_data_domejson(args.paths.calibration_json)
    print(f"Found {len(cameraParameters)} CameraParameter entries")
    cameraParametersBase = [param for param in cameraParameters if param.name == args.paths.cameras[0]][0]
    cameraParametersMatch = [param for param in cameraParameters if param.name == args.paths.cameras[1]][0]

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

    for image_inx in range(len(image_names)):  # for each image index
        print("image pair", image_inx)
        image_pair = [cv2.imread(os.path.join(image_path, camera_name, image_names[image_inx]), cv2.IMREAD_GRAYSCALE)
                      for camera_name in camera_names]

        # rectify IR pairs
        rectifyBase, rectifyMatch = rectification.apply(image_pair[0], image_pair[1])
        cv2.imwrite(os.path.join(args.paths.output_path, camera_names[0], "image" + str(image_inx) + ".png"),
                    rectifyBase)
        cv2.imwrite(os.path.join(args.paths.output_path, camera_names[1], "image" + str(image_inx) + ".png"),
                    rectifyMatch)

        if debug:
            def debug_rectification_images():
                frameBase = image_pair[0]
                frameMatch = image_pair[1]
                show_image_stack("input", [frameBase, frameMatch])
                undistortBase, undistortMatch = rectification.simple_undistort(frameBase, frameMatch)
                # show_image_stack("undistorted", [undistortBase, undistortMatch])
                # show_image_stack("undisortion difference", [frameBase-undistortBase,frameMatch - undistortMatch])
                show_image_stack("rectified", [rectifyBase, rectifyMatch])
                # show_image_stack("rectified Base", [np.abs(rectifyMatch-rectifyMatch_2)])
                # show_image_stack("rectify difference", [frameBase-rectifyBase, frameMatch - rectifyMatch])
                # show_image_stack("difference rectify and undistort", [undistortBase-rectifyBase, undistortMatch-rectifyMatch])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            debug_rectification_images()

            if args.paths.boards:
                pts_base, pts_match = CharucoBoardDetection(rectifyBase, rectifyMatch, args.paths.boards)
                ShowFeaturePairs(rectifyBase, rectifyMatch, pts_base, pts_match)

                pts_base_o, pts_match_o = CharucoBoardDetection(frameBase, frameMatch, args.paths.boards)
                ShowFeaturePairs(frameBase, frameMatch, pts_base_o, pts_match_o)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if False:
                leftlines, rightlines = epipolarlines(rectifyBase, rectifyMatch, pts_base, pts_match)
                # baselines_orig, matchlintes_orig = epipolarlines(frameBase, frameMatch)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    print("Rectification Finished! Output files written to " + args.paths.output_path)


if __name__ == '__main__':
    run_with()
