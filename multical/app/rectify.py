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
    print(f"Cameras: {args.paths.cameras}")  # --cameras cam0 cam1 cam3
    print(f"Reading calibration file {args.paths.calibration_json}")
    cameraParameters = read_calibration_data(args.paths.calibration_json)
    print(f"Found {len(cameraParameters)} CameraParameter entries")

    image_path = os.path.expanduser(args.paths.image_path)
    print(f"Finding images in {image_path}")
    camera_images = find_camera_images(image_path, args.paths.cameras, args.paths.camera_pattern, matching=False)

    camBase: CameraParameters = cameraParameters[0]
    camMatch = cameraParameters[1]

    #for camMatch in cameraParameters:
    #    if camMatch.name in args.paths.cameras:
    #        # perform image rectification for cameras
    #        rectification = Rectification(camBase, camMatch)

    print(f"output path: {args.paths.output_path}")

    rectification = Rectification(camBase, camMatch)
    frameBase = cv2.imread("./cam0/image83.png")
    frameMatch = cv2.imread("./cam1/image83.png")
    rectifyBase, rectifyMatch = rectification.apply(frameBase, frameMatch)
    cv2.imwrite("./cam0_rectified.png", rectifyBase)
    cv2.imwrite("./cam1_rectified.png", rectifyMatch)

    def debug_rectification_images():
        show_image_stack("input", [frameBase, frameMatch])
        undistortBase, undistortMatch = rectification.simple_undistort(frameBase, frameMatch)
        show_image_stack("undistorted", [undistortBase, undistortMatch])
        show_image_stack("undisortion difference", [frameBase-undistortBase,frameMatch - undistortMatch])
        show_image_stack("rectified", [rectifyBase, rectifyMatch])
        show_image_stack("rectify difference", [frameBase-rectifyBase, frameMatch - rectifyMatch])
        show_image_stack("difference rectify and undistort", [undistortBase-rectifyBase, undistortMatch-rectifyMatch])
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

if __name__ == '__main__':
    run_with()
