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
    cameraParameters = read_calibration_data(os.path.join(args.paths.image_path, args.paths.calibration_json))
    print(f"Found {len(cameraParameters)} CameraParameter entries")

    image_path = os.path.expanduser(args.paths.image_path)
    print(f"Finding images in {image_path}")
    camera_images = find_camera_images(image_path, args.paths.cameras, args.paths.camera_pattern)
    image_names = camera_images.image_names
    camera_names = camera_images.cameras
    image_path = camera_images.image_path
    if len(image_names) == 0:
        print("Could not find any images :(")
        return

    rectification = Rectification(cameraParameters[1], cameraParameters[2]) # Rectification class for IR camera pair

    for image_inx in range(1):

        image_list = [cv2.imread(os.path.join(image_path, camera_name, image_names[image_inx]), cv2.IMREAD_GRAYSCALE) for camera_name in camera_names]

        show_image_stack("input", image_list)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # rectify IR pairs
        rectifyBase, rectifyMatch = rectification.apply(image_list[1], image_list[2])
        cv2.imwrite(os.path.join("./", args.paths.output_path, camera_name[1],"image"+image_inx+".png"), rectifyBase)
        cv2.imwrite(os.path.join("./", args.paths.output_path, camera_name[2],"image"+image_inx+".png"), rectifyMatch)

        frameBase = image_list[0]
        frameMatch = image_list[1]
        def debug_rectification_images():
            show_image_stack("input", [frameBase, frameMatch])
            undistortBase, undistortMatch = rectification.simple_undistort(frameBase, frameMatch)
            #show_image_stack("undistorted", [undistortBase, undistortMatch])
            #show_image_stack("undisortion difference", [frameBase-undistortBase,frameMatch - undistortMatch])
            show_image_stack("rectified", [rectifyBase, rectifyMatch])
            #show_image_stack("rectified Base", [np.abs(rectifyMatch-rectifyMatch_2)])
            #show_image_stack("rectify difference", [frameBase-rectifyBase, frameMatch - rectifyMatch])
            #show_image_stack("difference rectify and undistort", [undistortBase-rectifyBase, undistortMatch-rectifyMatch])
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
