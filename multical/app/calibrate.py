
from multical.io.logging import setup_logging
from .vis import visualize_ws

from structs.struct import struct, map_none, to_structs
import numpy as np

from multical.config import *
from dataclasses import dataclass

@dataclass
class Calibrate:
    """Run camera calibration"""
    paths  : PathOpts 
    camera  : CameraOpts
    runtime    : RuntimeOpts
    optimizer  : OptimizerOpts 
    vis : bool = False        # Visualize result after calibration

    def execute(self):
        calibrate(self)


def calibrate(args): 
  np.set_printoptions(precision=4, suppress=True)

  # Use image path if not explicity specified
  output_path = args.paths.image_path or args.paths.output_path 

  ws = workspace.Workspace(output_path, args.paths.name)
  setup_logging(args.runtime.log_level, [ws.log_handler], log_file=path.join(output_path, f"{args.paths.name}.txt"))

  boards = find_board_config(args.paths.image_path, board_file=args.paths.boards)
  camera_images = find_camera_images(args.paths.image_path, # camera_images = path to camera folders, cameras = list of camera ids (cam folder names), image_names= basename of matching frame names (image0.tif, image1.tif ...), file_names = absolute path of all images
    args.paths.cameras, args.paths.camera_pattern, limit=args.paths.limit_images)

  initialise_with_images(ws, boards, camera_images, args.camera, args.runtime)
  optimize(ws, args.optimizer)

  ws.export(master=args.paths.master)
  ws.dump()

  if args.vis:
    visualize_ws(ws)



if __name__ == '__main__':
  run_with(Calibrate)
