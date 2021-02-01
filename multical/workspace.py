from collections import OrderedDict
from multical.io.export import export, export_cameras
from multical.image.detect import common_image_size
import numpy as np
from structs.numpy import shape
import os
from multical.optimization.calibration import Calibration
from structs.struct import map_list, split_dict, struct, subset
from . import tables, image
from .camera import calibrate_cameras

from .io.logging import MemoryHandler, info, warning, debug
import palettable.colorbrewer.qualitative as palettes

import pickle


def make_palette(n):
  n_colors = max(n, 4)
  colors = getattr(palettes, f"Set1_{n_colors}").colors
  return np.array(colors) / 255


def num_threads():
  return len(os.sched_getaffinity(0))


class Workspace:
  def __init__(self):

    self.calibrations = OrderedDict()
    self.detections = None
    self.boards = None
    self.board_colors = None

    self.filenames = None
    self.image_path = None
    self.names = struct()

    self.image_sizes = None
    self.images = None

    self.point_table = None
    self.pose_table = None

    self.log_handler = MemoryHandler()


  def find_images(self, image_path, camera_dirs=None):
    camera_names, image_names, filenames = image.find.find_images(image_path, camera_dirs)
    info("Found camera directories {} with {} matching images".format(str(camera_names), len(image_names)))

    self.names = self.names._extend(camera = camera_names, image = image_names)
    self.filenames = filenames
    self.image_path = image_path


  def load_images(self, j=num_threads()):
    assert self.filenames is not None 

    info("Loading images..")
    self.images = image.detect.load_images(self.filenames, j=j, prefix=self.image_path)
    self.image_size = map_list(common_image_size, self.images)

  def try_load_detections(self, filename):
    try:
      with open(filename, "rb") as file:
        loaded = pickle.load(file)
        # Check that the detections match the metadata
        if (loaded.filenames == self.filenames and 
            loaded.boards == self.boards and
            loaded.image_sizes == self.image_sizes):

          info(f"Loaded detections from {filename}")
          return loaded.detected_points
        else:
          info(f"Config changed, not using loaded detections in {filename}")
    except (OSError, IOError, EOFError) as e:
      return None

  def write_detections(self, filename):
    data = struct(
      filenames = self.filenames,
      boards = self.boards,
      image_sizes = self.image_sizes,
      detected_points = self.detected_points
    )
    with open(filename, "wb") as file:
      pickle.dump(data, file)

  def detect_boards(self, boards, j=num_threads(), cache_file=None, load_cache=True):
    board_names, self.boards = split_dict(boards)
    self.names = self.names._extend(board = board_names)
    self.board_colors = make_palette(len(boards))

    self.detected_points = self.try_load_detections(cache_file) if load_cache else None
    if self.detected_points is None:
      info("Detecting boards..")
      self.detected_points = image.detect.detect_images(self.boards, self.images, j=j)   

      if cache_file is not None:
        self.write_detections(cache_file)

    self.point_table = tables.make_point_table(self.detected_points, self.boards)
    info("Detected point counts:")
    tables.table_info(self.point_table.valid, self.names)



  def calibrate_single(self, camera_model, fix_aspect=False, max_images=None):
    assert self.detected_points is not None

    info("Calibrating single cameras..")
    self.cameras, errs = calibrate_cameras(self.boards, self.detected_points, 
      self.image_size, model=camera_model, fix_aspect=fix_aspect, max_images=max_images)
    
    for name, camera, err in zip(self.names.camera, self.cameras, errs):
      info(f"Calibrated {name}, with RMS={err:.2f}")
      info(camera)
      info("---------------")


  def initialise_poses(self):
    assert self.cameras is not None
    self.pose_table = tables.make_pose_table(self.point_table, self.boards, self.cameras)
    
    info("Pose counts:")
    tables.table_info(self.pose_table.valid, self.names)

    pose_initialisation = tables.initialise_poses(self.pose_table)
    calib = Calibration(self.cameras, self.boards, self.point_table, pose_initialisation)
    calib = calib.reject_outliers_quantile(0.75, 2)
    calib.report(f"Initialisation")

    self.calibrations['initialisation'] = calib
    return calib


  def calibrate(self, name, intrinsics=False, board=False, rolling=False, **opt_args):
    calib = self.latest_calibration.enable(intrinsics=intrinsics, board=board, rolling=rolling)
        
    calib = calib.adjust_outliers(**opt_args)
    self.calibrations[name] = calib
    return calib

  @property
  def sizes(self):
    return self.names._map(len)

  @property
  def latest_calibration(self):
    return list(self.calibrations.values())[-1]

  @property
  def log_entries(self):
    return self.log_handler.records

  def has_calibrations(self):
    return len(self.calibrations) > 0

  def get_calibrations(self):
    return self.calibrations

  def get_camera_sets(self):
      if self.has_calibrations():
        return {k:calib.cameras for k, calib in self.calibrations.items()}

      if self.cameras is not None:
        return dict(initialisation = self.cameras)

  def export(self, filename):
    info(f"Exporting calibration to {filename}")
    export(filename, self.latest_calibration, self.names)

  def dump(self, filename):
    info(f"Dumping state and history to {filename}")
    with open(filename, "wb") as file:
      pickle.dump(self, file)

  def __getstate__(self):
    d = subset(self.__dict__, ['calibrations', 'detections', 'boards', 
      'board_colors', 'filenames', 'image_path', 'names', 'image_sizes',
      'point_table', 'pose_table', 'log_handler'
    ])
    return d

  def __setstate__(self, d):
    for k, v in d:
      self.__dict__[k] = d  

    self.images = None