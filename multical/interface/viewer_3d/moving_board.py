
import numpy as np
from structs.struct import struct
from .marker import AxisSet, SceneMeshes, CameraSet, BoardSet
from structs.numpy import shape
from multical import tables

def exportcameraandboards(camnames, camera_poses, boardnames, board_poses, filename=""):
  import json
  if camnames is None:
    camnames = [str(indx) for indx in range(len(camera_poses.poses))]
  if boardnames is None:
    boardnames = [str(indx) for indx in range(len(board_poses.poses))]
  valid_camera_poses = {name: extrinsic for name, extrinsic in
                        zip(camnames, camera_poses.poses)}

  camera_view_matrix_json = [{"camera_id": name,
                              "extrinsics": {"view_matrix": valid_camera_poses[name].flatten().tolist()},
                              "intrinsics": {"camera_matrix": 0,
                                             "resolution": 0,
                                             "distortion_coefficients": 0}}
                             for name, camera
                             in zip(camnames, valid_camera_poses)]

  board_poses_json = [{"board_name": name, "extrinsics": {"view_matrix": extrinsicMatrix.flatten().tolist()}}
                      for name, extrinsicMatrix, valid
                      in zip(boardnames, board_poses.poses, board_poses.valid)
                      if valid]

  with open("./captureSimulation2/calibration_" + filename + ".json", "w") as f:
    json.dump({"meta": "camera and board poses relative to camera 0",
               "cameras": camera_view_matrix_json,
               "boards": board_poses_json}, f, indent=2)
  
class MovingBoard(object):
  def __init__(self, viewer, calib, board_colors):
    self.viewer = viewer
    self.board_colors = board_colors
    self.meshes = SceneMeshes(calib)

    camera_poses = tables.inverse(calib.pose_estimates.camera)
    self.camera_set = CameraSet(self.viewer, camera_poses, self.meshes.camera)
    board_poses = tables.expand_boards(calib.pose_estimates)
    self.board_sets = [
      BoardSet(self.viewer, poses, self.meshes.board, board_colors)
        for poses in board_poses._sequence()]
    exportcameraandboards(None, camera_poses, None, struct(poses=board_poses.poses[0], valid=board_poses.valid[0]), "meshes")

    self.axis_set = AxisSet(self.viewer, self.meshes.axis, camera_poses)
    self.show(False)


  def update_calibration(self, calib):
    self.meshes.update(calib)
    
    board_poses = tables.expand_boards(calib.pose_estimates)
    camera_poses = tables.inverse(calib.pose_estimates.camera)
    self.camera_set.update_poses(camera_poses)
    self.axis_set.update_poses(camera_poses)

    board_poses = tables.expand_boards(calib.pose_estimates)
    for board_set, poses in zip(self.board_sets, board_poses._sequence()):
      board_set.update_poses(poses)

  def show(self, shown):
    for marker in self.board_sets + [self.camera_set, self.axis_set]:
      marker.show(shown)

  def update(self, state):
    self.meshes.set_camera_scale(state.scale)
    self.camera_set.update(highlight=state.camera)
    for i, board_set in enumerate(self.board_sets):
      board_set.update(active = i == state.frame)

    self.viewer.update()

    
  def enable(self, state):
    self.show(True)
    self.update(state)

  def disable(self):
    self.show(False)