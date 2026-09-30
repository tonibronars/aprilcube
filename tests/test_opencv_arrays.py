"""The normal tracker accepts OpenCV 4 and OpenCV 5 marker-detection outputs.

OpenCV 5 returns marker IDs as a flat array and rejected candidates as a single
ndarray, instead of the nested IDs and lists of OpenCV 4.
"""

import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from aprilcube import detect
from aprilcube.generate import CubeConfig, FACE_DEFS


K = np.array([[1200., 0, 800], [0, 1200., 500], [0, 0, 1]])


@pytest.fixture
def cube(tmp_path):
    """A small generated cuboid config, independent of saved models."""
    config = CubeConfig(
        grid_x=1, grid_y=3, grid_z=3,
        dict_id=cv2.aruco.DICT_4X4_1000, dict_name="4x4_1000",
        tag_ids=list(range(30)), tag_size_mm=24,
    )
    config.compute()
    faces = {}
    cursor = 0
    for face in FACE_DEFS:
        rows, columns, *_ = config.face_layout(face)
        faces[face[0]] = config.tag_ids[cursor:cursor + rows * columns]
        cursor += rows * columns
    payload = {
        "dict": config.dict_name, "grid": "1x3x3", "tag_ids": config.tag_ids,
        "tag_size_mm": config.tag_size_mm, "cell_size_mm": config.cell_size,
        "margin_cells": config.margin_cells, "border_cells": config.border_cells,
        "marker_pixels": config.marker_pixels, "box_dims": config.box_dims,
        "faces": faces,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return detect.load_cube_config(str(path))


@pytest.mark.parametrize("rejected", [None, (), np.empty((0, 4, 2)), np.ones((2, 4, 2))])
def test_tracker_accepts_opencv_rejected_arrays(cube, rejected):
    config, faces = cube
    tracker = detect.CubePoseEstimator(config, faces, K, np.zeros(5))
    tracker.detector = Mock()
    tracker.detector.detectMarkers.return_value = ((), None, rejected)
    result = tracker.process_frame(np.zeros((1000, 1600), dtype=np.uint8), timestamp=1)
    assert not result["success"]
    assert result["detections"] == []


def test_tracker_accepts_fallback_rejected_array(cube):
    config, faces = cube
    tracker = detect.CubePoseEstimator(config, faces, K, np.zeros(5))
    tracker.prev_rvec = np.zeros((3, 1))
    tracker.prev_tvec = np.array([[0.], [0.], [600.]])
    tracker.detector = Mock()
    tracker.detector.detectMarkers.return_value = ((), None, np.ones((1, 4, 2)))
    tracker.fallback_detector = Mock()
    tracker.fallback_detector.detectMarkers.return_value = ((), None, np.ones((2, 4, 2)))
    tracker._recover_rejected = Mock(return_value=[])
    tracker._track_corners_optflow = Mock(return_value=None)
    result = tracker.process_frame(np.zeros((1000, 1600), dtype=np.uint8), timestamp=1)
    assert not result["success"]
    assert len(tracker._recover_rejected.call_args.args[2]) == 3
    tracker.fallback_detector.detectMarkers.assert_called_once()


@pytest.mark.parametrize("ids", [np.array([[0], [1]]), np.array([0, 1])])
def test_tracker_reads_nested_and_flat_marker_ids(cube, ids):
    config, faces = cube
    tracker = detect.CubePoseEstimator(config, faces, K, np.zeros(5))
    quad = np.float32([[[100, 100], [160, 100], [160, 160], [100, 160]]])
    tracker.detector = Mock()
    tracker.detector.detectMarkers.return_value = ((quad, quad + 300), ids, ())
    result = tracker.process_frame(np.zeros((1000, 1600), dtype=np.uint8), timestamp=1)
    assert sorted(tag for tag, _ in result["detections"]) == [0, 1]
