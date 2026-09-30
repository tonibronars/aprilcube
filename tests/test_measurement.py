"""Camera-free regression tests for independent target measurements."""

import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import aprilcube
from aprilcube import detect
from aprilcube.generate import CubeConfig, FACE_DEFS
from aprilcube.measurement import TargetDetection, TargetDetector


K = np.array([[1200., 0, 800], [0, 1200., 500], [0, 0, 1]])


@pytest.fixture
def config_path(tmp_path):
    """Generate ordinary cuboid metadata locally, independent of saved models."""
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
        count = rows * columns
        faces[face[0]] = config.tag_ids[cursor:cursor + count]
        cursor += count
    payload = {
        "dict": config.dict_name, "grid": "1x3x3", "tag_ids": config.tag_ids,
        "tag_size_mm": config.tag_size_mm, "cell_size_mm": config.cell_size,
        "margin_cells": config.margin_cells, "border_cells": config.border_cells,
        "marker_pixels": config.marker_pixels, "box_dims": config.box_dims,
        "faces": faces,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def target(config_path):
    return TargetDetector(config_path, K, np.zeros(5))


def pose(translation=(-.04, .01, .60), tilt=(.12, -.20, .06)):
    tilt_rotation, _ = cv2.Rodrigues(np.asarray(tilt, dtype=np.float64))
    face_rotation, _ = cv2.Rodrigues(np.array([0., np.pi / 2, 0.]))
    transform = np.eye(4)
    transform[:3, :3] = tilt_rotation @ face_rotation
    transform[:3, 3] = translation
    return transform


def projected_quads(target, transform, tags=range(9)):
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return [
        cv2.projectPoints(target.tag_corner_map[tag], rvec, transform[:3, 3],
                          target.camera_matrix, target.dist_coeffs)[0]
        .reshape(1, 4, 2).astype(np.float32)
        for tag in tags
    ]


def render_face(target, transform, visible=range(9)):
    """Rasterize ArUco markers at the generated model's perspective corners."""
    scale = 2
    camera = target.camera_matrix.copy()
    camera[:2] *= scale
    frame = np.full((1000 * scale, 1600 * scale), 245, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(target.config.dict_id)
    marker_size = 192
    source = np.float32([[0, 0], [marker_size - 1, 0],
                         [marker_size - 1, marker_size - 1], [0, marker_size - 1]])
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    for tag in visible:
        marker = cv2.aruco.generateImageMarker(dictionary, tag, marker_size)
        projected, _ = cv2.projectPoints(target.tag_corner_map[tag], rvec,
                                         transform[:3, 3], camera, np.zeros(5))
        homography = cv2.getPerspectiveTransform(source, projected.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(marker, homography, (frame.shape[1], frame.shape[0]),
                                    flags=cv2.INTER_NEAREST, borderValue=255)
        frame = np.minimum(frame, warped)
    return cv2.cvtColor(cv2.resize(frame, (1600, 1000), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)


def set_decoded(target, quads, ids):
    target.estimator.detector = Mock()
    target.estimator.detector.detectMarkers.return_value = (quads, np.asarray(ids), ())


def test_public_exports():
    assert aprilcube.TargetDetector is TargetDetector
    assert aprilcube.TargetDetection is TargetDetection


def test_real_detection_metric_pose_corner_order_and_rms(target):
    truth = pose()
    detection = target.process(render_face(target, truth), timestamp=123.5)
    assert detection.detected, detection.rejection_reason
    assert detection.tag_ids == tuple(range(9))
    assert detection.timestamp_monotonic_s == 123.5
    assert detection.image_points_px.shape == (36, 2)
    assert detection.object_points_m.shape == (36, 3)
    np.testing.assert_allclose(detection.T_camera_target[:3, 3], truth[:3, 3], atol=.002)
    np.testing.assert_allclose(detection.T_camera_target[:3, :3], truth[:3, :3], atol=.035)
    np.testing.assert_allclose(detection.object_points_m,
                               np.concatenate([target.tag_corner_map[tag] for tag in detection.tag_ids]))
    # Generated +X-face geometry is 16 mm from the object center.
    np.testing.assert_allclose(detection.object_points_m[:, 0], .016)
    np.testing.assert_array_equal(target.marker_normals[0], [1, 0, 0])
    rvec, _ = cv2.Rodrigues(detection.T_camera_target[:3, :3])
    projected, _ = cv2.projectPoints(detection.object_points_m, rvec,
                                     detection.T_camera_target[:3, 3], K, np.zeros(5))
    errors = np.linalg.norm(projected.reshape(-1, 2) - detection.image_points_px, axis=1)
    np.testing.assert_allclose(detection.corner_errors_px, errors)
    assert detection.reprojection_rms_px == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert detection.reprojection_rms_px < 1
    assert detection.T_camera_object is detection.T_camera_target
    json.dumps(detection.to_dict(), allow_nan=False)


def test_stateless_measurement_and_annotation(target):
    first = render_face(target, pose())
    assert target.process(first, 1).detected
    missing = target.process(np.full_like(first, 245), 1.1)
    assert not missing.detected and missing.T_camera_target is None
    assert missing.tag_ids == () and missing.object_points_m.shape == (0, 3)
    json.dumps(missing.to_dict(), allow_nan=False)
    # Farther than the live tracker's temporal translation and rotation gates.
    truth = pose((.13, -.08, .65), (.12, -.08, 1.0))
    image = render_face(target, truth)
    current = target.process(image, 1.2)
    assert current.detected, current.rejection_reason
    np.testing.assert_allclose(current.T_camera_target[:3, 3], truth[:3, 3], atol=.003)
    original = image.copy()
    assert target.annotate(image, current).shape == image.shape
    np.testing.assert_array_equal(original, image)


def test_partial_visibility_and_minimum_tags(config_path):
    target = TargetDetector(config_path, K, np.zeros(5), min_tags=3)
    set_decoded(target, projected_quads(target, pose(), [0, 4, 8]), [0, 4, 8])
    detection = target.process(np.zeros((1000, 1600), dtype=np.uint8), 1)
    assert detection.detected, detection.rejection_reason
    assert detection.tag_ids == (0, 4, 8)
    set_decoded(target, projected_quads(target, pose(), [4]), [4])
    insufficient = target.process(np.zeros((1000, 1600), dtype=np.uint8), 2)
    assert not insufficient.detected and insufficient.T_camera_target is None
    assert insufficient.image_points_px.shape == (4, 2)


def test_forward_distortion_matches_projection(config_path):
    distortion = np.array([.14, -.06, .001, -.002, .01])
    target = TargetDetector(config_path, K, distortion)
    transform = pose((.12, -.08, .58))
    set_decoded(target, projected_quads(target, transform), np.arange(9))
    detection = target.process(np.full((1000, 1600), 245, dtype=np.uint8), 1)
    assert detection.detected, detection.rejection_reason
    np.testing.assert_allclose(detection.T_camera_target, transform, atol=1e-5)
    assert detection.reprojection_rms_px < .001


def test_rms_rejection_retains_all_raw_corners(config_path):
    target = TargetDetector(config_path, K, np.zeros(5), max_reprojection_rms_px=1.)
    quads = projected_quads(target, pose())
    quads[0][0, 0, 0] += 18
    set_decoded(target, quads, np.arange(9))
    detection = target.process(np.full((1000, 1600), 245, dtype=np.uint8), 1)
    assert not detection.detected and detection.T_camera_target is None
    assert "reprojection RMS" in detection.rejection_reason
    assert detection.reprojection_rms_px > 1
    assert detection.image_points_px.shape == (36, 2)
    assert detection.corner_errors_px.shape == (36,)
    np.testing.assert_array_equal(detection.image_points_px, np.asarray(quads).reshape(-1, 2))
    json.dumps(detection.to_dict(), allow_nan=False)


@pytest.mark.parametrize("pnp_failure", ["no_pose", "exception", "nonfinite"])
def test_pnp_failure_retains_raw_corners(target, monkeypatch, pnp_failure):
    quads = projected_quads(target, pose())
    set_decoded(target, quads, np.arange(9))
    if pnp_failure == "exception":
        solve = Mock(side_effect=cv2.error("synthetic failure"))
    elif pnp_failure == "nonfinite":
        solve = Mock(return_value=(True, np.full(3, np.nan), np.ones(3), 0., None))
    else:
        solve = Mock(return_value=(False, None, None, float("inf"), None))
    monkeypatch.setattr(detect, "estimate_pose", solve)
    result = target.process(np.zeros((1000, 1600), dtype=np.uint8), 1)
    assert not result.detected and result.T_camera_target is None
    assert result.tag_ids == tuple(range(9))
    np.testing.assert_array_equal(result.image_points_px, np.asarray(quads).reshape(-1, 2))
    assert result.object_points_m.shape == (36, 3)
    assert "PnP" in result.rejection_reason
    json.dumps(result.to_dict(), allow_nan=False)


def test_raw_decode_preserves_duplicate_evidence_and_skips_unknown_ids(target, monkeypatch):
    quad = np.float32([[[100, 100], [150, 100], [150, 150], [100, 150]]])
    set_decoded(target, [quad, quad + 200, quad + 300], [[0], [0], [900]])
    solve = Mock(side_effect=AssertionError("Raw decode must not invoke PnP"))
    monkeypatch.setattr(detect, "estimate_pose", solve)
    observations, duplicates = target.decode_markers(np.zeros((600, 600), dtype=np.uint8))
    assert set(observations) == {0} and duplicates == {0}
    np.testing.assert_array_equal(observations[0], (quad + 200).reshape(4, 2))
    result = target.process(np.zeros((600, 600), dtype=np.uint8), 1)
    assert not result.detected and result.T_camera_target is None
    assert "duplicate" in result.rejection_reason
    assert result.tag_ids == (0,) and result.image_points_px.shape == (4, 2)
    solve.assert_not_called()


def test_explicit_marker_geometry_and_normals(config_path):
    payload = json.loads(config_path.read_text())
    payload.pop("faces")
    payload["tag_ids"] = [42]
    payload["markers"] = [{
        "id": 42, "corners_mm": [[0, 0, 8], [24, 0, 8], [24, 24, 8], [0, 24, 8]],
        "normal": [0, 0, -1], "face": "front",
    }]
    config_path.write_text(json.dumps(payload))
    target = TargetDetector(config_path, K, np.zeros(5))
    np.testing.assert_allclose(target.tag_corner_map[42], np.array(payload["markers"][0]["corners_mm"]) / 1000)
    np.testing.assert_array_equal(target.marker_normals[42], [0, 0, -1])
    transform = np.eye(4)
    transform[:3, 3] = [-.05, -.04, .5]
    set_decoded(target, projected_quads(target, transform, [42]), [42])
    result = target.process(np.zeros((1000, 1600), dtype=np.uint8), 1)
    assert result.detected, result.rejection_reason
    target.marker_normals[42] = np.array([0, 0, 1])
    rejected = target.process(np.zeros((1000, 1600), dtype=np.uint8), 2)
    assert not rejected.detected and "Back-facing" in rejected.rejection_reason
    assert rejected.image_points_px.shape == (4, 2)


@pytest.mark.parametrize("matrix,distortion", [
    (np.eye(2), np.zeros(5)), (np.zeros((3, 3)), np.zeros(5)),
    (np.full((3, 3), np.nan), np.zeros(5)),
    (K, [0, 0, 0]), (K, [0, 0, 0, float("inf")]),
])
def test_invalid_opencv_parameters(config_path, matrix, distortion):
    with pytest.raises(ValueError):
        TargetDetector(config_path, matrix, distortion)


@pytest.mark.parametrize("count", [4, 5, 8, 12, 14])
def test_opencv_distortion_lengths_and_parameter_copy(config_path, count):
    matrix = K.copy()
    distortion = np.zeros(count)
    target = TargetDetector(config_path, matrix, distortion)
    matrix[0, 0] = 1
    distortion[0] = 1
    np.testing.assert_array_equal(target.camera_matrix, K)
    np.testing.assert_array_equal(target.dist_coeffs, np.zeros(count))
