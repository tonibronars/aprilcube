"""Regressions for cold-start PnP on several tags in one planar surface."""

import cv2
import numpy as np
import pytest

from aprilcube.detect import estimate_pose


@pytest.fixture
def planar_observation():
    # Four separate tags, all on a surface 10 mm above the object origin.
    centers = [(-24, -24), (24, -24), (-24, 24), (24, 24)]
    corners = [(12, -12), (-12, -12), (-12, 12), (12, 12)]
    object_points = np.array([
        [x + dx, y + dy, 10]
        for x, y in centers for dx, dy in corners
    ], dtype=np.float64)
    camera_matrix = np.array([[2168., 0, 1898], [0, 2169, 1064], [0, 0, 1]])
    distortion = np.array([-0.18, 0.045, 0.001, -0.001, 0.005])
    tilt = cv2.Rodrigues(np.array([0.25, -0.35, 0.1]))[0]
    rotation = tilt @ np.diag([1., -1., -1.])
    rvec = cv2.Rodrigues(rotation)[0]
    tvec = np.array([[80.], [-35.], [650.]])
    image_points = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion,
    )[0].reshape(-1, 2)
    image_points += np.random.default_rng(8).normal(0, 0.08, image_points.shape)
    return object_points, image_points, camera_matrix, distortion, rotation, tvec


def test_planar_multi_tag_pose_survives_ransac_failure(planar_observation, monkeypatch):
    object_points, image_points, camera_matrix, distortion, rotation, translation = planar_observation
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda **kwargs: (False, None, None, None))

    success, rvec, tvec, error, inliers = estimate_pose(
        object_points, image_points, camera_matrix, distortion,
    )

    assert success
    assert error < 0.2
    assert len(inliers) == len(object_points)
    actual_rotation = cv2.Rodrigues(rvec)[0]
    np.testing.assert_allclose(tvec, translation, atol=0.5)
    np.testing.assert_allclose(actual_rotation, rotation, atol=0.003)
    assert np.linalg.det(actual_rotation) == pytest.approx(1)
    assert actual_rotation[2, 2] < 0  # The printed surface faces the camera.
    assert np.all((object_points @ actual_rotation.T + tvec.ravel())[:, 2] > 0)


def test_nonplanar_ransac_failure_keeps_existing_behavior(planar_observation, monkeypatch):
    object_points, _, camera_matrix, distortion, rotation, translation = planar_observation
    object_points = object_points.copy()
    object_points[4:8, 2] += 25
    image_points = cv2.projectPoints(
        object_points, cv2.Rodrigues(rotation)[0], translation, camera_matrix, distortion,
    )[0].reshape(-1, 2)
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda **kwargs: (False, None, None, None))

    def unexpected_direct_solve(**kwargs):
        pytest.fail("Planar fallback must not run on nonplanar correspondences")

    monkeypatch.setattr(cv2, "solvePnP", unexpected_direct_solve)
    success, rvec, tvec, error, inliers = estimate_pose(
        object_points, image_points, camera_matrix, distortion,
    )

    assert not success
    assert rvec is None and tvec is None and inliers is None
    assert np.isinf(error)


def test_corrupted_planar_tag_cannot_report_a_low_reprojection_error(planar_observation, monkeypatch):
    object_points, image_points, camera_matrix, distortion, _, _ = planar_observation
    image_points = image_points.copy()
    image_points[4:8] += [100, -90]
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda **kwargs: (False, None, None, None))

    success, rvec, tvec, error, _ = estimate_pose(
        object_points, image_points, camera_matrix, distortion,
    )

    assert not success or error > 3.0
    if success:
        projected = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion,
        )[0].reshape(-1, 2)
        measured_error = np.linalg.norm(projected - image_points, axis=1).mean()
        assert error == pytest.approx(measured_error)


def test_planar_direct_fit_beats_a_successful_bad_ransac_candidate(planar_observation, monkeypatch):
    object_points, image_points, camera_matrix, distortion, rotation, translation = planar_observation
    wrong_translation = translation + np.array([[60.], [30.], [100.]])
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda **kwargs: (
        True, cv2.Rodrigues(rotation)[0], wrong_translation.copy(),
        np.arange(4).reshape(-1, 1),
    ))

    success, _, tvec, error, _ = estimate_pose(
        object_points, image_points, camera_matrix, distortion,
    )

    assert success
    assert error < 0.2
    np.testing.assert_allclose(tvec, translation, atol=0.5)


def test_optional_planar_solver_error_preserves_ransac_pose(planar_observation, monkeypatch):
    object_points, image_points, camera_matrix, distortion, rotation, translation = planar_observation
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda **kwargs: (
        True, cv2.Rodrigues(rotation)[0], translation.copy(),
        np.arange(len(object_points)).reshape(-1, 1),
    ))

    def failed_direct_solve(*args, **kwargs):
        raise cv2.error("SQPNP could not solve the optional candidate")

    monkeypatch.setattr(cv2, "solvePnP", failed_direct_solve)
    success, _, tvec, error, _ = estimate_pose(
        object_points, image_points, camera_matrix, distortion,
    )

    assert success
    assert error < 0.2
    np.testing.assert_allclose(tvec, translation, atol=0.5)
