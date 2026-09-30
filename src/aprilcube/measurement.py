"""Independent AprilCube target measurements for calibration and offline work.

This API accepts images and OpenCV camera parameters; it does not open a
camera. Geometry from AprilCube configs is converted from millimeters to
meters for returned correspondences and target-to-camera transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np

from . import detect


def _camera_parameters(camera_matrix, dist_coeffs):
    """Copy and validate already OpenCV-compatible numeric parameters."""
    matrix = np.asarray(camera_matrix, dtype=np.float64).copy()
    distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1).copy()
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-12)
    ):
        raise ValueError("Camera matrix must be a finite 3x3 pinhole intrinsic matrix")
    if distortion.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
        raise ValueError("Distortion coefficients must be a finite OpenCV coefficient vector")
    return matrix, distortion


@dataclass(frozen=True)
class TargetDetection:
    detected: bool
    T_camera_target: np.ndarray | None
    tag_ids: tuple[int, ...]
    image_points_px: np.ndarray
    object_points_m: np.ndarray
    reprojection_rms_px: float | None
    corner_errors_px: np.ndarray
    inlier_indices: np.ndarray
    timestamp_monotonic_s: float
    rejection_reason: str | None

    @property
    def T_camera_object(self):
        """Object is the target's own configured coordinate frame."""
        return self.T_camera_target

    def to_dict(self) -> dict:
        transform = None
        if self.T_camera_target is not None:
            transform = self.T_camera_target.tolist()

        return {
            "detected": self.detected,
            "T_camera_target": transform,
            "tag_ids": list(self.tag_ids),
            "image_points_px": self.image_points_px.tolist(),
            "object_points_m": self.object_points_m.tolist(),
            "reprojection_rms_px": self.reprojection_rms_px,
            "corner_errors_px": self.corner_errors_px.tolist(),
            "inlier_indices": self.inlier_indices.tolist(),
            "timestamp_monotonic_s": self.timestamp_monotonic_s,
            "rejection_reason": self.rejection_reason,
            "transform_convention": (
                "target -> OpenCV camera (+X right, +Y down, +Z forward); meters"
            ),
            "corner_order": "Four corners TL,TR,BR,BL per tag, in tag_ids order",
        }


class TargetDetector:
    """Measure one target independently in each image, retaining raw corners.

    Geometry is loaded from an AprilCube JSON config. Camera parameters must
    already use OpenCV's pinhole and forward-distortion convention. Public
    ``tag_corner_map`` coordinates and accepted transforms use meters;
    ``marker_normals`` contains outward normals in the target frame.

    No previous pose, smoothing, or optical flow enters the measurement.
    Failed pose checks retain decoded correspondences for calibration.
    """

    def __init__(
        self,
        config_path: Path,
        camera_matrix,
        dist_coeffs,
        *,
        min_tags=1,
        max_reprojection_rms_px=2.0,
    ):
        self.config_path = Path(config_path).resolve()
        self.config, self.faces = detect.load_cube_config(str(self.config_path))

        # AprilCube geometry is in millimeters. PnP and saved points use meters.
        corners_mm = detect.build_tag_corner_map(self.config)
        self.tag_corner_map = {}
        for tag, corners in corners_mm.items():
            self.tag_corner_map[tag] = corners * 0.001

        self.camera_matrix, self.dist_coeffs = _camera_parameters(camera_matrix, dist_coeffs)
        if not math.isfinite(max_reprojection_rms_px) or max_reprojection_rms_px <= 0:
            raise ValueError("Maximum reprojection RMS must be positive and finite")
        if not isinstance(min_tags, int) or min_tags < 1:
            raise ValueError("Minimum tag count must be a positive integer")

        self.max_reprojection_rms_px = max_reprojection_rms_px
        self.min_tags = min_tags
        # AprilCube's own marker-detection pass; the estimator's pose tracking is unused.
        self.estimator = detect.CubePoseEstimator(
            self.config, self.faces, self.camera_matrix, self.dist_coeffs, enable_filter=False,
        )

        self.marker_normals = {}
        for name, axis, sign, *_ in detect.FACE_DEFS:
            normal = np.zeros(3)
            normal[axis] = sign
            for tag in self.faces.get(name, []):
                self.marker_normals[tag] = normal
        marker_normals = self.config.marker_normals or {}
        for tag, normal in marker_normals.items():
            self.marker_normals[int(tag)] = np.asarray(normal, dtype=np.float64)

    def decode_markers(self, frame_bgr: np.ndarray) -> tuple[dict[int, np.ndarray], set[int]]:
        """Decode configured markers without estimating or accepting a pose.

        Return image corners by tag ID and the set of IDs decoded more than
        once. Corners are pixel coordinates in TL, TR, BR, BL order. Only
        configured IDs passing the existing quadrilateral-quality check are
        included. For duplicate IDs the final quad is retained for diagnostics;
        the duplicate set records that its geometry is ambiguous.
        """
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim not in (2, 3):
            raise ValueError("Detector expects a uint8 grayscale or BGR image")

        # Decode markers in the current image with AprilCube's shared detection pass.
        markers = self.estimator.detect_markers(frame_bgr)
        corners, ids = markers.corners, markers.ids

        observations = {}
        duplicate_ids = set()
        if ids is not None:
            decoded_ids = np.asarray(ids).reshape(-1)
            for quad, tag in zip(corners, decoded_ids):
                tag = int(tag)
                quad = np.asarray(quad).reshape(4, 2)
                if tag not in self.tag_corner_map:
                    continue
                if detect._quad_quality(quad) <= 0.15:
                    continue
                if tag in observations:
                    duplicate_ids.add(tag)
                observations[tag] = quad.copy()

        return observations, duplicate_ids

    def process(self, frame_bgr: np.ndarray, timestamp=None) -> TargetDetection:
        if timestamp is None:
            timestamp = time.monotonic()
        else:
            timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Frame timestamp must be finite")
        observations, duplicate_ids = self.decode_markers(frame_bgr)

        # Keep image corners and target points in the same stable tag order.
        tag_ids = tuple(sorted(observations))
        if tag_ids:
            image_corners = [observations[tag] for tag in tag_ids]
            object_corners = [self.tag_corner_map[tag] for tag in tag_ids]
            image_points = np.concatenate(image_corners).astype(np.float64)
            object_points = np.concatenate(object_corners).copy()
        else:
            image_points = np.empty((0, 2))
            object_points = np.empty((0, 3))

        transform = None
        rms = None
        errors = np.empty(0, dtype=np.float64)
        inliers = np.empty(0, dtype=np.int64)

        def result(reason=None):
            detected = reason is None
            accepted_transform = transform
            if not detected:
                accepted_transform = None

            return TargetDetection(
                detected=detected,
                T_camera_target=accepted_transform,
                tag_ids=tag_ids,
                image_points_px=image_points,
                object_points_m=object_points,
                reprojection_rms_px=rms,
                corner_errors_px=errors,
                inlier_indices=inliers,
                timestamp_monotonic_s=timestamp,
                rejection_reason=reason,
            )

        if duplicate_ids:
            return result(f"Ambiguous duplicate tag IDs: {sorted(duplicate_ids)}")
        if len(tag_ids) < self.min_tags:
            return result(f"Only {len(tag_ids)} decoded tags; need {self.min_tags}")

        # Estimate target -> camera directly from this frame, without a prior.
        try:
            success, rvec, tvec, _, selected = detect.estimate_pose(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
            )
        except cv2.error as exc:
            return result(f"PnP failed: {exc}")
        if not success or rvec is None or tvec is None:
            return result("PnP did not produce a pose")
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            return result("PnP produced a nonfinite pose")

        # Column-vector convention: point_camera = rotation @ point_target + tvec.
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = object_points @ rotation.T + tvec.reshape(1, 3)
        if np.any(camera_points[:, 2] <= 0):
            return result("Estimated target corners are behind the camera")

        for index, tag in enumerate(tag_ids):
            if tag not in self.marker_normals:
                continue
            normal_camera = rotation @ self.marker_normals[tag]
            first_corner = index * 4
            tag_corners_camera = camera_points[first_corner:first_corner + 4]
            center_camera = tag_corners_camera.mean(axis=0)
            if np.dot(normal_camera, center_camera) >= 0:
                return result(f"Back-facing solution for tag {tag}")

        # Check every decoded corner, including any excluded by RANSAC.
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        corner_offsets = projected.reshape(-1, 2) - image_points
        errors = np.linalg.norm(corner_offsets, axis=1)
        rms = float(np.sqrt(np.mean(errors ** 2)))
        if selected is not None:
            inliers = np.asarray(selected, dtype=np.int64).reshape(-1)

        if not math.isfinite(rms) or not np.all(np.isfinite(errors)):
            rms = None
            errors = np.empty(0, dtype=np.float64)
            return result("Nonfinite corner reprojection error")
        if rms > self.max_reprojection_rms_px:
            return result(
                f"Corner reprojection RMS {rms:.3f}px "
                f"exceeds {self.max_reprojection_rms_px:.3f}px"
            )

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)
        return result()

    def annotate(self, frame_bgr: np.ndarray, detection: TargetDetection) -> np.ndarray:
        image = frame_bgr.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        tag_corners = detection.image_points_px.reshape(-1, 4, 2)
        for tag, quad in zip(detection.tag_ids, tag_corners):
            points = np.rint(quad).astype(np.int32)
            cv2.polylines(image, [points], True, (255, 210, 40), 2, cv2.LINE_AA)
            cv2.putText(
                image,
                str(tag),
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 210, 40),
                1,
                cv2.LINE_AA,
            )

        if detection.detected:
            rvec, _ = cv2.Rodrigues(detection.T_camera_target[:3, :3])
            cv2.drawFrameAxes(
                image,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                detection.T_camera_target[:3, 3],
                0.04,
                2,
            )
            message = (
                f"Target: {len(detection.tag_ids)} tags | "
                f"RMS {detection.reprojection_rms_px:.2f}px"
            )
            status_color = (0, 230, 0)
        else:
            message = f"Target: {detection.rejection_reason}"
            status_color = (0, 100, 255)

        cv2.putText(
            image,
            message,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA,
        )
        return image
