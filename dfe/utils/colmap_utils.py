"""Utils for colmap data processing.
"""

import math
import numpy as np


def pair_id_to_image_ids(pair_id):
    """Get image ids from pair id. (Taken from colmap)

    Args:
        pair_id (int): image pair id

    Returns:
        tuple: id of first image, id of second image
    """
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647

    return int(image_id1), int(image_id2)


def get_cam(img, cams):
    """Get camera data.

    Args:
        img: image data
        cams: camera data

    Returns:
        tuple: intrinsic, extrinsic, image size
    """
    R = quaterion_to_rotation_matrix(np.asarray(img.qvec))
    t = img.tvec
    camera = cams[int(img.camera_id)]

    params = camera.params

    sz = np.array([camera.width, camera.height])

    T = np.eye(4)
    T[:3, :3] = R[:3, :3]
    T[:3, 3] = t

    K = np.eye(3)
    K[0, 0] = params[0]
    K[1, 1] = params[0]
    K[0, 2] = params[1]
    K[1, 2] = params[2]

    return K, T, sz


def compose_fundamental_matrix(K1, T1, K2, T2):
    """Compose fundamental matrix.

    Args:
        K1 (array): intrinsic of 1st camera
        T1 (array): extrinsic of 1st camera
        K2 (array): intrinsic of 2nd camera
        T2 (array): extrinsic of 2nd camera

    Returns:
        array: fundamental matrix
    """
    T12 = T2 @ np.linalg.inv(T1)

    R = T12[:3, :3]
    t = T12[:3, 3]

    A = np.dot(np.dot(K1, R.T), t)
    C = vector_to_cross(A)

    F = np.linalg.inv(K2).T @ R @ K1.T @ C

    return F


def quaterion_to_rotation_matrix(quaternion):
    """Get rotation matrix from quaternion.

    Args:
        quaternion (array): quaternion

    Returns:
        array: rotation matrix
    """

    #    Return homogeneous rotation matrix from quaternion.

    # >> > M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    # >> > numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    # True
    # >> > M = quaternion_matrix([1, 0, 0, 0])
    # >> > numpy.allclose(M, numpy.identity(4))
    # True
    # >> > M = quaternion_matrix([0, 1, 0, 0])
    # >> > numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    # True

    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)

    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def compute_residual(pts, F):
    """Compute epipolar residual.

    Args:
        pts (array): points
        F (array): fundamental matrix

    Returns:
        array: residuals
    """
    pts1 = points_to_homogeneous(pts[:, :2])
    pts2 = points_to_homogeneous(pts[:, 2:])

    line_2 = np.dot(F.T, pts1.T)
    line_1 = np.dot(F, pts2.T)

    dd = np.sum(line_2.T * pts2, 1)

    d = np.abs(dd) * (
        1.0 / np.sqrt(line_1[0, :] ** 2 + line_1[1, :] ** 2)
        + 1.0 / np.sqrt(line_2[0, :] ** 2 + line_2[1, :] ** 2)
    )

    return d


def vector_to_cross(vec):
    """Compute cross product matrix.

    Args:
        vec (array): vector

    Returns:
        array: cros product matrix
    """
    T = np.zeros((3, 3))

    T[0, 1] = -vec[2]
    T[0, 2] = vec[1]
    T[1, 0] = vec[2]
    T[1, 2] = -vec[0]
    T[2, 0] = -vec[1]
    T[2, 1] = vec[0]

    return T


def points_to_homogeneous(pts):
    """Transform points to homogeneous points.

    Args:
        pts (array): points

    Returns:
        array: homogeneous points
    """
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), 1)
