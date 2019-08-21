"""Loss functions.
"""
import torch


def symmetric_epipolar_distance(pts1, pts2, fundamental_mat):
    """Symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix

    Returns:
        tensor: symmetric epipolar distance
    """

    line_1 = torch.bmm(pts1, fundamental_mat)
    line_2 = torch.bmm(pts2, fundamental_mat.permute(0, 2, 1))

    scalar_product = (pts2 * line_1).sum(2)

    ret = scalar_product.abs() * (
        1 / line_1[:, :, :2].norm(2, 2) + 1 / line_2[:, :, :2].norm(2, 2)
    )

    return ret


# def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=1.0):
def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=0.5):
    """Robust symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix
        gamma (float, optional): Defaults to 0.5. robust parameter

    Returns:
        tensor: robust symmetric epipolar distance
    """

    sed = symmetric_epipolar_distance(pts1, pts2, fundamental_mat)
    ret = torch.clamp(sed, max=gamma)

    return ret
