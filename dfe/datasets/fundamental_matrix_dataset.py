"""Base class for a fundamental matrix estimation dataset.
"""
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset
import numpy as np
import cv2


class FundamentalMatrixDataset(Dataset):
    """Base class for a fundamental matrix estimation dataset.
    """

    def __init__(self, num_points, step=0.01):
        """Init.

        Args:
            num_points (int): number of points per sample
            step (float, optional): Step size of virtual evaluation points. Defaults to 0.01.
        """
        self.num_points = num_points
        self.step = step

        self.pts = []
        self.F = []

        self.size_1 = []
        self.size_2 = []

        self.num_points_eval = 0
        self.pts1_virt = []
        self.pts2_virt = []
        self.pts1_grid = []
        self.pts2_grid = []

    def write_point_cache(self, index):
        """Helper method for parallel virtual point computation.

        Args:
            index (int): sample index

        Returns:
            int: index
        """
        pts1_virt, pts2_virt = self.compute_virtual_points(index)

        self.pts1_virt[index] = pts1_virt
        self.pts2_virt[index] = pts2_virt

        return index

    def compute_virtual_points(self, index):
        """Compute virtual points for a single sample.

        Args:
            index (int): sample index

        Returns:
            tuple: virtual points in first image, virtual points in second image
        """
        pts2_virt, pts1_virt = cv2.correctMatches(
            self.F[index], self.pts2_grid[index], self.pts1_grid[index]
        )

        valid_1 = np.logical_and(
            np.logical_not(np.isnan(pts1_virt[:, :, 0])),
            np.logical_not(np.isnan(pts1_virt[:, :, 1])),
        )
        valid_2 = np.logical_and(
            np.logical_not(np.isnan(pts2_virt[:, :, 0])),
            np.logical_not(np.isnan(pts2_virt[:, :, 1])),
        )

        _, valid_idx = np.where(np.logical_and(valid_1, valid_2))
        good_pts = len(valid_idx)

        while good_pts < self.num_points_eval:
            valid_idx = np.hstack(
                (valid_idx, valid_idx[: (self.num_points_eval - good_pts)])
            )
            good_pts = len(valid_idx)

        valid_idx = valid_idx[: self.num_points_eval]

        pts1_virt = pts1_virt[:, valid_idx]
        pts2_virt = pts2_virt[:, valid_idx]

        ones = np.ones((pts1_virt.shape[1], 1))

        pts1_virt = np.hstack((pts1_virt[0], ones))
        pts2_virt = np.hstack((pts2_virt[0], ones))

        return pts1_virt, pts2_virt

    def compute_virtual_points_all(self):
        """Compute virtual points for all samples.
        """
        # set grid points for each image
        grid_x, grid_y = np.meshgrid(
            np.arange(0, 1, self.step), np.arange(0, 1, self.step)
        )
        self.num_points_eval = len(grid_x.flatten())

        for size_1, size_2 in zip(self.size_1, self.size_2):
            pts1_grid = np.float32(
                np.vstack(
                    (size_1[0] * grid_x.flatten(), size_1[1] * grid_y.flatten())
                ).T
            )
            pts2_grid = np.float32(
                np.vstack(
                    (size_2[0] * grid_x.flatten(), size_2[1] * grid_y.flatten())
                ).T
            )

            self.pts1_grid.append(pts1_grid[np.newaxis, :, :])
            self.pts2_grid.append(pts2_grid[np.newaxis, :, :])

        # make grid points fit to epipolar constraint
        self.pts1_virt = [None] * len(self.F)
        self.pts2_virt = [None] * len(self.F)

        pool = ThreadPool(processes=cpu_count())

        pool.map(self.write_point_cache, range(len(self.F)))

        return

    def __getitem__(self, index):
        """Get dataset sample.

        Args:
            index (int): sample index

        Returns:
            tuple: points, side information, fundamental matrix, virtual points 1, virtual points 2
        """
        pts = self.pts[index]
        F = self.F[index]
        pts1_virt = self.pts1_virt[index]
        pts2_virt = self.pts2_virt[index]

        # print(self.img_paths[index])

        # add data if too small for training
        if self.num_points > 0 and pts.shape[0] < self.num_points:
            while pts.shape[0] < self.num_points:
                num_missing = self.num_points - pts.shape[0]
                idx = np.random.permutation(pts.shape[0])[:num_missing]

                pts_pert = pts[idx]
                pts = np.concatenate((pts, pts_pert), 0)

        # normalize side information
        side_info = pts[:, 4:] / np.amax(pts[:, 4:], 0)

        pts = pts[:, :4]

        # remove data if too big for training
        if self.num_points > 0 and (pts.shape[0] > self.num_points):
            idx = np.random.permutation(pts.shape[0])[: self.num_points]

            pts = pts[idx, :]
            side_info = side_info[idx]

        return (np.float32(pts[:, :4]), side_info, np.float32(F), pts1_virt, pts2_virt)

    def __len__(self):
        """Get length of dataset.

        Returns:
            int: length
        """
        return len(self.F)
