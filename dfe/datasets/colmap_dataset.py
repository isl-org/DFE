"""Colmap dataset for fundametal matrix estimation. Derived from FundamentalMatrixDataset.
"""

import sqlite3
import numpy as np

from dfe.datasets import FundamentalMatrixDataset
from dfe.utils import colmap_read, colmap_utils


class ColmapDataset(FundamentalMatrixDataset):
    """Colmap dataset for fundametal matrix estimation. Derived from FundamentalMatrixDataset.
    """

    def __init__(
        self,
        path,
        num_points=-1,
        threshold=1,
        max_F=None,
        random=False,
        min_matches=20,
        compute_virtual_points=True,
    ):
        """Init.

        Args:
            path (str): path to dataset folder
            num_points (int, optional): number of points per sample. Defaults to -1.
            threshold (int, optional): epipolar threshold. Defaults to 1.
            max_F (int, optional): maximal number of samples (if None: use all). Defaults to None.
            random (bool, optional): random database access. Defaults to False.
            min_matches (int, optional): minimal number of good matches per sample. Defaults to 20.
        """
        super(ColmapDataset, self).__init__(num_points)

        cameras = colmap_read.read_cameras_binary("%s/sparse/0/cameras.bin" % path)
        images = colmap_read.read_images_binary("%s/sparse/0/images.bin" % path)

        connection = sqlite3.connect("%s/reconstruction.db" % path)
        cursor = connection.cursor()

        self.img_paths = []

        if random:
            cursor.execute(
                "SELECT pair_id, data FROM matches WHERE rows>=? ORDER BY RANDOM();",
                (min_matches,),
            )
        else:
            cursor.execute(
                "SELECT pair_id, data FROM matches WHERE rows>=?;", (min_matches,)
            )

        for row in cursor:
            # max number of image pairs
            if max_F and len(self.F) == max_F:
                break

            img1_id, img2_id = colmap_utils.pair_id_to_image_ids(row[0])

            try:
                img1 = images[img1_id]
                img2 = images[img2_id]
            except KeyError:
                print("Image doesn't match id")
                continue

            # check if both images share enough 3D points
            pts1 = img1.point3D_ids[img1.point3D_ids != -1]
            pts2 = img2.point3D_ids[img2.point3D_ids != -1]

            common = len(np.intersect1d(pts1, pts2))

            if common < min_matches:
                continue

            # get cameras
            K1, T1, sz1 = colmap_utils.get_cam(img1, cameras)
            K2, T2, sz2 = colmap_utils.get_cam(img2, cameras)

            F = colmap_utils.compose_fundamental_matrix(K1, T1, K2, T2)

            # pull the matches
            matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)

            cursor_2 = connection.cursor()
            cursor_2.execute(
                "SELECT data, cols FROM keypoints WHERE image_id=?;", (img1_id,)
            )
            row_2 = next(cursor_2)
            keypoints1 = np.fromstring(row_2[0], dtype=np.float32).reshape(-1, row_2[1])

            cursor_2.execute(
                "SELECT data, cols FROM keypoints WHERE image_id=?;", (img2_id,)
            )
            row_2 = next(cursor_2)
            keypoints2 = np.fromstring(row_2[0], dtype=np.float32).reshape(-1, row_2[1])

            cursor_2.execute(
                "SELECT data FROM descriptors WHERE image_id=?;", (img1_id,)
            )
            row_2 = next(cursor_2)
            descriptor_1 = np.float32(
                np.fromstring(row_2[0], dtype=np.uint8).reshape(-1, 128)
            )

            cursor_2.execute(
                "SELECT data FROM descriptors WHERE image_id=?;", (img2_id,)
            )
            row_2 = next(cursor_2)
            descriptor_2 = np.float32(
                np.fromstring(row_2[0], dtype=np.uint8).reshape(-1, 128)
            )

            dist = np.sqrt(
                np.mean(
                    (descriptor_1[matches[:, 0]] - descriptor_2[matches[:, 1]]) ** 2, 1
                )
            )[..., None]

            rel_scale = np.abs(
                keypoints1[matches[:, 0], 2] - keypoints2[matches[:, 1], 2]
            )[..., None]

            angle1 = keypoints1[matches[:, 0], 3]
            angle2 = keypoints2[matches[:, 1], 3]

            rel_orient = np.minimum(np.abs(angle1 - angle2), np.abs(angle2 - angle1))[
                ..., None
            ]
            # rel_orient = np.abs(angle1 - angle2)[..., None]

            pairs = np.hstack(
                (
                    keypoints1[matches[:, 0], :2],
                    keypoints2[matches[:, 1], :2],
                    dist,
                    rel_scale,
                    rel_orient,
                )
            )
            dist = colmap_utils.compute_residual(pairs[:, :4], F.T)

            if np.sum(np.uint8(dist < threshold)) >= min_matches:
                self.pts.append(pairs)
                self.F.append(F.T)

                self.size_1.append(sz1)
                self.size_2.append(sz2)
                img1_path = "%s/%s" % (path, img1.name)
                img2_path = "%s/%s" % (path, img2.name)

                self.img_paths.append((img1_path, img2_path))

            cursor_2.close()

        cursor.close()
        connection.close()

        if compute_virtual_points:
            self.compute_virtual_points_all()
        else:
            self.pts1_virt = [0.0] * len(self.F)
            self.pts2_virt = [0.0] * len(self.F)


def write_point_cache(self, index):
    pts1_virt, pts2_virt = self.get_virtual_points(index)
    self.pts1_virt[index] = pts1_virt
    self.pts2_virt[index] = pts2_virt

    return index
