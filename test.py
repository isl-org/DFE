"""Script for testing the NormalizedEightPointNet.

    to see help:
    $ python test.py -h

"""

import argparse
import cv2

import numpy as np

import torch

from dfe.datasets import ColmapDataset
from dfe.models import NormalizedEightPointNet
from dfe.utils import compute_residual

from sklearn.metrics import f1_score


def eval_model(pts, side_info, model, device, postprocess=True):
    pts_orig = pts.copy()
    pts = torch.from_numpy(pts).to(device).unsqueeze(0)
    side_info = torch.from_numpy(side_info).to(torch.float).to(device).unsqueeze(0)

    F_est, rescaling_1, rescaling_2, weights = model(pts, side_info)

    F_est = rescaling_1.permute(0, 2, 1).bmm(F_est[-1].bmm(rescaling_2))

    F_est = F_est / F_est[:, -1, -1].unsqueeze(-1).unsqueeze(-1)
    F = F_est[0].data.cpu().numpy()
    weights = weights[0, 0].data.cpu().numpy()

    F_best = F

    if postprocess:
        inliers_best = np.sum(compute_residual(pts_orig, F) <= 1.0)

        for th in [25, 50, 75]:
            perc = np.percentile(weights, th)
            good = np.where(weights > perc)[0]

            if len(good) < 9:
                continue

            pts_ = pts_orig[good]
            F, _ = cv2.findFundamentalMat(pts_[:, 2:], pts_[:, :2], cv2.FM_LMEDS)
            inliers = np.sum(compute_residual(pts_orig, F) <= 1.0)

            if inliers > inliers_best:
                F_best = F
                inliers_best = inliers

    return F_best


def compute_error(pts, model_est, model_gt, th=1.0):
    residuals = compute_residual(pts, model_est)
    residuals_gt = compute_residual(pts, model_gt)

    gt_inliers = pts[residuals_gt <= th]
    rms = np.mean(compute_residual(gt_inliers, model_est))

    il_list = []
    tp_list = []

    inl1 = (residuals) <= th
    inl2 = (residuals_gt) <= th

    tp = f1_score(inl2, inl1)
    inl = np.sum(inl1) / len(residuals)

    il_list.append(inl)
    tp_list.append(tp)

    return il_list, tp_list, rms


def test(options):
    """Test NormalizedEightPointNet.

    Args:
        options: testing options
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device: %s" % device)

    print("-- Data loading --")
    data_sets = []
    for dset_path in options.dataset:
        print('Loading dataset "%s"' % dset_path)
        data_sets.append(
            ColmapDataset(
                dset_path,
                num_points=-1,
                compute_virtual_points=False,
                max_F=100,
                random=False,
            )
        )

    dset = torch.utils.data.ConcatDataset(data_sets)

    print(len(dset))

    print("-- Loading model --")
    print(options.side_info_size)
    model = NormalizedEightPointNet(
        depth=options.depth, side_info_size=options.side_info_size
    )

    model.load_state_dict(torch.load(options.model))
    model.to(device)
    model = model.eval()

    all_il_ours = []
    all_f1_ours = []
    all_rms_ours = []

    idxs = np.random.permutation(len(dset))

    for batch_idx in idxs:
        (pts, side_info, F_gt, _, _) = dset.__getitem__(batch_idx)

        # Compute our result
        with torch.no_grad():
            F_ours = eval_model(pts, side_info, model, device)

        # Evaluate
        il_ours, f1_ours, rms_ours = compute_error(pts, F_ours, F_gt)

        all_il_ours.append(il_ours[0])
        all_f1_ours.append(f1_ours[0])
        all_rms_ours.append(rms_ours)

        print(
            "  (%.4f, %.4f), (%.4f, %.4f), (%.4f, %.4f), (%.4f, %.4f)"
            % (
                np.mean(np.asarray(all_il_ours)),
                il_ours[0],
                np.mean(np.asarray(all_f1_ours)),
                f1_ours[0],
                np.mean(np.asarray(all_rms_ours)),
                rms_ours,
                np.median(np.asarray(all_rms_ours)),
                rms_ours,
            )
        )


if __name__ == "__main__":
    np.random.seed(42)
    PARSER = argparse.ArgumentParser(description="Testing")

    PARSER.add_argument("--depth", type=int, default=3, help="depth")
    PARSER.add_argument(
        "--side_info_size", type=int, default=3, help="size of side information"
    )
    PARSER.add_argument(
        "--dataset", default=["Panther"], nargs="+", help="list of datasets"
    )
    PARSER.add_argument("--num_workers", type=int, default=1, help="number of workers")
    PARSER.add_argument("--model", type=str, required=True, help="model file")

    ARGS = PARSER.parse_args()

    # pytorch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    test(ARGS)
