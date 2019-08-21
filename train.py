"""Script for training the NormalizedEightPointNet.

Example:
    $ python train.py

    to see help:
    $ python train.py -h

"""

import argparse
import time

import torch
import torch.optim as optim

from dfe.datasets import ColmapDataset
from dfe.models import NormalizedEightPointNet
import dfe.models.loss as L


def train(options):
    """Train NormalizedEightPointNet.

    Args:
        options: training options
    """

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device: %s" % device)

    # data
    print("-- Data loading --")
    data_sets = []
    for dset_path in options.dataset:
        print('Loading dataset "%s"' % dset_path)
        data_sets.append(ColmapDataset(dset_path, num_points=1000))
        print("Number of pairs: %d" % len(data_sets[-1]))

    dset = torch.utils.data.ConcatDataset(data_sets)

    print("Total number of training samples: %d" % len(dset))

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    model = NormalizedEightPointNet(
        depth=options.depth, side_info_size=options.side_info_size
    )
    model = model.to(device)

    # loss
    criterion = L.robust_symmetric_epipolar_distance

    # optimizer
    optimizer = optim.Adamax(model.parameters(), lr=options.learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # train
    print("-- Training --")

    model.train()

    for epoch in range(options.num_epochs):

        # init
        num_batches = 0

        avg_loss = 0
        avg_task_loss = 0
        avg_task_loss2 = 0

        time_start = time.time()

        # loop batches
        for batch_idx, (pts, side_info, F_gt, pts1_virt, pts2_virt) in enumerate(
            data_loader
        ):

            # input
            pts = pts.to(device)
            F_gt = F_gt.to(device)
            side_info = side_info.to(torch.float).to(device)
            pts1_virt = pts1_virt.to(torch.float).to(device)
            pts2_virt = pts2_virt.to(torch.float).to(device)

            # step
            model.zero_grad()

            F, rescaling_1, rescaling_2, _ = model(pts, side_info)

            pts1_eval = torch.bmm(rescaling_1, pts1_virt.permute(0, 2, 1)).permute(
                0, 2, 1
            )
            pts2_eval = torch.bmm(rescaling_2, pts2_virt.permute(0, 2, 1)).permute(
                0, 2, 1
            )

            loss = 0

            for depth in range(0, options.depth):
                loss += criterion(pts1_eval, pts2_eval, F[depth]).mean()

            loss.backward()
            optimizer.step()

            num_batches += 1

            # check loss
            F_end = F[options.depth - 1]

            avg_loss += criterion(pts1_eval, pts2_eval, F_end).mean().item()
            avg_task_loss += (
                L.symmetric_epipolar_distance(pts1_eval, pts2_eval, F_end).mean().item()
            )

            # fundamental matrix in image space
            F_est = rescaling_1.permute(0, 2, 1).bmm(F_end.bmm(rescaling_2))
            F_est = F_est / F_est[:, -1, -1].unsqueeze(-1).unsqueeze(-1)

            avg_task_loss2 += (
                L.symmetric_epipolar_distance(pts1_virt, pts2_virt, F_est).mean().item()
            )

            if batch_idx % 1 == 0:
                print(
                    "epoch = %d, iter = %d,  lr = %f, time = %f, loss = %f, task_loss = %f"
                    % (
                        epoch,
                        batch_idx,
                        optimizer.param_groups[0]["lr"],
                        time.time() - time_start,
                        avg_loss / (batch_idx + 1),
                        avg_task_loss2 / (batch_idx + 1),
                    )
                )

        if (epoch + 1) % options.checkpoint_interval == 0:
            print("Writing checkpoint")
            torch.save(model.state_dict(), "checkpoints/model_epoch%04d.pt" % epoch)

        scheduler.step()

    # save model
    print("saving model")
    torch.save(model.state_dict(), options.output)

    print("done")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="Training")

    PARSER.add_argument("--depth", type=int, default=3, help="depth")
    PARSER.add_argument(
        "--side_info_size", type=int, default=3, help="size of side information"
    )
    PARSER.add_argument(
        "--dataset", default=["Family"], nargs="+", help="list of datasets"
    )
    PARSER.add_argument("--output", type=str, default="output.pt", help="output file")
    PARSER.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
    PARSER.add_argument("--batch_size", type=int, default=16, help="batch size")
    PARSER.add_argument("--num_workers", type=int, default=8, help="number of workers")
    PARSER.add_argument(
        "--checkpoint_interval", type=int, default=1, help="checkpoint interval"
    )
    PARSER.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )

    ARGS = PARSER.parse_args()

    # pytorch options

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train(ARGS)
