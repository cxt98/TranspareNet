import torch

import argparse

import numpy as np
np.random.seed(0)

import random
random.seed(0)
import os
import re
import torch
torch.manual_seed(0)

from saic_depth_completion.engine.train import train
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.utils.tracker import ComposedTracker, Tracker
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, SSIM, DepthL2Loss, DepthL1Loss, DepthRel
from saic_depth_completion.data.datasets.clearpose import ClearPose

# def find_weights(path_to_weights_folder):
#     weights_files = os.listdir(path_to_weights_folder)
#     if len(weights_files) == 0:
#         return None,None
#     else:
#         weights_files.sort(key=lambda f: int(re.sub('\D', '', f)))
#         #weights_files.sort(key=lambda f: int(filter(str.isdigit, f)))
#         return weights_files[-1],len(weights_files)

def find_weights(path_to_weights_folder):
    weights_files = os.listdir(path_to_weights_folder)
    if len(weights_files) == 0:
        return None,None
    else:
        weights_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        epoch_resume = int(re.sub('\D', '', weights_files[-1]))
        print('resuming from epoch: ', epoch_resume+1)
        #weights_files.sort(key=lambda f: int(filter(str.isdigit, f)))
        return os.path.join(path_to_weights_folder, weights_files[-1]),epoch_resume+1
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(processed = False):
    parser = argparse.ArgumentParser(description="Some training params.")
    parser.add_argument(
        "--debug", dest="debug", type=bool, default=False, help="Setup debug mode"
    )
    parser.add_argument(
        "--postfix", dest="postfix", type=str, default="", help="Postfix for experiment's name"
    )
    parser.add_argument(
        "--default_cfg", dest="default_cfg", type=str, default="DM-LRN", help="Default config"
    )
    parser.add_argument(
        "--config_file", default="configs/dm_lrn/DM-LRN_efficientnet-b4_pepper.yaml", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--snapshot_period", default=1, type=int, help="Snapshot model one time over snapshot period"
    )
    parser.add_argument(
        "--image_dim", default='small', type=str, help="Image dimension for training {small, large}"
    )
    args = parser.parse_args()

    cfg = get_default_config(args.default_cfg) # seems read from this file: saic_depth_completion/config/dm_lrn.py

    print('cwd: ', os.getcwd())
    #cfg.merge_from_file(os.path.join(os.getcwd(),args.config_file))
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)

    logger = setup_logger()
    experiment = setup_experiment(
        cfg, args.config_file, logger=logger, training=True, debug=args.debug, postfix=args.postfix
    )

    optimizer  = torch.optim.Adam(
        params=model.parameters(), lr=cfg.train.lr
    )
    if not args.debug:
        snapshoter = Snapshoter(
            model, optimizer, period=args.snapshot_period, logger=logger, save_dir=experiment.snapshot_dir
        )
        tensorboard = Tensorboard(experiment.tensorboard_dir)
        tracker = ComposedTracker([
            Tracker(subset="test_cleargrasp", target="mse", snapshoter=snapshoter, eps=0.01),
            Tracker(subset="val_cleargrasp", target="mse", snapshoter=snapshoter, eps=0.01),
        ])
    else:
        snapshoter, tensorboard, tracker = None, None, None


    # if args.image_dim == 'small':
    #     from saic_depth_completion.data.datasets.cleargrasp_small import ClearGrasp
    # elif args.image_dim =='large':
    #     from saic_depth_completion.data.datasets.cleargrasp import ClearGrasp
    # else:
    #     raise ValueError('Provided image_dim argument is invalid')

    metrics = {
        'mse': DepthL2Loss(),
        'mae': DepthL1Loss(),
        'd105': Miss(1.05),
        'd110': Miss(1.10),
        'd125_1': Miss(1.25),
        'd125_2': Miss(1.25**2),
        'd125_3': Miss(1.25**3),
        'rel': DepthRel(),
        'ssim': SSIM(),
    }

    train_dataset = ClearPose(split="train", processed = processed, small_large=args.image_dim)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=default_collate,
        worker_init_fn=seed_worker
    )

    val_datasets = {
        "val_cleargrasp": ClearPose(split="val",processed = processed), # make the same for ClearPose with test set
        # "test_cleargrasp": ClearPose(split="test",processed = processed),
    }
    val_loaders = {
        k: torch.utils.data.DataLoader(
            dataset=v,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=default_collate,
            worker_init_fn=seed_worker
        )
        for k, v in val_datasets.items()
    }
    # Load the checkpoint to continue training if needed
    # Get the most recent set of weights from log file if exists
    path_to_weights_folder = experiment.snapshot_dir
    weights_path, start_epoch = find_weights(path_to_weights_folder)

    init_epoch = 0
    if weights_path:
        print('Loading weights from: ', weights_path)
        snapshoter.load(weights_path)
        init_epoch = start_epoch
    # Get the epoch number which the weights are from


    train(
        model,
        train_loader,
        val_loaders=val_loaders,
        optimizer=optimizer,
        snapshoter=snapshoter,
        epochs=10, #TODO: add into config
        init_epoch=init_epoch,
        logger=logger,
        metrics=metrics,
        tensorboard=tensorboard,
        tracker=tracker
    )


if __name__ == "__main__":
    main()