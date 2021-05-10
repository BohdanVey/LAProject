import os
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader

from cae.src.data_loader import ImageFolder720p
from cae.src.utils import save_imgs

from bagoftools.namespace import Namespace
from bagoftools.logger import Logger

from cae.src.models.cae_32x32x32_zero_pad_bin import CAE

ROOT_EXP_DIR = "experiments"

logger = Logger(__name__, colorize=True)


def create_root_dir(dir):
    path = ''
    for i in dir.split('/'):
        if path != '':
            path += '/'
        path += i
        if not os.path.exists(path):
            os.mkdir(path)


def test_image(image_src, model=None, loss_criterion=None):
    get_image = ImageFolder720p.get_src(image_src[0])
    img, patches, _ = get_image
    patches = patches.unsqueeze(0)
    patches = patches.cuda()
    out = T.zeros(6, 10, 3, 128, 128)
    avg_loss = 0

    for i in range(6):
        for j in range(10):
            x = patches[:, :, i, j, :, :].cuda()
            y = model(x)
            out[i, j] = y.data
            if loss_criterion:
                loss = loss_criterion(y, x)
                avg_loss += (1 / 60) * loss.item()

    # save output
    out = np.transpose(out, (0, 3, 1, 4, 2))
    out = np.reshape(out, (768, 1280, 3))
    out = np.transpose(out, (2, 0, 1))
    if loss_criterion:
        return out, avg_loss
    else:
        return out


def test(cfg: Namespace) -> None:
    assert cfg.checkpoint not in [None, ""]
    assert cfg.device == "cpu" or (cfg.device == "cuda" and T.cuda.is_available())

    exp_dir = ROOT_EXP_DIR + '/' + cfg.exp_name
    create_root_dir(exp_dir + '/out')
    logger.info(f"[exp dir={exp_dir}]")

    model = CAE()
    model.load_state_dict(T.load(cfg.checkpoint))
    model.eval()
    if cfg.device == "cuda":
        model.cuda()
    logger.info(f"[model={cfg.checkpoint}] on {cfg.device}")

    dataloader = DataLoader(
        dataset=ImageFolder720p(cfg.dataset_path), batch_size=1, shuffle=cfg.shuffle
    )
    logger.info(f"[dataset={cfg.dataset_path}]")

    loss_criterion = nn.MSELoss()

    for batch_idx, data in enumerate(dataloader, start=1):
        img, patches, path = data
        out, avg_loss = test_image(path, model, loss_criterion)
        logger.debug("[%5d/%5d] avg_loss: %f", batch_idx, len(dataloader), avg_loss)
        print(path)
        save_imgs(
            imgs=out.unsqueeze(0),
            to_size=(3, 768, 1280),
            name=exp_dir + f"/out/{path[0].split('/')[-1]}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "rt") as fp:
        cfg = Namespace(**yaml.safe_load(fp))
    test(cfg)
