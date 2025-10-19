#!/usr/bin/env python
# coding: utf-8

import pathlib
import torch
import yaml
import sys
import os

from math import pi
from PIL import Image
from munch import Munch
from argparse import ArgumentParser as AP
from torchvision.transforms import ToPILImage, ToTensor

p_mod = str(pathlib.Path('.').absolute())
sys.path.append(p_mod.replace("/scripts", ""))

from data.base_dataset import get_transform
from networks import create_model

device='cuda' if torch.cuda.is_available() else 'cpu'
def printProgressBar(i, max, postText):
    n_bar = 20 # size of progress bar
    j = i / max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

def inference(model, opt, A_path, phi):
    t_phi = torch.tensor(phi)
    A_img = Image.open(A_path).convert('RGB')
    A = get_transform(opt, convert=False)(A_img)
    img_real = (((ToTensor()(A)) * 2) - 1).unsqueeze(0)
    img_fake = model.forward(img_real.to(device), t_phi.to(device))

    return ToPILImage()((img_fake[0].cpu() + 1) / 2)

def main(cmdline):
    # ---- Đường dẫn trực tiếp đến hparams.yaml và checkpoint ----
    hparams_path = "/content/CoMoGAN_Modified/logs/remain-low-lot-door/checkpoints/lightning_logs/version_0/hparams.yaml"
    checkpoint_path = "/content/CoMoGAN_Modified/logs/pretrained/tensorboard/default/version_0/checkpoints/iter_000000.pth"

    print(f"Loading hparams from {hparams_path}")
    print(f"Loading checkpoint from {checkpoint_path}")

    # ---- Load parameters ----
    with open(hparams_path) as cfg_file:
        opt = Munch(yaml.safe_load(cfg_file))
    opt.no_flip = True

    # ---- Load model từ checkpoint ----
    model_class = create_model(opt).__class__
    model = model_class.load_from_checkpoint(checkpoint_path, opt=opt)
    model.to(device)

    # ---- Load dataset ----
    p = pathlib.Path(cmdline.load_path)
    dataset_paths = [str(x.relative_to(cmdline.load_path)) for x in p.iterdir()]
    dataset_paths.sort()

    if cmdline.sequence is not None:
        sequence_name = [f for f in dataset_paths if cmdline.sequence in f]
    else:
        sequence_name = dataset_paths

    os.makedirs(cmdline.save_path, exist_ok=True)

    i = 0
    for path_img in sequence_name:
        printProgressBar(i, len(sequence_name), path_img)
        # Loop over phi values from 0 to 2pi with increments of 0.2
        for phi in torch.arange(0, 2 * pi, 0.2):
            # Forward our image into the model with the specified ɸ
            out_img = inference(model, opt, os.path.join(cmdline.load_path, path_img), phi)
            # Saving the generated image with phi in the filename
            save_path = os.path.join(cmdline.save_path, f"{os.path.splitext(os.path.basename(path_img))[0]}_phi_{phi:.1f}.png")
            out_img.save(save_path)
        i += 1

if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--load_path', default='/datasets/waymo_comogan/val/sunny/Day/', type=str, help='Set a path to load the dataset to translate')
    ap.add_argument('--save_path', default='/CoMoGan/images/', type=str, help='Set a path to save the dataset')
    ap.add_argument('--sequence', default=None, type=str, help='Set a sequence, will only use the image that contained the string specified')
    ap.add_argument('--checkpoint', default=None, type=str, help='Set a path to the checkpoint that you want to use')
    ap.add_argument('--phi', default=0.0, type=float, help='Choose the angle of the sun 𝜙 between [0,2𝜋], which maps to a sun elevation ∈ [+30◦,−40◦]')
    main(ap.parse_args())
    print("\n")
