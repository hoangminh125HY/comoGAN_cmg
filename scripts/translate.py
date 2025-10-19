
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def printProgressBar(i, max, postText):
    n_bar = 20
    j = i / max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


from torch.cuda.amp import autocast

def inference(model, opt, A_path, phi):
    t_phi = torch.tensor(phi)
    A_img = Image.open(A_path).convert('RGB')
    A = get_transform(opt, convert=False)(A_img)
    img_real = (((ToTensor()(A)) * 2) - 1).unsqueeze(0)

    with torch.no_grad(), autocast():
        img_fake = model.forward(img_real.to(device), t_phi.to(device))

    return ToPILImage()((img_fake[0].cpu() + 1) / 2)

def main(cmdline):
    # ---- Đường dẫn trực tiếp đến hparams.yaml và checkpoint ----
    hparams_path = "/kaggle/input/logs-pretrains/hparams.yaml"
    checkpoint_path = "/kaggle/input/logs-pretrains/iter_000000.pth"

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
        phi = cmdline.phi  # dùng tham số từ command line
        out_img = inference(model, opt, os.path.join(cmdline.load_path, path_img), phi)
        save_path = os.path.join(
            cmdline.save_path,
            f"{os.path.splitext(os.path.basename(path_img))[0]}_phi_{phi:.1f}.png"
        )
        out_img.save(save_path)
        # ✅ Giải phóng bộ nhớ để tránh dồn VRAM
        del out_img
        torch.cuda.empty_cache()

        i += 1

if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--load_path', default='/datasets/waymo_comogan/val/sunny/Day/', type=str,
                    help='Path to load the dataset to translate')
    ap.add_argument('--save_path', default='/CoMoGan/images/', type=str,
                    help='Path to save the dataset')
    ap.add_argument('--sequence', default=None, type=str,
                    help='Only process images containing this string')
    ap.add_argument('--phi', default=0.0, type=float,
                    help='Angle of the sun φ between [0,2π]')
    main(ap.parse_args())
    print("\n")


