import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable TensorFlow logs

from options import get_options
from data import create_dataset
from networks import create_model
from argparse import ArgumentParser as AP

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from util.callbacks import LogAndCheckpointEveryNSteps
from human_id import generate_id


def start(cmdline):
    pl.seed_everything(cmdline.seed)
    opt = get_options(cmdline)

    # --- Dataset ---
    dataset = create_dataset(opt)

    # --- Model ---
    model = create_model(opt)

    # --- Logger & Callback ---
    callbacks = []
    logger = None
    if not cmdline.debug:
        root_dir = os.path.join('logs/', generate_id()) if cmdline.id is None else os.path.join('logs/', cmdline.id)
        logger = TensorBoardLogger(save_dir=os.path.join(root_dir, 'tensorboard'))
        logger.log_hyperparams(opt)
        callbacks.append(
            LogAndCheckpointEveryNSteps(
                save_step_frequency=opt.save_latest_freq,
                viz_frequency=opt.display_freq,
                log_frequency=opt.print_freq,
            )
        )
    else:
        root_dir = os.path.join('/tmp', generate_id())

    precision = 16 if cmdline.mixed_precision else 32

    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        default_root_dir=os.path.join(root_dir, 'checkpoints')
    )

    # --- Load checkpoint nếu có ---
    checkpoint_path = "/kaggle/input/checkpoints1111/logs_comoGAN/iter_000000.pth"

    if os.path.exists(checkpoint_path):
        print(f"➡️ Loading pretrained checkpoint: {checkpoint_path}")
        try:
            model = model.__class__.load_from_checkpoint(checkpoint_path, opt=opt)
        except Exception as e:
            print(f"⚠️ Load checkpoint thất bại ({e}), train từ đầu.")
    else:
        print("⚠️ Không tìm thấy checkpoint, train từ đầu.")

    # --- Freeze Discriminator ---
    if hasattr(model, 'netD'):
        for param in model.netD.parameters():
            param.requires_grad = False
        print("🧊 Discriminator frozen (fine-tuning Generator only)")

    # --- Train ---
    trainer.fit(model, dataset)


if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--id', default=None, type=str, help='Set an existing uuid to resume a training')
    ap.add_argument('--debug', default=False, action='store_true', help='Disables experiment saving')
    ap.add_argument('--gpus', default=[0], type=int, nargs='+', help='gpus to train on')
    ap.add_argument('--model', default='comomunit', type=str, help='Choose model for training')
    ap.add_argument('--data_importer', default='day2timelapse', type=str, help='Module name of the dataset importer')
    ap.add_argument('--path_data', default='/content/PPE-detection-8/train', type=str, help='Path to the dataset')
    ap.add_argument('--learning_rate', default=0.00005, type=float, help='Learning rate')
    ap.add_argument('--scheduler_policy', default='step', type=str, help='Scheduler policy')
    ap.add_argument('--decay_iters_step', default=200000, type=int, help='Decay iterations step')
    ap.add_argument('--decay_step_gamma', default=0.5, type=float, help='Decay step gamma')
    ap.add_argument('--seed', default=1, type=int, help='Random seed')
    ap.add_argument('--mixed_precision', default=False, action='store_true', help='Use mixed precision to reduce memory usage')

    start(ap.parse_args())
