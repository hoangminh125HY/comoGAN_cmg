"""
base_dataset.py:
All datasets are a subclass of BaseDataset and implement abstract methods.
Includes augmentation strategies which can be used at sampling time.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def __resize_max(img, max_size=1, method=Image.BICUBIC):
    ow, oh = img.size
    scale = min(max_size / ow, max_size / oh, 1.0)
    new_w, new_h = int(ow * scale), int(oh * scale)
    print(f"[DEBUG] Resize: ({ow}, {oh}) -> ({new_w}, {new_h})")
    return img.resize((new_w, new_h), method)


def __resize_and_pad(img, size=512, method=Image.BICUBIC):
    ow, oh = img.size
    scale = min(size / ow, size / oh)
    new_w, new_h = int(ow * scale), int(oh * scale)
    img = img.resize((new_w, new_h), method)

    # Pad cho đủ (256, 256)
    new_img = Image.new("RGB", (size, size))
    new_img.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return new_img


def get_transform(opt=None, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    # 🔥 Resize + Pad
    transform_list.append(transforms.Lambda(lambda img: __resize_and_pad(img, 512, method)))

    transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size (only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        logger.warning(
            f"The image size needs to be a multiple of 4. "
            f"The loaded image size was ({ow}, {oh}), so it was adjusted to "
            f"({w}, {h}). This adjustment will be done to all images "
            f"whose sizes are not multiples of 4"
        )
        __print_size_warning.has_printed = True

