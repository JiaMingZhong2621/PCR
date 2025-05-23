from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, RandomResizedCrop,ColorJitter,
    RandomHorizontalFlip
)
from torchvision.transforms.functional import InterpolationMode


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


SIZE = (224, 224)
# Mode of interpolation in resize functions
INTERPOLATION = INTERPOLATION_MODES["bicubic"]
# Mean and std (default: CoOp)
PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
# Random crop
CROP_PADDING = 0
# Random resized crop
RRCROP_SCALE = (0.08, 1.0)
import torch
import torchvision.transforms.functional as F
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.4):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0., 1.)

class AdjustBrightness(object):
    def __init__(self, factor_range=(0.6, 1.4)):
        self.factor_range = factor_range

    def __call__(self, img):
        factor = random.uniform(*self.factor_range)
        return F.adjust_brightness(img, factor)

def build_transform(image_augmentation,
                    size=SIZE,
                    interpolation=INTERPOLATION,
                    pixel_mean=PIXEL_MEAN,
                    pixel_std=PIXEL_STD,
                    crop_padding=CROP_PADDING,
                    rrcrop_scale=RRCROP_SCALE):
    """Build transformation function.

    Args:
        image_augmentation (str): name of image augmentation method. If none, just use center crop.
    """
    normalize = Normalize(mean=pixel_mean, std=pixel_std)

    if image_augmentation == "none":
        # center crop
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            CenterCrop(size=size),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "flip":
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            CenterCrop(size=size),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "randomcrop":
        transform = Compose([
            Resize(size=max(size), interpolation=interpolation),
            RandomCrop(size=size, padding=crop_padding),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize,
        ])
    elif image_augmentation == "randomresizedcrop":
        transform = Compose([
            RandomResizedCrop(size=size, scale=rrcrop_scale, interpolation=interpolation),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize,
        ])
    else:
        raise ValueError("Invalid image augmentation method: {}".format(image_augmentation))
        
    return transform