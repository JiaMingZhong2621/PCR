from engine.config import default
from engine.datasets import dataset_classes
import argparse
parser = argparse.ArgumentParser()
###########################
# Directory Config (modify if using your own paths)
###########################
parser.add_argument(
    "--data_dir",
    type=str,
    default=default.DATA_DIR,
    help="where the dataset is saved",
)
parser.add_argument(
    "--indices_dir",
    type=str,
    default=default.FEW_SHOT_DIR,
    help="where the (few-shot) indices is saved",
)
parser.add_argument(
    "--feature_dir",
    type=str,
    default=default.FEATURE_DIR,
    help="where to save pre-extracted features",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default=default.RESULT_DIR,
    help="where to save experiment results",
)

###########################
# Dataset Config (few_shot_split.py)
###########################
parser.add_argument(
    "--dataset",
    type=str,
    default="PCR",
    choices=dataset_classes.keys(),
    help="number of train shot",
)
parser.add_argument(
    "--seed",
    type=int,
    default=3,
    help="seed number",
)

###########################
# Feature Extraction Config (features.py)
###########################

parser.add_argument(
    "--clip-encoder",
    type=str,
    default="RN50",
    choices=["ViT-B/16", "ViT-B/32", "RN50", "RN101"],
    help="specify the clip encoder to use",
)
parser.add_argument(
    "--image-layer-idx",
    type=int,
    default=0,
    choices=[0, 1, -1],
    help="specify how many image encoder layers to finetune. 0 means none. -1 means full finetuning.",
)
parser.add_argument(
    "--image-augmentation",
    type=str,
    default='randomcrop',
    choices=['none', # only a single center crop
             'flip', # add random flip view
             'randomcrop', # add random crop view
             ],
    help="specify the image augmentation to use.",
)
parser.add_argument(
    "--image-views",
    type=int,
    default=1,
    help="if image-augmentation is not None, then specify the number of extra views.",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=32,
    help="batch size for test (feature extraction and evaluation)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="number of workers for dataloader",
)


###########################
# Training Config (train.py)
###########################

parser.add_argument(
    "--classifier_head",
    type=str,
    default="adapter",
    help="classifier head architecture",
)
parser.add_argument(
    "--classifier_init",
    type=str,
    default="zeroshot",
    choices=["zeroshot", # zero-shot/one-shot-text-based initialization
             "random", # random initialization
    ],
    help="classifier head initialization",
)
parser.add_argument(
    "--logit",
    type=float,
    default=4.60517, # CLIP's default logit scaling
    choices=[4.60517, # CLIP's default logit scaling
             4.0, # for partial finetuning
    ],
    help="logit scale (exp(logit) is the inverse softmax temperature)",
)
parser.add_argument(
    "--hyperparams",
    type=str,
    default="adapter",
    help="hyperparams sweep",
)
