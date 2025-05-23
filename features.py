import os

from engine.config import parser

import torch

from engine.tools.utils import makedirs, set_random_seed
from engine.transforms.default import build_transform
from engine.datasets.utils import DatasetWrapper, get_few_shot_setup_name, get_few_shot_benchmark
from engine import clip
from engine.clip import partial_model

def get_backbone_name(clip_encoder):
    return clip_encoder.replace("/", "-")


def get_image_encoder_name(clip_encoder, image_layer_idx):
    return "_".join([get_backbone_name(clip_encoder), str(image_layer_idx)])


def get_view_name(image_augmentation, image_views=1):
    name = f"{image_augmentation}"
    if image_augmentation != "none":
        assert image_views > 0
        name += f"_view_{image_views}"
    return name


def get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx):
    image_encoder_path = os.path.join(
        feature_dir,
        'image',
        get_image_encoder_name(clip_encoder, image_layer_idx)
    )
    return image_encoder_path


def get_image_features_path(dataset,
                            feature_dir,
                            clip_encoder,
                            image_layer_idx,
                            image_augmentation,
                            image_views=1):
    image_features_path = os.path.join(
        get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx),
        dataset,
        get_view_name(image_augmentation, image_views),
        f"get_few_shot_setup_name.pth")
    return image_features_path


def get_test_features_path(dataset,
                           feature_dir,
                           clip_encoder,
                           image_layer_idx):
    test_features_path = os.path.join(
        get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx),
        dataset,
        "test.pth"
    )
    return test_features_path


def extract_features(image_encoder, data_source, transform, num_views=1, test_batch_size=32, num_workers=4):
    features_dict = {
        'features': torch.Tensor(),
        'labels': torch.Tensor(),
        'paths': [],
    }
    ######################################
    #   Setup DataLoader
    ######################################
    loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source, transform=transform),
        batch_size=test_batch_size,
        sampler=None,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    ########################################
    # Start Feature Extractor
    ########################################
    image_encoder.feature_extractor.eval()

    with torch.no_grad():
        for _ in range(num_views):
            for batch_idx, batch in enumerate(loader):
                data = batch["img"].cuda()
                feature = image_encoder.feature_extractor(data) # This is not L2 normed
                feature = feature.cpu()
                if batch_idx == 0:
                    features_dict['features'] = feature
                    features_dict['labels'] = batch['label']
                    features_dict['paths'] = batch['impath']
                else:
                    features_dict['features'] = torch.cat((features_dict['features'], feature), 0)
                    features_dict['labels'] = torch.cat((features_dict['labels'], batch['label']))
                    features_dict['paths'] = features_dict['paths'] + list(batch['impath'])
    return features_dict

def get_image_encoder(clip_model, args):
    image_encoder_dir = get_image_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    makedirs(image_encoder_dir)
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")
    # Check if image partial model exists already
    if os.path.exists(image_encoder_path):
        print(f"Image encoder already saved at {image_encoder_path}")
        image_encoder = torch.load(image_encoder_path)
    else:
        print(f"Saving image encoder to {image_encoder_path}")
        image_encoder = partial_model.get_image_encoder(
            args.clip_encoder,
            args.image_layer_idx,
            clip_model
        )
        torch.save(image_encoder, image_encoder_path)
    return image_encoder


def prepare_few_shot_image_features(clip_model, args, benchmark_train, benchmark_val):
    image_encoder = get_image_encoder(clip_model, args)
    # Check if (image) features are saved already
    image_features_path = get_image_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx,
        args.image_augmentation,
        image_views=args.image_views
    )

    makedirs(os.path.dirname(image_features_path))
    
    # import pdb; pdb.set_trace()
    if os.path.exists(image_features_path):
        print(f"Features already saved at {image_features_path}")
    else:
        print(f"Saving features to {image_features_path}")
        image_features = {
            'train': {},
            'val': {},
        }
        train_transform = build_transform(args.image_augmentation)
        test_transform = build_transform('none')
        print(f"Extracting features for train split ...")
        if args.image_augmentation == 'none':
            num_views = 1
        else:
            num_views = args.image_views
        assert num_views > 0, "Number of views must be greater than 0"
        image_features['train'] = extract_features(
            image_encoder, benchmark_train, 
            train_transform, num_views=num_views, test_batch_size=args.test_batch_size, num_workers=args.num_workers)
        
        print(f"Extracting features for val split ...")
        image_features['val'] = extract_features(
            image_encoder, benchmark_val,
            test_transform, num_views=1, test_batch_size=args.test_batch_size, num_workers=args.num_workers)
    
        torch.save(image_features, image_features_path)


def prepare_test_image_features(clip_model, args, benchmark_test):
    image_encoder = get_image_encoder(clip_model, args)
    # Check if features are saved already
    test_features_path = get_test_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )

    makedirs(os.path.dirname(test_features_path))
    if os.path.exists(test_features_path):
        print(f"Test features already saved at {test_features_path}")
    else:
        print(f"Saving features to {test_features_path}")
        test_transform = build_transform('none')
        print(f"Extracting features for test split ...")
        test_features = extract_features(
            image_encoder, 
            benchmark_test, test_transform,
            num_views=1, test_batch_size=args.test_batch_size, num_workers=args.num_workers)
        torch.save(test_features, test_features_path)


def main(args):
    set_random_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    ########################################
    #   Train/Val/Test Split
    ########################################
    few_shot_benchmark = get_few_shot_benchmark(
        args.data_dir,
        args.indices_dir,
        args.dataset
    )
    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load(args.clip_encoder, jit=False)
    clip_model.float()
    clip_model.eval()
    print(type(clip_model))
    ########################################
    #   Feature Extraction
    ########################################

    prepare_few_shot_image_features(clip_model, args, few_shot_benchmark['train'], few_shot_benchmark['val'])

    prepare_test_image_features(clip_model, args, few_shot_benchmark['test'])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)