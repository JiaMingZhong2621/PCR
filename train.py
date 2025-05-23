from copy import deepcopy
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from engine.config import parser
from engine.tools.utils import makedirs, set_random_seed
from engine.datasets.utils import TensorDataset
from engine.model.head import make_classifier_head
from engine.model.logit import LogitHead,LogitScale
from engine.optimizer.default import HYPER_DICT
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from features import get_backbone_name, \
                     get_few_shot_setup_name, \
                     get_view_name, \
                     get_image_features_path, \
                     get_image_encoder_dir, \
                     get_test_features_path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
torch.set_num_threads(4) # To maximize efficiency, please tune the number of threads for your machine

CROSS_MODAL_BATCH_RATIO = 0.5 # Half of the batch is image, the other half is text
EVAL_FREQ = 100 # Evaluate on val set per 100 iterations (for early stopping)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_benchmark_name(dataset):
    benchmark_name = "-".join([
        dataset,
        get_few_shot_setup_name(dataset)
    ])
    return benchmark_name


def get_modality_name(
                      clip_encoder,
                      image_augmentation,
                      image_layer_idx,
                      image_views=1):
    image_feature_name = f"image_{image_layer_idx}_{get_view_name(image_augmentation, image_views=image_views)}"
    return os.path.join(
        get_backbone_name(clip_encoder),
        image_feature_name
    )


def get_architecture_name(classifier_head, classifier_init):
    return classifier_head + "_" + classifier_init


def get_logit_name(logit):
    name = f"logit_{logit}"
    return name


def get_save_dir(args):
    save_dir = os.path.join(
        args.result_dir,
        get_benchmark_name(
            args.dataset),
        get_modality_name(
            args.clip_encoder,
            args.image_augmentation,
            args.image_layer_idx,
            image_views=args.image_views
        ),
        get_architecture_name(
            args.classifier_head,
            args.classifier_init
        ),
        get_logit_name(
            args.logit
        ),
    )
    return save_dir


def get_hyperparams_str(optim,
                        lr,
                        wd,
                        batch_size,
                        iters):
    hyperparams_str = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}"
    return hyperparams_str

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
    class_mask = torch.reshape(class_mask_indices, (-1,))
    return class_mask
def get_eval_heads(head,logit=None):
    logit_head = LogitHead(
        deepcopy(head),
        logit_scale=logit,
    )

    eval_heads = {
        'head': logit_head.cuda().eval(),
    }
    return eval_heads

def train(logit_head, image_encoder,
          image_loader, val_loader,
          optimizer, scheduler, criterion, iters ,
          eval_freq=EVAL_FREQ, device="cuda"):
    if image_loader is not None:
        image_loader_iter = iter(image_loader)
    else:
        image_loader_iter = None

    print_result_dict = {
        'iter': [],
        'test_accs': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    result_dict = {
        "iter": None,
        "val_acc": None,
        "image_encoder": None,
        "logit_head": None,
    }

    for i in range(iters):
        logit_head.train()
        image_encoder.train()
        if image_loader_iter is not None:
            try:
                image, image_label = next(image_loader_iter)
            except StopIteration:
                image_loader_iter = iter(image_loader)
                image, image_label = next(image_loader_iter)
            image = image.to(device)
            image_label = image_label.to(device)
            image_feature = image_encoder(image)
        else:
            image_feature = None


        if image_feature is not None:
            feature = image_feature
            label = image_label
        else:
            raise ValueError("Both image_feature and text_feature are None")

        logit = logit_head(feature)
        loss = criterion(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % eval_freq == 0:
            val_acc, precision, recall, f1 = validate(logit_head, image_encoder, val_loader, device=device)
            # print(
            #     f"Iteration: {i}, Val Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print_result_dict['iter'].append(i)
            print_result_dict['test_accs'].append(val_acc)
            print_result_dict['precision'].append(precision)
            print_result_dict['recall'].append(recall)
            print_result_dict['f1'].append(f1)
            # 更新最佳模型
            if result_dict["val_acc"] is None or val_acc > result_dict["val_acc"]:
                print(f"New best model found at iteration {i} with val_acc: {val_acc:.4f}")
                result_dict["iter"] = i
                result_dict["val_acc"] = val_acc
                result_dict["image_encoder"] = deepcopy(image_encoder.state_dict())
                result_dict["logit_head"] = deepcopy(logit_head.state_dict())
    #load best model
    image_encoder.load_state_dict(result_dict["image_encoder"])
    logit_head.load_state_dict(result_dict["logit_head"])
    # val_acc = validate(logit_head, image_encoder, val_loader,scale_head, device=device)
    print(f"Best val acc: {result_dict['val_acc']:.4f} at iter {result_dict['iter']}")

    return result_dict

def validate(logit_head, image_encoder, val_loader, device="cuda"):
    with torch.no_grad():
        logit_head.eval()
        image_encoder.eval()

        val_acc = 0.0
        val_count = 0.0
        all_labels = []
        all_preds = []

        for image, image_label in val_loader:
            image = image.to(device)
            image_label = image_label.to(device)

            image_feature = image_encoder(image)
            logit = logit_head(image_feature)
            pred = torch.argmax(logit, dim=1)

            val_acc += torch.sum(pred == image_label).item()
            val_count += image_label.size(0)

            all_labels.extend(image_label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

            image.cpu()

        val_acc /= val_count

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

    return val_acc, precision, recall, f1

def test(logit_head, image_encoder, val_loader, device="cuda", class_names=["1", "5", "10", "15", "20"]):
    with torch.no_grad():
        logit_head.eval()
        image_encoder.eval()

        val_acc = 0.0
        val_count = 0.0
        all_labels = []
        all_preds = []

        for image, image_label in val_loader:
            image = image.to(device)
            image_label = image_label.to(device)

            image_feature = image_encoder(image)
            logit = logit_head(image_feature)
            pred = torch.argmax(logit, dim=1)
            val_acc += (pred == image_label).sum().item()
            val_count += image_label.size(0)

            all_labels.extend(image_label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

            image.cpu()


        val_acc /= val_count
        val_acc *=100
        precision_per_class = precision_score(all_labels, all_preds, average=None) * 100
        recall_per_class = recall_score(all_labels, all_preds, average=None) * 100
        f1_per_class = f1_score(all_labels, all_preds, average=None) * 100

        precision_weighted = precision_score(all_labels, all_preds, average='weighted') * 100
        recall_weighted = recall_score(all_labels, all_preds, average='weighted') * 100
        f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100

        print("Per-Class Metrics:")
        for i, class_name in enumerate(class_names):
            print(f"Class {class_name}: Precision={precision_per_class[i]:.2f}%, Recall={recall_per_class[i]:.2f}%, F1={f1_per_class[i]:.2f}%")

        print("\nWeighted Average Metrics:")
        print(f"Precision: {precision_weighted:.2f}%, Recall: {recall_weighted:.2f}%, F1: {f1_weighted:.2f}%")

    return val_acc, precision_weighted, recall_weighted, f1_weighted


def main(args):
    set_random_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    image_encoder_dir = get_image_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")

    ccrop_features_path = get_image_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx,
        "none",
    )
    ccrop_features = torch.load(ccrop_features_path)

    if args.image_augmentation == "none":
        train_features = ccrop_features['train']['features']
        train_labels = ccrop_features['train']['labels']
    else:
        # Add extra views
        image_features_path = get_image_features_path(
            args.dataset,
            args.feature_dir,
            args.clip_encoder,
            args.image_layer_idx,
            args.image_augmentation,
            image_views=args.image_views,
        )
        image_features = torch.load(image_features_path)
        train_features = torch.cat([ccrop_features['train']['features'], image_features['train']['features']], dim=0)
        train_labels = torch.cat([ccrop_features['train']['labels'], image_features['train']['labels']], dim=0)
    
    image_train_dataset = TensorDataset(
        train_features,
        train_labels
    )
    image_val_dataset = TensorDataset(
        ccrop_features['val']['features'],
        ccrop_features['val']['labels']
    )

    test_features_path = get_test_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    test_features = torch.load(test_features_path)
    test_dataset = TensorDataset(
        test_features['features'],
        test_features['labels']
    )
    
    save_dir = get_save_dir(args)

    hyperparams = HYPER_DICT[args.hyperparams]
    # filter out invalid batch sizes
    VALID_BATCH_SIZES = hyperparams['batch_size']

    def get_experiment_count(hyperparams):
        count = 1
        count *= len(hyperparams['lr'])
        count *= len(hyperparams['weight_decay'])
        count *= len(VALID_BATCH_SIZES)
        count *= len(hyperparams['max_iter'])
        return count
    experiment_count = get_experiment_count(hyperparams)
    cur_count = 0

    # sweep through hyperparameters
    for lr in hyperparams['lr']:
        for wd in hyperparams['weight_decay']:
            for batch_size in VALID_BATCH_SIZES:
                for iters in hyperparams['max_iter']:
                    cur_count += 1

                    hyperparams_str = get_hyperparams_str(
                        hyperparams['optim'], lr, wd, batch_size, iters)

                    # check if experiment has been done
                    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                    makedirs(checkpoint_dir)
                    test_result_dict = {}
                    test_result_path = os.path.join(checkpoint_dir, "test_result.pth")
                    if os.path.exists(test_result_path):
                        print(f"Already exists: {hyperparams_str} {cur_count}/{experiment_count}")
                        test_result_dict = torch.load(test_result_path)
                        continue
                    else:
                        print(f"Starting: {hyperparams_str} {cur_count}/{experiment_count}")

                    # train logreg

                    # Create the logreg model
                    image_encoder = torch.load(
                        image_encoder_path).partial_model.train().cuda()

                    head, num_classes, in_features = make_classifier_head(
                        args.classifier_head,
                        args.clip_encoder,
                        args.classifier_init
                    )
                    logit_head = LogitHead(
                        head,
                        logit_scale=args.logit,
                    ).train().cuda()

                    total_params_image = sum(p.numel() for p in image_encoder.parameters()) / 1e6
                    trainable_params_image = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad) / 1e6
                    total_params_head = sum(p.numel() for p in logit_head.parameters()) / 1e6
                    trainable_params_head = sum(p.numel() for p in logit_head.parameters() if p.requires_grad) / 1e6
                    print(f"Total parameters: {total_params_image+total_params_head:.2f}M")
                    print(f"Trainable parameters: {trainable_params_image+trainable_params_head:.2f}M")

                    scale_head = LogitScale().train().cuda()
                    # Create the optimizer
                    params_groups = [
                        {'params': logit_head.parameters()},
                        {'params': image_encoder.parameters()},
                        {'params': scale_head.parameters()},
                    ]
                    optimizer = build_optimizer(params_groups, hyperparams['optim'], lr, wd)
                    scheduler = build_lr_scheduler(
                        optimizer,
                        hyperparams['lr_scheduler'],
                        hyperparams['warmup_iter'],
                        iters,
                        warmup_type=hyperparams['warmup_type'],
                        warmup_lr=hyperparams['warmup_min_lr']
                    )
                    # criterion = torch.nn.CrossEntropyLoss()
                    criterion = FocalLoss(alpha=0.25, gamma=4)
                    image_batch_size = batch_size


                    image_loader = None
                    if image_batch_size > 0:
                        image_loader = DataLoader(
                            image_train_dataset,
                            batch_size=image_batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                        )

                    val_loader = DataLoader(
                        image_val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )

                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=args.test_batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )

                    result_dict = train(
                        logit_head, image_encoder,
                        image_loader, val_loader,
                        optimizer, scheduler, criterion, iters,
                        eval_freq=EVAL_FREQ
                    )
                    start_time = time.time()
                    print_result_dict = {
                    'iter': [],
                    'test_accs': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                }
                    test_result_dict = {}
                    test_result_dict['val_acc'] = result_dict['val_acc']
                    test_result_dict['iter'] = result_dict['iter']
                    test_result_dict['test_accs'] = {}
                    test_result_dict['precision'] = {}
                    test_result_dict['recall'] = {}
                    test_result_dict['f1'] = {}
                    # Create the logreg model and load the weights
                    head, num_classes, in_features = make_classifier_head(
                        args.classifier_head,
                        args.clip_encoder,
                        args.classifier_init,
                        bias=False
                    )
                    old_logit_head = LogitHead(
                        head,
                        logit_scale=args.logit,
                    )
                    old_scale_head = LogitScale().cuda()
                    old_logit_head.load_state_dict(result_dict['logit_head'])

                    image_encoder = torch.load(image_encoder_path).partial_model
                    image_encoder.load_state_dict(result_dict['image_encoder'])
                    image_encoder = image_encoder.cuda().eval()

                    eval_heads = get_eval_heads(
                        deepcopy(old_logit_head.head),
                        logit=args.logit
                    )

                    for eval_type in eval_heads:
                        eval_head = eval_heads[eval_type]
                        eval_head.cuda().eval()

                        test_acc, precision, recall, f1 = test(eval_head, image_encoder, test_loader, device="cuda")
                        test_result_dict['iter'] = iters
                        test_result_dict['test_accs'] = test_acc
                        test_result_dict['precision'] = precision
                        test_result_dict['recall'] = recall
                        test_result_dict['f1'] = f1

                        print_result_dict['iter'].append(cur_count)
                        print_result_dict['test_accs'].append(test_acc)
                        print_result_dict['precision'].append(precision)
                        print_result_dict['recall'].append(recall)
                        print_result_dict['f1'].append(f1)
                        eval_head.cpu()
                    torch.save(test_result_dict, test_result_path)
                    print(test_result_dict)
                    print(f"Finished testing {hyperparams_str} {cur_count}/{experiment_count}")
                    elapsed_time = time.time() - start_time

                    # Print or log the time taken for evaluation
                    print(f"Time taken for evaluation: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)