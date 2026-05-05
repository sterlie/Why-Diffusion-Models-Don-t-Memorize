"""
Compute fraction collapsed (memorization metric) for diffusion models.
This script analyzes generated samples to compute the fraction of samples that collapse
to training data using gap ratio analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import warnings

# Add Utils to path
sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Evaluation
import Diffusion as dm
import cfg
from loader_milk import MILK10Dataset

warnings.filterwarnings("ignore")


def bootstrap_mean_se(data, threshold, n_bootstrap=1000, random_state=None):
    """
    Compute bootstrap estimate of the mean and its standard error for values below a threshold.

    Parameters:
    - data: 1D array-like of values.
    - threshold: numeric threshold; only values < threshold are considered.
    - n_bootstrap: number of bootstrap samples.
    - random_state: seed for reproducibility.

    Returns:
    - mean_est: bootstrap estimate of the mean.
    - se_est: bootstrap estimate of the standard error of the mean.
    - lower: lower bound of 95% confidence interval.
    - upper: upper bound of 95% confidence interval.
    """
    # Prepare RNG
    rng = np.random.default_rng(random_state)
    
    # Generate bootstrap samples
    means = np.empty(n_bootstrap)
    n_data = len(data)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n_data, replace=True)
        collapsed = np.where(sample < threshold)[0]
        means[i] = len(collapsed) / len(sample)
    
    # Compute estimates
    mean_est = means.mean()
    se_est = means.std(ddof=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return mean_est, se_est, lower, upper


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute fraction collapsed (memorization metric) for diffusion models."
    )
    
    # Model configuration arguments
    parser.add_argument("-n", "--num", help="Number of training data", type=int, required=True)
    parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int, required=True)
    parser.add_argument("-s", "--img_size", help="Size of the images to use", type=int, required=True)
    parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float, required=True)
    parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str, required=True)
    parser.add_argument("-W", "--nbase", help="Number of base filters", type=int, required=True)
    parser.add_argument("-B", "--batch_size", help="Batch size used to train the model", type=int, required=True)
    parser.add_argument("-D", "--dataset", help="Dataset used to train the model", type=str, required=True)
    
    # Analysis parameters
    parser.add_argument("-Ns", "--Nsamples", help="Number of sample batches to analyze", type=int, default=100)
    parser.add_argument("--batch_sample_size", help="Size of each sample batch", type=int, default=100)
    parser.add_argument("--gap_threshold", help="Gap ratio threshold for collapsed samples", type=float, default=1/3)
    parser.add_argument("--device", help="Device to use (cuda:0, cpu)", type=str, default='cuda:0')
    parser.add_argument("--num_classes", help="Number of classes (for class-conditional models)", type=int, default=None)
    parser.add_argument("--image_pth", help="Path to .pth image file (MILK10). If set, overrides cfg path_data.", type=str, default=None)
    parser.add_argument("--metadata_csv", help="Path to metadata CSV (MILK10).", type=str, default=None)
    
    return parser.parse_args()

def compute_fraction_mem(training_times, train_images, type_model, config, file_fc,
                             nsamples, sample_size, gap_threshold, num_classes=None):
    """Compute fraction collapsed for all training times."""
    N = np.prod(config.IMG_SHAPE)
    X = train_images.reshape(-1, N).float()

    classes = list(range(num_classes)) if num_classes is not None else [None]

    # Open a per-class file for each class
    class_files = {}
    if num_classes is not None:
        for c in classes:
            path_class = os.path.dirname(file_fc) + f'/fraction_memorized_class{c}.txt'
            if os.path.exists(path_class):
                os.remove(path_class)
            class_files[c] = path_class

    pbar = tqdm(training_times)
    for tau in pbar:
        k = min(2, len(train_images))

        # Per-class distances
        class_distances = {c: [] for c in classes}

        for c in classes:
            for i in range(nsamples):
                base = config.path_save + type_model + 'Samples/' + '{:d}/'.format(tau)
                if c is not None:
                    path = base + 'class_{:d}/generated'.format(c)
                else:
                    path = base + 'generated'
                file_a = path + '/samples_a_{:d}'.format(i)

                try:
                    images_a = torch.load(file_a, map_location='cpu')
                except FileNotFoundError:
                    print(f"Warning: File not found: {file_a}")
                    continue

                s = images_a.reshape(-1, 1, N).to(config.DEVICE)
                dist = torch.norm(s - X, dim=2, p=2)
                knn = dist.topk(k, dim=1, largest=False)
                class_distances[c].append(knn[0].cpu())

        desc_parts = []
        for c in classes:
            if not class_distances[c]:
                continue
            distances_tensor = torch.cat(class_distances[c], dim=0)
            gap_ratio = distances_tensor[:, 0] / distances_tensor[:, 1]

            collapsed_samples = np.where(gap_ratio.numpy() < gap_threshold)[0]
            fraction_mem = len(collapsed_samples) / len(gap_ratio)

            if len(collapsed_samples) > 0:
                fraction_mem, std_frac, lower, upper = bootstrap_mean_se(
                    gap_ratio.numpy(), gap_threshold
                )
            else:
                std_frac = 0.0
                lower = 0.0
                upper = 0.0

            label = f'c{c}' if c is not None else 'all'
            desc_parts.append(f'Fmem[{label}]={fraction_mem*100:.1f}%')

            out_file = class_files[c] if c in class_files else file_fc
            with open(out_file, "a") as myfile:
                myfile.write(f"\n{tau:d}\t{fraction_mem*100:.3f}\t{std_frac*100:.5f}\t"
                             f"{lower*100:.5f}\t{upper*100:.5f}")

        if desc_parts:
            pbar.set_description(' | '.join(desc_parts))


def main():
    """Main function to compute fraction collapsed."""
    # Parse arguments
    args = parse_arguments()
    print("Arguments:", args)
    
    # Load configuration
    config = cfg.load_config(args.dataset)
    config.IMG_SHAPE = (config.IMG_SHAPE[0], args.img_size, args.img_size)  # preserve channel count from dataset
    config.n_images = args.num
    config.BATCH_SIZE = min(args.batch_size, config.n_images)
    config.OPTIM = args.optim
    config.LR = args.learning_rate
    _requested_device = args.device
    if _requested_device.startswith('cuda') and not torch.cuda.is_available():
        _fallback = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'Warning: {_requested_device} not available, falling back to {_fallback}.')
        _requested_device = _fallback
    config.DEVICE = _requested_device
    
    # Model type string for paths
    type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(
        config.DATASET, args.img_size, config.n_images, args.nbase, 
        config.OPTIM, config.BATCH_SIZE, config.LR, args.index
    )
    
    # Create output directory and file
    path_file = config.path_save + type_model + 'Memorization/'
    file_fc = path_file + 'fraction_memorized.txt'
    if os.path.exists(file_fc):     # Remove existing file
        os.remove(file_fc)
    os.makedirs(path_file, exist_ok=True)
    
    # Define training times to analyze
    training_times = cfg.get_training_times()
    
    print(f"Computing memorization fraction for {len(training_times)} checkpoints...")
    print(f"Model: {type_model}")
    print(f"Output file: {file_fc}")
    
    # Load training data
    if args.dataset == 'MILK10' and args.image_pth is not None:
        # Load directly from .pth file (no centering, matches training preprocessing)
        raw = torch.load(args.image_pth, map_location='cpu')
        if isinstance(raw, dict):
            raw = torch.stack(list(raw.values()))  # {isic_id -> tensor} -> (N, C, H, W)
        raw = raw.float()
        if raw.shape[-1] != args.img_size:
            import torch.nn.functional as F
            raw = F.interpolate(raw, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
        # Use saved training indices if available (exact match to training set)
        indices_path = config.path_save + type_model + 'training_indices.npy'
        if os.path.exists(indices_path):
            indices = np.load(indices_path)
            train_images = raw[indices].to(config.DEVICE)
            print(f"Loaded {len(train_images)} training images using saved indices from {indices_path}")
        else:
            train_images = raw[:config.n_images].to(config.DEVICE)
            print(f"WARNING: No saved indices found at {indices_path}. Using first {len(train_images)} images — may not match training set.")
    else:
        config.CENTER = False
        train_images, _ = cfg.load_training_data(config, args.index)
        train_images = train_images[:config.n_images, :, :, :].to(config.DEVICE)

    # Setup diffusion configuration
    df = dm.DiffusionConfig(
        n_steps=config.TIMESTEPS,
        img_shape=config.IMG_SHAPE,
        device=config.DEVICE,
    )
    
    # Compute fraction collapsed for each checkpoint
    compute_fraction_mem(
        training_times=training_times,
        train_images=train_images,
        type_model=type_model,
        config=config,
        file_fc=file_fc,
        nsamples=args.Nsamples,
        sample_size=args.batch_sample_size,
        gap_threshold=args.gap_threshold,
        num_classes=args.num_classes
    )
    
    print("Memorization fraction computation completed!")


if __name__ == "__main__":
    main()