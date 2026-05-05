"""
Compute FID (Fréchet Inception Distance) for diffusion models.
This script analyzes generated samples to compute FID scores against reference statistics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import warnings
import torchvision
import subprocess
import shutil

# Add Utils to path
sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Evaluation
import Diffusion as dm
import cfg

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute FID (Fréchet Inception Distance) for diffusion models."
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
    parser.add_argument("-istat", "--id_stat", help="Index of the reference statistics (1 to 5)", type=int, required=True)
    
    # Analysis parameters
    parser.add_argument("--N1", help="Starting batch index", type=int, default=0)
    parser.add_argument("--N2", help="Ending batch index", type=int, default=100)
    parser.add_argument("--batch_size_samples", help="Size of each sample batch", type=int, default=100)
    parser.add_argument("--device", help="Device to use (cuda:0, cpu)", type=str, default='cuda:0')
    parser.add_argument("--available_only", action='store_true',
                        help='Only use checkpoints that exist in the Samples/ folder.')
    parser.add_argument("--image_pth", type=str, default=None,
                        help='Path to .pth file with training images (overrides config.path_data).')
    
    return parser.parse_args()


def detransform_images(images, config):
    """Detransform images from normalized to original scale."""
    t = images.clone()
    mean = torch.tensor(config.mean, dtype=images.dtype, 
                        device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(config.std[0], dtype=images.dtype,
                       device=images.device).view(1, -1, 1, 1)
    return t * std + mean


def compute_fid_for_checkpoint(tau, type_model, config, path_stats_testset, 
                             N1, N2, batch_size_samples, file_FID):
    """Compute FID for a specific training checkpoint."""
    # Save directory for temporary images
    file_img_gen = config.path_save + type_model + 'FID/{:d}/'.format(tau)
    os.makedirs(file_img_gen, exist_ok=True)
    
    try:
        # Load generated images for the current training time
        for i in range(N1, N2):
            path_save = config.path_save + type_model + 'Samples/' + '{:d}/'.format(tau)
            path = path_save + 'generated'
            file_a = path + '/samples_a_{:d}'.format(i)
            
            # Load generated samples
            images_a = torch.load(file_a, map_location='cpu')
            
            # Detransform data to original scale
            t = detransform_images(images_a, config)
            
            # Save images as PNG files
            for (index_im, x) in enumerate(t):
                torchvision.utils.save_image(x, file_img_gen + '{:d}.png'.format(index_im + i*batch_size_samples))
        
        # Compute FID using pytorch_fid
        fid_device = config.DEVICE if config.DEVICE.startswith('cuda') and torch.cuda.is_available() else 'cpu'
        args = '{:s} {:s} --device {:s}'.format(path_stats_testset, file_img_gen, fid_device)
        cmd = 'python -m pytorch_fid {:s}'.format(args)
        p = subprocess.check_output(cmd, shell=True, text=True)
        fid = float(p.split(' ')[2][0:-2])
        
        # Save result
        with open(file_FID, "a") as myfile:
            myfile.write("\n{:d}\t{:.3f}".format(tau, fid))
        
    except Exception as e:
        print(f"Error computing FID for checkpoint {tau}: {e}")
        fid = -1.000
        with open(file_FID, "a") as myfile:
            myfile.write("\n{:d}\t{:.3f}".format(tau, fid))
        print('Skipping...')
    
    finally:
        # Clean up temporary images directory
        if os.path.exists(file_img_gen):
            shutil.rmtree(file_img_gen)
    
    return fid


def ensure_ref_stats(path_stats, config, image_pth=None):
    """Create reference FID statistics from training images if they don't exist."""
    if os.path.exists(path_stats):
        return
    print(f"Reference stats not found at {path_stats}. Creating from training images...")
    ref_img_dir = os.path.join(os.path.dirname(path_stats), 'ref_images_tmp')
    os.makedirs(ref_img_dir, exist_ok=True)
    try:
        pth = image_pth if image_pth else config.path_data
        raw = torch.load(pth, map_location='cpu')
        if isinstance(raw, dict):
            images = torch.stack(list(raw.values()))
        else:
            images = raw
        images = images.float()
        # Normalize to [0, 1] for PNG saving
        images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        images = images.clamp(0, 1)
        for idx, img in enumerate(tqdm(images, desc='Saving ref images')):
            torchvision.utils.save_image(img, os.path.join(ref_img_dir, f'{idx}.png'))
        fid_device = config.DEVICE if config.DEVICE.startswith('cuda') and torch.cuda.is_available() else 'cpu'
        cmd = f'python -m pytorch_fid --save-stats {ref_img_dir} {path_stats} --device {fid_device}'
        subprocess.check_call(cmd, shell=True)
        print(f"Saved reference stats to {path_stats}")
    finally:
        shutil.rmtree(ref_img_dir, ignore_errors=True)


def compute_fid_all_checkpoints(training_times, type_model, config, args):
    """Compute FID for all training checkpoints."""
    # Setup paths and files
    path_stats_testset = config.path_save + 'FID_ref/stats{:d}.npz'.format(args.id_stat)
    path_file = config.path_save + type_model + 'FID/'
    file_FID = path_file + 'FID_{:d}.txt'.format(args.id_stat)
    if os.path.exists(file_FID):     # Remove existing file
        os.remove(file_FID)
    os.makedirs(path_file, exist_ok=True)
    os.makedirs(os.path.dirname(path_stats_testset), exist_ok=True)
    ensure_ref_stats(path_stats_testset, config, image_pth=getattr(args, 'image_pth', None))

    print(f"Computing FID for {len(training_times)} checkpoints...")
    print(f"Model: {type_model}")
    print(f"Reference statistics: {path_stats_testset}")
    print(f"Output file: {file_FID}")
    
    pbar = tqdm(training_times)
    for tau in pbar:
        fid = compute_fid_for_checkpoint(
            tau=tau,
            type_model=type_model,
            config=config,
            path_stats_testset=path_stats_testset,
            N1=args.N1,
            N2=args.N2,
            batch_size_samples=args.batch_size_samples,
            file_FID=file_FID
        )
        pbar.set_description(f'FID = {fid:.3f}')


def main():
    """Main function to compute FID scores."""
    # Parse arguments
    args = parse_arguments()
    print("Arguments:", args)
    
    # Load configuration
    config = cfg.load_config(args.dataset)
    # Preserve dataset channel count; only override spatial dims
    config.IMG_SHAPE = (config.IMG_SHAPE[0], args.img_size, args.img_size)
    config.n_images = args.num
    config.BATCH_SIZE = min(args.batch_size, config.n_images)
    config.OPTIM = args.optim
    config.LR = args.learning_rate
    config.DEVICE = args.device
    
    # Model type string for paths
    type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(
        config.DATASET, args.img_size, config.n_images, args.nbase, 
        config.OPTIM, config.BATCH_SIZE, config.LR, args.index
    )
    
    # Define training times to analyze
    if args.available_only:
        samples_dir = os.path.join(config.path_save,
            '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}'.format(
                config.DATASET, args.img_size, config.n_images, args.nbase,
                config.OPTIM, config.BATCH_SIZE, config.LR, args.index),
            'Samples')
        training_times = sorted([
            int(d) for d in os.listdir(samples_dir)
            if os.path.isdir(os.path.join(samples_dir, d)) and d.isdigit()
        ])
    else:
        training_times = cfg.get_training_times()

    # Set mean/std on config so detransform_images works (needed if loader not called)
    if args.image_pth:
        config.path_data = args.image_pth
    if not hasattr(config, 'mean'):
        config.mean = torch.zeros(config.IMG_SHAPE[0])
    if not hasattr(config, 'std'):
        config.std = torch.ones(config.IMG_SHAPE[0])
    
    # Compute FID for all checkpoints
    compute_fid_all_checkpoints(
        training_times=training_times,
        type_model=type_model,
        config=config,
        args=args
    )
    
    print("FID computation completed!")


if __name__ == "__main__":
    main()