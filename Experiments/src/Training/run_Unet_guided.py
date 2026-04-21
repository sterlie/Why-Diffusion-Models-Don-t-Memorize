#%%
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os
import numpy as np
import argparse
import glob

sys.path.insert(1, '../Utils/')     # In case we run from Experiments/Training
import Unet
import Plot
import Diffusion
import loader
import cfg
from loader_milk import MILK10Dataset
from numpy.random import default_rng

#%% 


def pick_available_device(requested_device='cuda:0'):
    req = str(requested_device).lower()
    if req.startswith('cuda'):
        return req if torch.cuda.is_available() else 'cpu'
    if req == 'mps':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    return 'cpu' if req == 'cpu' else req

parser = argparse.ArgumentParser("Diffusion on MILK10 dataset with U-net and CFG.")
parser.add_argument("-n", "--num", help="Number of training data", type=int, default=128)
parser.add_argument("-i", "--index", help="Index for the dataset", type=int, default=0)
parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=16)
parser.add_argument("-l", "--label_col", help="Label column for conditioning", type=str, default="skin_tone_class")
parser.add_argument("-m", "--metadata_csv", help="Path to metadata CSV", type=str, default="../../Data/milk10/MILK10k_Training_Metadata.csv")
parser.add_argument("-p", "--image_pth", help="Path to .pth image file", type=str, default="../../Data/milk10/MILK10k_Training_Input.pth")
parser.add_argument("-s", "--img_size", help="Size of the images to use", type=int, default=32)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float, default=0.0001)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str, default="Adam")
parser.add_argument("-W", "--nbase", help="Number of base filters", type=str, default="128")
parser.add_argument("-g", "--generate", help="Whether to generate samples during training", action="store_true")
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int, default=-1)
parser.add_argument("--device", help="Device to use (cpu, mps, cuda:0)", type=str, default='cuda:0')
args = vars(parser.parse_args())
generate = args.get('generate', False)
print(args)

# Get arguments
n = args['num']
index = args['index']
size = args['img_size']
lr = args['learning_rate']
optim = args['optim']
n_base = int(args['nbase'])
time_step = args['time']
device = args['device']
if time_step == -1:
    mode = 'normal'
else:
    mode = 'fixed_time'

# Overwrite config with command line arguments
DATASET = 'MILK10'
config = cfg.load_config(DATASET)
config.IMG_SHAPE = (3, size, size)
config.n_images = n
config.BATCH_SIZE = min(512, n)
config.OPTIM = optim
config.LR = lr
config.mode = mode
config.time_step = time_step
config.DEVICE = pick_available_device(device)
print('Using device:', config.DEVICE)

if config.mode == 'normal':
    suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(config.DATASET, size,
                                        config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                        config.LR, index)
elif config.mode == 'fixed_time':
    suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_t{:d}/'.format(config.DATASET, size,
                                        config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                        config.LR, index, time_step)
    print('Training at fixed diffusion time: {:d}'.format(config.time_step))

# Create path to images and model save
path_images = config.path_save + suffix + 'Images/'
path_models = config.path_save + suffix + 'Models/'
os.makedirs(path_images, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

os.system('cp run_Unet_guided.py {:s}'.format(path_models + '_run_Unet_guided.py'))
os.system('cp ../Utils/loader.py {:s}'.format(path_models + '_loader.py'))
os.system('cp ../Utils/cfg.py {:s}'.format(path_models + '_cfg.py'))

# Raw images version
# loading_func = 'loader.load_{:s}(config, index={:d})'.format(config.DATASET, index)
# testset = None
# trainset, testset = eval(loading_func)

# # Test to put the full trainset on the device
# train_images = torch.zeros(size=(config.n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2]))
# for i in np.arange(config.n_images):
#     train_images[i, :, :] = trainset[i]
# train_images = train_images.to(config.DEVICE)


# MILK10 loader version (for classifier-free guidance)

metadata_csv = args['metadata_csv']
image_pth = args['image_pth']
label_col = args['label_col']
batch_size = args['batch_size']

dataset = MILK10Dataset(
    metadata_csv=metadata_csv,
    image_pth=image_pth,
    label_col=label_col,
    img_size=size
)
# Optionally subsample for small dataset
if n < len(dataset):
    indices = np.random.choice(len(dataset), n, replace=False)
    dataset = torch.utils.data.Subset(dataset, indices)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
testloader = None
print("Trainloader length:", len(trainloader))

# In[]
#testset = None
#if __name__ == '__main__':
#    trainloader = torch.utils.data.DataLoader(path_images, 
#                                              batch_size=config.BATCH_SIZE,
#                                              shuffle=True)
#    if testset is not None:
#        testloader = torch.utils.data.DataLoader(testset, 
#                                                  batch_size=config.BATCH_SIZE,
#                                                  shuffle=False)

# del trainset
# In[] Plot one random batch of training images


#dataiter = iter(trainloader)
#images, labels = next(dataiter)
#Plot.imshow(images[0:32].cpu(), config.mean, config.std)
# # Plot a batch of training images
#plt.savefig(path_images + 'Training_set.pdf', bbox_inches='tight')

# In[] Model definition


if __name__ == '__main__':
    print("Initialized model")
    model = Unet.UNet(
        input_channels          = config.IMG_SHAPE[0],
        output_channels         = config.IMG_SHAPE[0],
        base_channels           = n_base,
        base_channels_multiples = (1, 2, 4),
        apply_attention         = (False, True, True),
        dropout_rate            = 0.1,
        num_classes=dataset.dataset.num_classes if isinstance(dataset, torch.utils.data.Subset) else dataset.num_classes
    )
    # Resume training from last weights in the folder
    weights_files = glob.glob(os.path.join(path_models + 'Model_' + '*'))
    if weights_files:
        offset = max([int(f.split('_')[-1]) for f in weights_files])
    else:
        offset = 0
    if offset > 0:
        path_checkpoint = config.path_save + '/{:s}/Models/Model_{:d}'.format(suffix, offset)
        model = loader.load_model(model, path_checkpoint)
        model.to(config.DEVICE)
    if config.DEVICE.startswith('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.DEVICE)


if __name__ == '__main__':
    n_params = sum(p.numel() for p in model.parameters())
    print('{:.2f}M'.format(n_params/1e6))
    print("Starting training setup")

# In[] Training and saving


if __name__ == '__main__':
    if config.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    elif config.OPTIM == 'SGD_Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.95)

    df = Diffusion.DiffusionConfig(
        n_steps                 = config.TIMESTEPS,
        img_shape               = config.IMG_SHAPE,
        device                  = config.DEVICE,
    )
    loss_fn = nn.MSELoss()
    model.train()
    sweeping = 1.0
    times_save = cfg.get_training_times()

    # --- Classifier-Free Guidance label dropping ---
    cfg_drop_prob = 0.1  # Probability to drop label for unconditional training

    print("Starting custom training loop with CFG and model saving")
    n_steps = offset
    k_steps = 100
    bar = range(config.N_STEPS)
    from tqdm import tqdm
    bar = tqdm(bar, leave=True, position=0)
    bar.update(offset)
    while n_steps < config.N_STEPS:
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            # Randomly drop labels for CFG
            mask = (torch.rand(labels.shape[0], 1, device=labels.device) < cfg_drop_prob).float()
            labels = labels * (1 - mask)
            # Sample random timesteps
            t = torch.randint(1, df.n_steps, (images.size(0),), device=images.device)
            # Apply forward diffusion to get noisy image and target noise
            x_t, noise = Diffusion.forward_diffusion(df, images, t, config)
            x_t = x_t.to(config.DEVICE)
            # Forward pass with label conditioning
            output = model(x_t.float(), t, y=labels)
            loss = loss_fn(noise, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Model saving logic matching Diffusion.train
            shallSave = n_steps in times_save
            if n_steps >= config.N_STEPS:
                shallSave = 1
            if shallSave == 1:
                p = config.path_save + suffix + 'Models/' + 'Model_{:d}'.format(n_steps)
                torch.save(model.state_dict(), p)
                print(f"Model saved to {p}")
                if generate:
                    # Optionally sample and save images for visual check
                    pass

            n_steps += 1
            if n_steps % k_steps == 0:
                bar.set_description(f'loss: {loss:.5f}, n_steps: {n_steps:d}')
                bar.update(k_steps)
            if n_steps >= config.N_STEPS:
                break