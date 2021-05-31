#!/usr/local/bin/python
# coding: utf-8
'''
@University  : FEI
@Local       : São Bernardo do Campo, São Paulo, Brazil
@Laboratory  : Image Processing Lab (IPL)
@file        : train.py
@author      : Lucas Fontes Buzuti
@version     : 1.0.0
@created     : 06/01/2020
@modified    : 10/21/2020
@e-mail      : lucas.buzuti@outlook.com
'''

import os
import cv2
import random
import argparse
import numpy as np
from glob import glob
from PIL import Image
from kornia.color import bgr_to_rgb

from torchsummary import summary
from torch_fidelity import calculate_metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from util.config import cfg, cfg_from_file
from util.utils import walk_latent_space, save_model
from util.database import Dataset, DatasetTensor
from util.models.GenerativeAdversarialNets import Generator, Generator_v2, Discriminator, Discriminator_v2, weights_init



# Set random seed for reproducibility
MANUAL_SEED = 42
#MANUAL_SEED = random.randint(1, 10000) # use if you want new results
print(f'Random Seed: {MANUAL_SEED}')
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

# Flags
FOLDER = None


def parse_args():
    parse = argparse.ArgumentParser(description="Train a Paintron Neural Network")
    parse.add_argument(
        '--cfg', dest='cfg_file', 
        help='Path of configure file of the training model',
        default='', type=str
    )
    parse.add_argument(
        '--load', action='store_true', help='Load model'
    )
    parse.add_argument(
        '--freeze', action='store_true', 
        help='Freezing last layer of generator and discriminator.'
    )
    parse.add_argument(
        '--ckpt', dest='ckpt', 
        help='Path where the checkpoint are saved',
        default='', type=str
    )
    parse.add_argument(
        '--resolution', dest='resolution', 
        help='Resolution of model',
        default=128, type=int
    )
    parse.add_argument(
        '--bicubic', action='store_true', help='Enable bicubic into dataset.'
    )
    parse.add_argument(
        '--retinaface', action='store_true', help='Enable bicubic into dataset.'
    )
    args = parse.parse_args()
    return args


def GAN(device, args):
    """
    Generative Adversarial Networks
    Param:
        device (type -> torch.device): defines the type of device 
        the model will be process.
    
    Return: Model generator and discriminator
    """
    # Create the generator and discriminator
    if args.resolution == 128:
        netG = Generator()
        netD = Discriminator()
        print("\n\nGenerator Parameters")
        summary(netG, (cfg.MODEL.Z, 1, 1))
        print("\nDiscriminator Parameters\n\n")
        summary(netD, (3, 128, 128))
    elif args.resolution == 512:
        netG = Generator_v2()
        netD = Discriminator_v2()
        print("\n\nGenerator Parameters")
        summary(netG, (cfg.MODEL.Z, 1, 1))
        print("\nDiscriminator Parameters\n\n")
        summary(netD, (3, 512, 512))
    else:
        raise SystemError("ERROR: you need to pick `--resolution` between 128 or 512!")

    # Load any state dictionary to initialize weights
    if args.load:
        print("Loading model...")
        #if isinstance(netG, nn.DataParallel):
        #    checkpoint = torch.load(args.ckpt)
        #    netG.module.load_state_dict(checkpoint['netG'])
        #    netD.module.load_state_dict(checkpoint['netD'])
        #else:
        checkpoint = torch.load(args.ckpt)
        netG.load_state_dict(checkpoint['netG'])
        netD.load_state_dict(checkpoint['netD'])

    # Freezing the parameters so that the gradients 
    # are not computed in backward()
    #if args.freeze:
    #    print("Freezing weights...")
    #    for name, param in netG.named_parameters():
    #        if not name in ['convT6.weight']:
    #            param.requires_grad = False
    #    for name, param in netD.named_parameters():
    #        if not name in ['conv6.weight']:
    #            param.requires_grad = False

    netG.to(device)
    netD.to(device)
    
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (cfg.TRAIN.NUMBER_GPU > 1):
        netG = nn.DataParallel(netG, list(range(cfg.TRAIN.NUMBER_GPU)))
        netD = nn.DataParallel(netD, list(range(cfg.TRAIN.NUMBER_GPU)))

    if not args.load:
        print("Initializing weights randomly...")
        # Apply the weights_init function to randomly initialize 
        # all weights to mean=0, stdev=0.2
        netG.apply(weights_init)
        netD.apply(weights_init)

    return netD, netG


def loss():
    """
    Initialize BCELoss function (Binary Cross Entropy)

    Return: The loss.
    """
    return nn.BCELoss()


def opt(netD, netG):
    """
    Setup Adam optimizers for both G and D
    Param:
        netD (type -> util.models.GenerativeAdversarialNets.Discriminator):
            Model discriminator.
        netG (type -> util.models.GenerativeAdversarialNets.Generator):
            Model generator.

    Return: The generator and discriminator optimizer.
    """
    optD = optim.Adam(
        netD.parameters(), 
        lr=cfg.TRAIN.LEARNING_RATE, 
        betas=(cfg.TRAIN.BETA1, 0.999)
    )
    optG = optim.Adam(
        netG.parameters(), 
        lr=cfg.TRAIN.LEARNING_RATE, 
        betas=(cfg.TRAIN.BETA1, 0.999)
    )
    return optD, optG


def training_loop(device, dataloader, netD, netG, criterion, optD, optG):
    """
    Training Loop
    Param:
        device (type -> torch.device): defines the type of device 
            the model will be process.
        dataloader (type -> torch.utils.data.dataloader.DataLoader):
            Dataset.
        netD (type -> util.models.GenerativeAdversarialNets.Discriminator):
            Model discriminator.
        netG (type -> util.models.GenerativeAdversarialNets.Generator):
            Model generator.
        criterion (type -> torch.nn.modules.loss.BCELoss): BCELoss function 
            (Binary Cross Entropy).
        optD (type -> torch.optim): the generator optimizer.
        optG (type -> torch.optim): the discriminator optimizer.
    """
    global FOLDER

    # Dir where summary will be saved
    if not os.listdir(f'runs/{args.resolution}/'):
        writer = SummaryWriter(f'runs/{args.resolution}/1/training')
        FOLDER = 1
    else:
        folders = os.listdir(f'runs/{args.resolution}/')
        nfolders = sorted(list(map(int, folders)))
        FOLDER = nfolders[-1]+1
        writer = SummaryWriter(f'runs/{args.resolution}/{FOLDER}/training')

    # Flags to keep track of progress
    epoch = 1

    metric_fid = np.inf
    metric_fid_best = np.inf

    # Create batch of latent vectors that we will use to 
    # visualize the progression of the generator
    fixed_noise = torch.randn(cfg.TRAIN.BATCH_SIZE, cfg.MODEL.Z, 1, 1, device=device)

    print("Starting Training Loop...")
    try:
        # For each epoch
        while True:
            errD_epoch, errG_epoch = [], []
            D_x_epoch = []
            D_G_z1_epoch, D_G_z2_epoch = [], []
            original_images = None
            # For each batch in the dataloader
            if not np.isinf(metric_fid):
                print(f'\n[METRIC (FID)] Best: {metric_fid_best:.4f} / Current: {metric_fid:.4f}')
            for i, local_batch in enumerate(dataloader, 0):
                if i == 0:
                    original_images = local_batch
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = local_batch.to(device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,), cfg.TRAIN.REAL_LABEL*cfg.TRAIN.SMOOTHING, 
                    dtype=torch.float, device=device
                )
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, cfg.MODEL.Z, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(cfg.TRAIN.FAKE_LABEL)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate the D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optD.step()


                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(cfg.TRAIN.REAL_LABEL) # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch 
                # through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optG.step()
                

                # Output training stats
                print(f'[BATCH] {i+1:04d}/{len(dataloader):04d}: \tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

                errD_epoch.append(errD.item())
                errG_epoch.append(errG.item())
                D_x_epoch.append(D_x)
                D_G_z1_epoch.append(D_G_z1)
                D_G_z2_epoch.append(D_G_z2)


            # Output training stats
            print(f'\n\n[EPOCH] {epoch}/{cfg.TRAIN.MAX_EPOCH}')
            print(f'[AVERAGE] \tLoss_D: {np.mean(errD_epoch):.4f}\tLoss_G: {np.mean(errG_epoch):.4f}\tD(x): {np.mean(D_x_epoch):.4f}\tD(G(z)): {np.mean(D_G_z1_epoch):.4f} / {np.mean(D_G_z2_epoch):.4f}')

            # Summary
            writer.add_scalars('loss/mean',
                {'discriminator': np.mean(errD_epoch),
                'generator': np.mean(errG_epoch)},
                epoch
            )
            writer.add_scalars('loss/total',
                {'discriminator': np.sum(errD_epoch),
                'generator': np.sum(errG_epoch)},
                epoch
            )
            writer.flush()

 
            # output on fixed_noise
            with torch.no_grad():
                # Generation images
                fake = netG(fixed_noise)
            fake = ((fake - fake.min())) / (fake.max() - fake.min()) # convert image to [0, 1]
            original_images = ((original_images - original_images.min())) / (original_images.max() - original_images.min()) # convert image to [0, 1]
            
            if (epoch % cfg.TRAIN.FREQUENCY_TENSORB == 0) or (epoch == cfg.TRAIN.MAX_EPOCH-1):
                # Walking latent space z_dim
                walk_images = walk_latent_space(device, netG, cfg.MODEL.Z, -4, 4, 32)
                walk_images = ((walk_images - walk_images.min())) / (walk_images.max() - walk_images.min()) # convert image to [0, 1]

                writer.add_images('original', original_images, epoch)
                writer.add_images('sampling', fake, epoch)
                writer.add_images('walking_latent_space', walk_images, epoch)
                writer.flush()
            
            for i, (fk, orig) in enumerate(zip(fake, original_images)):
                # Fake images
                fk = bgr_to_rgb(fk)
                fk *= 255
                cv2.imwrite(f'images/{args.resolution}/fakes/id_{i:04d}.png', fk.permute(1,2,0).cpu().numpy())

                # Original images
                orig = bgr_to_rgb(orig)
                orig *= 255
                cv2.imwrite(f'images/{args.resolution}/originals/id_{i:04d}.png', orig.permute(1,2,0).cpu().numpy())

            try:
                metrics = calculate_metrics(f'images/{args.resolution}/originals', f'images/{args.resolution}/fakes', cuda=True, isc=True, fid=True, kid=False, verbose=False)
                metric_fid = metrics['frechet_inception_distance']
                metric_is_mean = metrics['inception_score_mean']
                metric_is_std = metrics['inception_score_std']
                print(f'[METRIC] FID {metric_fid:.4f} | IS {metric_is_mean:.4f} ± {metric_is_std:.4f}')

                # Summary
                writer.add_scalar('matrics/IS', metric_is_mean, epoch)
                writer.add_scalar('matrics/FID', metric_fid, epoch)
                #writer.add_scalar('matrics/KID', metrics['kernel_inception_distance_mean'], epoch)
                writer.flush()

                # Save the model checkpoint
                if metric_fid <= metric_fid_best:
                    save_model(netG, netD, epoch, f'{args.resolution}/DCGAN_best.pth')
                    print(f'[SAVE] The best model was saved with FID {metric_fid:.4f}')
                    # Summary
                    writer.add_text('metric', f'The best model with FID {metric_fid:.4f}', global_step=epoch)
                    writer.flush()
                    metric_fid_best = metric_fid
            
            except Exception as err:
                print(f'[METRIC ERROR] Complex number')
                # Summary
                writer.add_scalar('matrics/IS', np.nan, epoch)
                writer.add_scalar('matrics/FID', np.nan, epoch)
                #writer.add_scalar('matrics/KID', metrics['kernel_inception_distance_mean'], epoch)
                writer.flush()

            if epoch % 100 == 0:
                save_model(netG, netD, epoch, f'{args.resolution}/DCGAN_{epoch:05d}.pth')
                print(f'[SAVE] Epoch {epoch:05d} - Saved model!')

            # Stop training loop
            if epoch == cfg.TRAIN.MAX_EPOCH:
                save_model(netG, netD, epoch, f'{args.resolution}/DCGAN_{epoch:05d}.pth')
                print(f'[SAVE] Epoch {epoch:05d} - Saved last model!')
                print('[TRAIN] Stoping Training Loop...')
                break
            save_model(netG, netD, epoch, f'{args.resolution}/DCGAN_last.pth')
            print(f'[SAVE] Saved last model!')
            epoch += 1

    except KeyboardInterrupt:
        save_model(netG, netD, epoch, f'{args.resolution}/DCGAN_last.pth')
        print('[TRAIN] Stoping Training Loop...')

    writer.close()



if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file == '':
        raise SystemError(
            """ERROR: You need input a path of configure! 
            \npython train.py --cfg /Path/of/config/file.yaml"""
        )
    else:
        # Load file of config
        cfg_from_file(args.cfg_file)

    if not os.path.exists(f'runs/{args.resolution}'):
        os.mkdir(f'runs/{args.resolution}')
    if not os.path.exists(f'images/{args.resolution}'):
        os.mkdir(f'images/{args.resolution}')
        os.mkdir(f'images/{args.resolution}/fakes')
        os.mkdir(f'images/{args.resolution}/originals')
    if not os.path.exists(f'checkpoints/{args.resolution}'):
        os.mkdir(f'checkpoints/{args.resolution}')

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # find the best algorithm to use for your hardware
    torch.backends.cudnn.benchmark = True 

    # Model
    netD, netG = GAN(device, args)

    # Loss functions
    criterion = loss()

    # optimizers
    optimizerD, optimizerG = opt(netD, netG)


    # Create the dataset
    _types = ('*.png', '*.jpeg', '*.jpg', '*.bmp')
    path_list = [
        path for _type in _types 
        for path in glob(os.path.join(cfg.DATASET.DIR, _type))
    ]

    if args.bicubic:
        print("Enabled Bicubic.")
        transform=transforms.Compose([
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            # Normalize does the following for each channel:
            # image = (image - mean) / std 
            # The parameters mean, std are passed as 0.5, 0.5 in this case. 
            # This will normalize the image in the range [-1,1].
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            # Normalize does the following for each channel:
            # image = (image - mean) / std 
            # The parameters mean, std are passed as 0.5, 0.5 in this case. 
            # This will normalize the image in the range [-1,1].
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    params = {
        'batch_size': cfg.TRAIN.BATCH_SIZE,
        'shuffle': True,
        'num_workers': cfg.DATASET.WORKERS
    }
    data = Dataset(path_list, retinaface=args.retinaface, transform=transform)
    print('Dataset size: {}'.format(data.__len__()))
    training_gen = torch.utils.data.DataLoader(data, **params)


    params = {
        'device': device,
        'dataloader': training_gen, 
        'netD': netD, 
        'netG': netG, 
        'criterion': criterion, 
        'optD': optimizerD, 
        'optG': optimizerG
    }
    training_loop(**params)
