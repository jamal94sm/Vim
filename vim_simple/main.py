# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

from contextlib import suppress

import models_mamba

import utils

# log about
import mlflow

from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def main():
    # fix the seed for reproducibility
    seed = 0
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    
    cudnn.benchmark = True
    
    # log about
    output_dir = 'Vim/vim_simple/output_dir/'
    local_rank = 0 #?
    run_name = output_dir.split("/")[-1]
    
    batch_size = 64
    num_workers = 10 #?

###################################################################### DATA
    ### Dataset  
    def build_transform(input_size, eval_crop_ratio):
        resize_im = input_size > 32
        t = []
        if resize_im:
            size = int(input_size / eval_crop_ratio)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))
    
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
    
    
    eval_crop_ratio = 0.875
    # MNIST
    #dataset = datasets.MNIST('Vim/data_set/', train=is_train, transform=transform, download=True)

    # CIFAR
    nb_classes = 10
    input_size = 32
    transform = build_transform (input_size, eval_crop_ratio)
    dataset_train = datasets.CIFAR10('Vim/data_set/', train=True, transform=transform, download=True)
    dataset_val = datasets.CIFAR10('Vim/data_set/', train=False, transform=transform, download=True)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * batch_size * 20),
        num_workers=num_workers,
        drop_last=False
    )


    ########################################################## MODEL
    model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    print(f"Creating model: {model}")
    model = create_model(
        model,
        pretrained=False,
        num_classes=nb_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=input_size
    )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    args = SimpleNamespace()
    args.sched = 'cosine'
    args.clip_grad = None
    args.weight_decay = 0
    args.lr = 1e-4
    args.opt = 'adam' 
    args.momentum = 0.9

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(output_dir)
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    epochs = 20
    amp_autocast = suppress
    loss_scaler = "none"
    for epoch in range(0, epochs):

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            args.clip_grad,
            args=args
        )

        lr_scheduler.step(epoch)
        if output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                    'args': args,
                }, checkpoint_path)
             

        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        if output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    main()
