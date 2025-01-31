"""
Dual-Teacher
Copyright (c) 2023-present NAVER Cloud Corp.
distributed under NVIDIA Source Code License for SegFormer
--------------------------------------------------------
References:
SegFormer: https://github.com/NVlabs/SegFormer
--------------------------------------------------------
"""

import os
import time
import argparse
from datetime import datetime
import cv2
from copy import deepcopy

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
from monai.losses import *
from monai.metrics import *

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Utils.DMT_loss import compute_cutmix, compute_classmix, compute_ic, ClassMixLoss

def main(config):
    """
    Main training function.
    """
    # Setup datasets and dataloaders
    dataset = get_dataset_without_full_label(
        config, 
        img_size=config.data.img_size,
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment4
    )
      
    l_train_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    u_train_loader = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )
    
    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Initialize models
    model = deeplabv3plus_resnet50(num_classes=3, output_stride=8, pretrained_backbone=True)
    model_teacher = deeplabv3plus_resnet50(num_classes=3, output_stride=8, pretrained_backbone=True)
    model_teacher2 = deeplabv3plus_resnet50(num_classes=3, output_stride=8, pretrained_backbone=True)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model = model.cuda()
    model_teacher = model_teacher.cuda()
    model_teacher2 = model_teacher2.cuda()

    # Freeze teacher models
    for p in model_teacher.parameters():
        p.requires_grad = False
    for p in model_teacher2.parameters():
        p.requires_grad = False

    # Setup loss functions
    criterion = GeneralizedDiceFocalLoss(
        include_background=True,
        to_onehot_y=False,
        softmax=True,
        reduction='mean'
    ).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    cm_loss_fn = ClassMixLoss(weight=None, reduction='none', ignore_index=255)

    # Train and test
    model = train_val(config, model, model_teacher, model_teacher2, train_loader, val_loader, criterion, criterion_u, cm_loss_fn)
    test(config, model, best_model_dir, test_loader, criterion)

def train_val(config, model, model_teacher, model_teacher2, train_loader, val_loader, criterion, criterion_u, cm_loss_fn):
    """
    Training and validation function with Dual Teacher strategy.
    """
    # Setup optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs)

    # Initialize MONAI metrics
    train_dice = DiceMetric(include_background=True, reduction="mean")
    
    max_dice = -float('inf')
    best_epoch = 0
    
    torch.save(model.state_dict(), best_model_dir)
    
    for epoch in range(config.train.num_epochs):
        start = time.time()
        
        # Training phase
        model.train()
        train_metrics = {'dice': 0, 'loss': 0}
        num_train = 0

        # Choose teacher model based on epoch
        if epoch % 2 == 0:
            ema_model = model_teacher
            do_cut_mix = True
            do_class_mix = False
        else:
            ema_model = model_teacher2
            do_cut_mix = False
            do_class_mix = True
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        
        train_dice.reset()
        
        for idx, (batch, batch_w_s) in enumerate(train_loop):
            # Get batch data
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            image_u = batch_w_s['img_w'].cuda().float()
            label_u = batch['label'].cuda().float()
            
            # Apply strong augmentation
            image_u_strong = deepcopy(image_u)
            image_u_strong = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_u_strong)
            image_u_strong = transforms.RandomGrayscale(p=0.2)(image_u_strong)

            # Compute losses based on mixing strategy
            if do_class_mix:
                loss = compute_classmix(
                    criterion=criterion,
                    cm_loss_fn=cm_loss_fn,
                    model=model,
                    ema_model=ema_model,
                    imgs=img,
                    labels=label,
                    unsup_imgs=image_u,
                    image_u_strong=image_u_strong,
                    threshold=0.95
                )
            if do_cut_mix:
                loss = compute_cutmix(
                    imgs=img,
                    labels=label,
                    criterion=criterion,
                    model=model,
                    ema_model=ema_model,
                    image_u=image_u,
                    threshold=0.95
                )

            # Compute consistency loss
            loss_dc = compute_ic(
                model=model,
                ema_model=ema_model,
                image_u=image_u,
                image_u_strong=image_u_strong,
                criterion_u=criterion_u,
                label_u=label_u,
                threshold=0.95
            )

            # Total loss
            total_loss = loss + loss_dc * 0.2

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update teacher model
            update_ema(model_teacher=ema_model, model=model, alpha_teacher=0.99, iteration=idx)

            # Calculate metrics
            with torch.no_grad():
                output = torch.softmax(model(img), dim=1)
                output_onehot = torch.zeros_like(output)
                output_onehot.scatter_(1, output.argmax(dim=1, keepdim=True), 1)
                
                train_dice(y_pred=output_onehot, y=label)
                
                train_metrics['loss'] = (train_metrics['loss'] * num_train + total_loss.item() * img.shape[0]) / (num_train + img.shape[0])
                num_train += img.shape[0]
            
            train_loop.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}"
            })
            
            if config.debug:
                break

        # Get final training metrics
        train_metrics['dice'] = train_dice.aggregate().item()
        
        # Validation phase
        val_metrics = validate_model(model, val_loader, criterion)
        current_dice = val_metrics['dice']
        
        # Save best model
        if current_dice > max_dice:
            max_dice = current_dice
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)
            
            message = f'New best epoch {epoch}! Dice: {current_dice:.4f}'
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch time
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        
        if config.debug:
            break
    
    print(f'Training completed. Best epoch: {best_epoch}')
    return model

def validate_model(model, val_loader, criterion):
    """
    Validate model using MONAI metrics.
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
        
        with torch.no_grad():
            output = torch.softmax(model(img), dim=1)
            loss = criterion(output, label)
            
            # Convert predictions to one-hot format
            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)
            
            # Convert labels to one-hot format if needed
            if len(label.shape) == 4:  # If already one-hot
                labels_onehot = label
            else:  # If not one-hot
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label.unsqueeze(1), 1)
            
            # Compute metrics
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)
            
            # Update loss
            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * batch_len) / (num_val + batch_len)
            num_val += batch_len
            
            val_loop.set_postfix({
                'Loss': f"{loss.item():.4f}"
            })
    
    # Aggregate metrics
    metrics['dice'] = dice_metric.aggregate().item()
    metrics['iou'] = iou_metric.aggregate().item()
    metrics['hd'] = hd_metric.aggregate().item()
    
    # Reset metrics
    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()
    
    return metrics

def test(config, model, model_dir, test_loader, criterion):
    """
    Test the model.
    """
    model.load_state_dict(torch.load(model_dir))
    metrics = validate_model(model, test_loader, criterion)
    
    results_str = (f"Test Results:\n"
                  f"Loss: {metrics['loss']:.4f}\n"
                  f"Dice: {metrics['dice']:.4f}\n"
                  f"IoU: {metrics['iou']:.4f}\n"
                  f"HD: {metrics['hd']:.4f}")
    
    with open(test_results_dir, 'w') as f:
        f.write(results_str)
    
    print('='*80)
    print(results_str)
    print('='*80)
    
    file_log.write('\n' + '='*80 + '\n')
    file_log.write(results_str + '\n')
    file_log.write('='*80 + '\n')
    file_log.flush()

def update_ema(model_teacher, model, alpha_teacher, iteration):
    """Update teacher model using EMA"""
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    for ema_param, param in zip(model_teacher.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str,default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=2)
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method']=args.adapt_method
    config['model_adapt']['num_domains']=args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    folds_to_train = [2,1,3,4,5]
    
    for fold in folds_to_train:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        exp_dir = '{}/{}/fold{}'.format(config.data.save_folder, args.exp, fold)
        
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = '{}/best.pth'.format(exp_dir)
        test_results_dir = '{}/test_results.txt'.format(exp_dir)

        # Store yml file
        if config.debug == False:
            yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
            
        file_log = open('{}/log.txt'.format(exp_dir), 'w')
        
        # Train the model
        main(config)
        
        # Close log file
        file_log.close()
        
        # Clear GPU memory
        torch.cuda.empty_cache()
