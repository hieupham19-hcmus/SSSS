"""
FixMatch implementation for semi-supervised medical image segmentation.
Uses DeepLabV3+ with ResNet50 backbone and combined CE + Dice loss.

Example usage:
nohup python -u multi_train_adapt.py \
    --exp_name test \
    --config_yml Configs/multi_train_local.yml \
    --model MedFormer \
    --batch_size 16 \
    --adapt_method False \
    --num_domains 1 \
    --dataset PH2 \
    --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
"""

import os
import time
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle

from Datasets.create_dataset import get_dataset_without_full_label, SkinDataset2
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Utils.metrics import calc_dice, calc_iou, calc_hd
from Utils.losses import dice_loss
from Models.DeepLabV3Plus import deeplabv3plus_resnet50
from monai.losses import GeneralizedDiceFocalLoss

def main(config):
    """
    Main training function.
    
    Args:
        config: Configuration object containing training parameters
    """
    # Setup datasets and dataloaders
    dataset = get_dataset_without_full_label(
        config, 
        img_size=config.data.img_size,
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=SkinDataset2
    )
    
    l_train_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    u_train_loader = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    test_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Initialize model
    model = deeplabv3plus_resnet50(num_classes=3, output_stride=8, pretrained_backbone=True)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model = model.cuda()

    # Setup loss function - thay đổi criterion
    criterion = [
        GeneralizedDiceFocalLoss(
            include_background=True,
            to_onehot_y=False,  # Vì labels đã là one-hot
            softmax=True
        )
    ]
    
    # Train and test
    if config.test.only_test:
        test(config, model, best_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
        test(config, model, best_model_dir, test_loader, criterion)

def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    """Calculate the consistency weight for the current epoch."""
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def train_val(config, model, train_loader, val_loader, criterion):
    """
    Training and validation function with FixMatch.
    
    Args:
        config: Training configuration
        model: Model to train
        train_loader: Dictionary containing labeled and unlabeled data loaders
        val_loader: Validation data loader
        criterion: List of loss functions
    """
    # Setup optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    max_score = -float('inf')
    best_epoch = 0
    w_dice = 0.5  # Weight for Dice score
    w_hd = 0.5    # Weight for HD score
    warmup_epochs = 15
    
    torch.save(model.state_dict(), best_model_dir)
    
    for epoch in range(config.train.num_epochs):
        start = time.time()
        
        # Training phase
        model.train()
        train_metrics = {
            'dice': 0,
            'iou': 0,
            'loss': 0
        }
        num_train = 0
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        
        for idx, (batch, batch_w_s) in enumerate(train_loop):
            # Get batch data
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            weak_batch = batch_w_s['img_w'].cuda().float()
            strong_batch = batch_w_s['img_s'].cuda().float()
            
            sup_batch_len = img.shape[0]
            unsup_batch_len = weak_batch.shape[0]
            
            # Forward pass on labeled data
            output = model(img)
            output = torch.softmax(output, dim=1)
            
            if epoch < warmup_epochs:
                # Only supervised loss during warmup
                loss = criterion[0](output, label)
            
            else:
                # FixMatch training
                with torch.no_grad():
                    # Generate pseudo-labels from weak augmentations
                    outputs_weak = model(weak_batch)
                    max_probs, pseudo_labels = torch.max(outputs_weak.softmax(dim=1), dim=1)
                    mask = max_probs.ge(config.semi.conf_thresh).float()
                    
                    # Convert pseudo-labels to one-hot
                    pseudo_labels_one_hot = F.one_hot(pseudo_labels, num_classes=3).permute(0, 3, 1, 2).float()
                    pseudo_labels_masked = pseudo_labels_one_hot * mask.unsqueeze(1)
                
                # Get predictions on strong augmentations
                outputs_strong = model(strong_batch)
                
                # Calculate supervised and unsupervised losses
                sup_loss = criterion[0](output, label)
                unsup_loss = criterion[0](outputs_strong, pseudo_labels_masked)
                
                # Combine losses
                consistency_weight = get_current_consistency_weight(idx // 150)
                loss = sup_loss + unsup_loss * consistency_weight * (sup_batch_len / unsup_batch_len)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                output_np = output.argmax(dim=1).cpu().numpy()
                label_np = label.argmax(dim=1).cpu().numpy()
                
                # Calculate per-class metrics
                batch_dice = 0
                batch_iou = 0
                num_classes = output.shape[1]
                
                for i in range(num_classes):
                    if i > 0:  # Skip background class
                        batch_dice += calc_dice(output_np == i, label_np == i)
                        batch_iou += calc_iou(output_np == i, label_np == i)
                
                # Average across foreground classes
                num_fg_classes = num_classes - 1
                batch_dice /= num_fg_classes
                batch_iou /= num_fg_classes
                
                # Update running averages
                train_metrics['dice'] = (train_metrics['dice'] * num_train + batch_dice * sup_batch_len) / (num_train + sup_batch_len)
                train_metrics['iou'] = (train_metrics['iou'] * num_train + batch_iou * sup_batch_len) / (num_train + sup_batch_len)
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * sup_batch_len) / (num_train + sup_batch_len)
                
                num_train += sup_batch_len
            
            # Update progress bar
            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{train_metrics['dice']:.4f}"
            })
            
            if config.debug:
                break
        
        # Log training metrics
        log_message = (f'Epoch {epoch}, Total train steps {idx} || '
                      f'Loss: {train_metrics["loss"]:.5f}, '
                      f'Dice: {train_metrics["dice"]:.4f}, '
                      f'IoU: {train_metrics["iou"]:.4f}')
        print(log_message)
        file_log.write(log_message + '\n')
        file_log.flush()
        
        # Validation phase
        val_metrics = validate_model(model, val_loader, criterion)
        
        # Calculate combined score
        hd_norm = np.exp(-val_metrics['hd']/100)
        combined_score = w_dice * val_metrics['dice'] + w_hd * hd_norm
        
        # Save best model
        if combined_score > max_score and epoch >= 30:
            max_score = combined_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)
            
            message = (f'New best epoch {epoch}! '
                      f'Score: {combined_score:.4f} '
                      f'Dice: {val_metrics["dice"]:.4f} '
                      f'HD: {val_metrics["hd"]:.4f}')
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

def validate_model(model, val_loader, criterion):
    """
    Validate model performance.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss functions
    
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
        
        with torch.no_grad():
            output = model(img)
            loss = criterion[0](output, label)
            
            # Calculate metrics
            output_np = output.argmax(dim=1).cpu().numpy()
            label_np = label.argmax(dim=1).cpu().numpy()
            
            batch_dice = 0
            batch_iou = 0
            batch_hd = 0
            num_classes = output.shape[1]
            
            for i in range(num_classes):
                if i > 0:  # Skip background
                    batch_dice += calc_dice(output_np == i, label_np == i)
                    batch_iou += calc_iou(output_np == i, label_np == i)
                    batch_hd += calc_hd(output_np == i, label_np == i)
            
            # Average across foreground classes
            num_fg_classes = num_classes - 1
            batch_dice /= num_fg_classes
            batch_iou /= num_fg_classes
            batch_hd /= num_fg_classes
            
            # Update running averages
            metrics['dice'] = (metrics['dice'] * num_val + batch_dice * batch_len) / (num_val + batch_len)
            metrics['iou'] = (metrics['iou'] * num_val + batch_iou * batch_len) / (num_val + batch_len)
            metrics['hd'] = (metrics['hd'] * num_val + batch_hd * batch_len) / (num_val + batch_len)
            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * batch_len) / (num_val + batch_len)
            
            num_val += batch_len
            
            val_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{metrics['dice']:.4f}"
            })
    
    return metrics

def test(config, model, model_dir, test_loader, criterion):
    """
    Test the model on the test set.
    
    Args:
        config: Test configuration
        model: Model to test
        model_dir: Path to saved model weights
        test_loader: Test data loader
        criterion: Loss functions
    """
    model.load_state_dict(torch.load(model_dir))
    metrics = validate_model(model, test_loader, criterion)
    
    # Save and print results
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with FixMatch')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    
    args = parser.parse_args()
    
    # Load and update config
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model_adapt']['adapt_method'] = args.adapt_method
    config['model_adapt']['num_domains'] = args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    # Setup CUDA and seeds
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])
    
    # Print configuration
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")
    
    store_config = config
    config = DotDict(config)
    
    # Train each fold
    for fold in [1, 2, 3, 4, 5]:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        # Setup directories
        exp_dir = f"{config.data.save_folder}/{args.exp}/fold{fold}"
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = f'{exp_dir}/best.pth'
        test_results_dir = f'{exp_dir}/test_results.txt'
        
        # Save config
        if not config.debug:
            yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))
        
        # Train fold
        with open(f'{exp_dir}/log.txt', 'w') as file_log:
            main(config)
        
        torch.cuda.empty_cache()
