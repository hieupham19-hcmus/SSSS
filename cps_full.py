"""
Cross Pseudo Supervision (CPS) training implementation for medical image segmentation.
Uses DeepLabV3+ with ResNet50 backbone and GeneralizedDiceFocalLoss.

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
import cv2

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
from monai.losses import GeneralizedDiceFocalLoss, HausdorffDTLoss, DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU

from Datasets.create_dataset import get_dataset_without_full_label_without_val, SkinDataset2, StrongWeakAugment4
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Utils.metrics import calc_dice, calc_iou, calc_hd
from Models.DeepLabV3Plus import deeplabv3plus_resnet50

def main(config):
    """
    Main training function.
    
    Args:
        config: Configuration object containing training parameters
    """
    # Setup datasets and dataloaders
    dataset = get_dataset_without_full_label_without_val(
        config, 
        img_size=config.data.img_size,
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=SkinDataset2,
        ulb_dataset=StrongWeakAugment4
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
        dataset['lb_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    test_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Initialize models
    model1 = deeplabv3plus_resnet50(num_classes=3, output_stride=8, pretrained_backbone=True)
    model2 = deeplabv3plus_resnet50(num_classes=3, output_stride=8, pretrained_backbone=True)

    # Print model statistics
    total_params = sum(p.numel() for p in model1.parameters())
    total_trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model1 = model1.cuda()
    model2 = model2.cuda()

    # Setup loss function
    criterion = DiceCELoss(
        include_background=True,
        to_onehot_y=False,
        softmax=True
    )
    criterion = [criterion]
    
    global loss_weights
    loss_weights = [1.0]

    # Train and test
    model = train_val(config, model1, model2, train_loader, val_loader, criterion)
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
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def flatten_features(features):
    return [f.view(f.size(0), -1) for f in features]

def calculate_cosine_similarity(features_1, features_2):
    """
    Calculate mean cosine similarity between two sets of feature maps.
    
    Args:
        features_1: First set of feature maps
        features_2: Second set of feature maps
        
    Returns:
        Mean cosine similarity across all feature map pairs
    """
    # Flatten feature maps
    flattened_1 = flatten_features(features_1)
    flattened_2 = flatten_features(features_2)
    
    # Calculate cosine similarity for each feature map pair
    cosine_similarities = []
    for f1, f2 in zip(flattened_1, flattened_2):
        cos_sim = F.cosine_similarity(f1, f2, dim=1, eps=1e-6)
        cosine_similarities.append(cos_sim)
    
    # Return mean similarity
    return torch.stack(cosine_similarities).mean()



# =======================================================================================================
def train_val(config, model1, model2, train_loader, val_loader, criterion):
    """
    Training and validation function with Cross Pseudo Supervision.
    
    Args:
        config: Training configuration
        model1: First model
        model2: Second model
        train_loader: Dictionary containing labeled and unlabeled data loaders
        val_loader: Validation data loader
        criterion: Loss function(s)
    
    Returns:
        Best performing model
    """
    # Setup optimizers
    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model1.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    optimizer2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model2.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)

    # Initialize MONAI metrics for training
    train_dice_1 = DiceMetric(include_background=True, num_classes=3, reduction="mean")
    train_dice_2 = DiceMetric(include_background=True, num_classes=3, reduction="mean")
    
    max_score = -float('inf')
    best_epoch = 0
    w_dice = 0.5  # Weight for Dice score
    w_hd = 0.5    # Weight for HD score
    model = model1  # Default model to return
    
    torch.save(model.state_dict(), best_model_dir)
    
    for epoch in range(config.train.num_epochs):
        start = time.time()
        
        # Training phase
        model1.train()
        model2.train()
        train_metrics = {
            'dice_1': 0, 'dice_2': 0, 'loss': 0
        }
        num_train = 0
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        
        # Reset metrics at start of epoch
        train_dice_1.reset()
        train_dice_2.reset()
        
        for idx, (batch, batch_w_s) in enumerate(train_loop):
            # Get batch data
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            weak_batch = batch_w_s['img_w'].cuda().float()
            
            sup_batch_len = img.shape[0]
            unsup_batch_len = weak_batch.shape[0]
            
            # Forward passes
            output1 = torch.softmax(model1(img), dim=1)
            output2 = torch.softmax(model2(img), dim=1)
            
            # Supervised losses
            sup_loss_1 = criterion[0](output1, label)
            sup_loss_2 = criterion[0](output2, label)
            
            # Generate pseudo-labels for CPS
            with torch.no_grad():
                outputs_u1 = torch.softmax(model1(weak_batch), dim=1)
                outputs_u2 = torch.softmax(model2(weak_batch), dim=1)
                
                # Create pseudo-labels based on confidence threshold
                pseudo_mask_u1 = (outputs_u1.max(dim=1)[0] > config.semi.conf_thresh).float().unsqueeze(1)
                pseudo_mask_u2 = (outputs_u2.max(dim=1)[0] > config.semi.conf_thresh).float().unsqueeze(1)
                
                # Convert to one-hot
                pseudo_u1 = torch.zeros_like(outputs_u1)
                pseudo_u1.scatter_(1, outputs_u1.argmax(dim=1, keepdim=True), 1)
                pseudo_u2 = torch.zeros_like(outputs_u2)
                pseudo_u2.scatter_(1, outputs_u2.argmax(dim=1, keepdim=True), 1)
            
            # Unsupervised losses
            unsup_loss_1 = criterion[0](outputs_u1, pseudo_u2)
            unsup_loss_2 = criterion[0](outputs_u2, pseudo_u1)
            
            # Calculate consistency weight
            consistency_weight = get_current_consistency_weight(idx // 150)
            
            # Total losses
            loss_1 = sup_loss_1 + unsup_loss_1 * consistency_weight * (sup_batch_len / unsup_batch_len)
            loss_2 = sup_loss_2 + unsup_loss_2 * consistency_weight * (sup_batch_len / unsup_batch_len)
            loss = loss_1 + loss_2
            
            # Optimization step
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Convert predictions to one-hot format
                output1_onehot = torch.zeros_like(output1)
                output1_onehot.scatter_(1, output1.argmax(dim=1, keepdim=True), 1)
                
                output2_onehot = torch.zeros_like(output2)
                output2_onehot.scatter_(1, output2.argmax(dim=1, keepdim=True), 1)
                
                # Update MONAI metrics
                train_dice_1(y_pred=output1_onehot, y=label)
                train_dice_2(y_pred=output2_onehot, y=label)
                
                # Update loss metric
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * sup_batch_len) / (num_train + sup_batch_len)
                num_train += sup_batch_len
            
            # Update progress bar with current batch metrics
            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice1': f"{train_dice_1.aggregate().item():.4f}",
                'Dice2': f"{train_dice_2.aggregate().item():.4f}"
            })
            
            if config.debug:
                break
        
        # Get final training metrics for the epoch
        train_metrics['dice_1'] = train_dice_1.aggregate().item()
        train_metrics['dice_2'] = train_dice_2.aggregate().item()
        
        # Validation phase for both models
        val_metrics_1 = validate_model(model1, val_loader, criterion)
        val_metrics_2 = validate_model(model2, val_loader, criterion)
        
        # Select better model based on combined score
        score_1 = w_dice * val_metrics_1['dice'] + w_hd * np.exp(-val_metrics_1['hd']/100)
        score_2 = w_dice * val_metrics_2['dice'] + w_hd * np.exp(-val_metrics_2['hd']/100)
        
        current_score = max(score_1, score_2)
        current_model = model1 if score_1 > score_2 else model2
        
        # Save best model
        if current_score > max_score and epoch >= 0:
            max_score = current_score
            best_epoch = epoch
            model = current_model
            torch.save(model.state_dict(), best_model_dir)
            
            message = (f'New best epoch {epoch}! '
                      f'Score: {current_score:.4f} '
                      f'Dice: {val_metrics_1["dice"]:.4f} '
                      f'HD: {val_metrics_1["hd"]:.4f}')
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
        
        # Update learning rate
        scheduler1.step()
        scheduler2.step()
        
        # Log epoch time
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        
        if config.debug:
            break
    
    print(f'Training completed. Best epoch: {best_epoch}')
    return model

def validate_model(model, val_loader, criterion):
    """
    Validate a single model using MONAI metrics.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function(s)
    
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=True, num_classes=3, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
        
        with torch.no_grad():
            output = torch.softmax(model(img), dim=1)
            loss = criterion[0](output, label)
            
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
    
    # Reset metrics for next validation
    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()
    
    return metrics

def test(config, model, model_dir, test_loader, criterion):
    """
    Test the model on the test set.
    
    Args:
        config: Test configuration
        model: Model to test
        model_dir: Path to saved model weights
        test_loader: Test data loader
        criterion: Loss function(s)
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
    parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
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
    
    folds_to_train = [1,2,3,4,5]
    
    for fold in folds_to_train:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        # Update paths for each fold
        # exp_dir = '{}/{}_{}/fold{}'.format(config.data.save_folder, args.exp, config['data']['supervised_ratio'], fold)
        exp_dir = '{}/{}/fold{}'.format(config.data.save_folder, args.exp, fold)
        
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = '{}/best.pth'.format(exp_dir)
        test_results_dir = '{}/test_results.txt'.format(exp_dir)

        # Store yml file for each fold
        if config.debug == False:
            yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
            
        file_log = open('{}/log.txt'.format(exp_dir), 'w')
        
        # Train the model for this fold
        main(config)
        
        # Close the log file
        file_log.close()
        
        # Clear GPU memory between folds
        torch.cuda.empty_cache()