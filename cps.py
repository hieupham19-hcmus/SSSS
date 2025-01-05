'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 16 --adapt_method False --num_domains 1 --dataset PH2  --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
'''
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime
import cv2

import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics

from Datasets.create_dataset import *
from Datasets.transform import normalize
from Utils.losses import dice_loss, hausdorff_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Utils.metrics import calc_metrics, calc_hd, calc_dice, calc_iou

import torch.nn.functional as F
from Models.Transformer.SwinUnet import SwinUnet
# from Models.unet import UNet
from Models.unetCCT import UNet

from itertools import cycle
from tqdm import tqdm

torch.cuda.empty_cache()

# At the top of the file, add loss_weights
loss_weights = [0.3, 0.5, 0.2]  # Same weights as in train_sup.py

def main(config):
    
    # dataset = get_dataset(config, img_size=config.data.img_size, 
    #                                                 supervised_ratio=config.data.supervised_ratio, 
    #                                                 train_aug=config.data.train_aug,
    #                                                 k=config.fold,
    #                                                 ulb_dataset=StrongWeakAugment2,
    #                                                 lb_dataset=SkinDataset2)
    
    dataset = get_dataset_without_full_label(config, img_size=config.data.img_size,
                                                train_aug=config.data.train_aug,
                                                k=config.fold,
                                                lb_dataset=SkinDataset2)

    
    l_train_loader = torch.utils.data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    u_train_loader = torch.utils.data.DataLoader(dataset['ulb_dataset'],
                                                batch_size=config.train.u_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    test_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    train_loader = {'l_loader':l_train_loader, 'u_loader':u_train_loader}
    print(len(u_train_loader), len(l_train_loader))

    
    model1  = UNet(in_chns=3, class_num=3)
    model2  = UNet(in_chns=3, class_num=3)
    
    # model1  = SwinUnet(img_size=config.data.img_size)
    # model2  = SwinUnet(img_size=config.data.img_size)





    total_trainable_params = sum(
                    p.numel() for p in model1.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model1.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    
    # from thop import profile
    # input = torch.randn(1,3,224,224)
    # flops, params = profile(model, (input,))
    # print(f"total flops : {flops/1e9} G")

    # test model
    # x = torch.randn(5,3,224,224)
    # y = model(x)
    # print(y.shape)

    model1 = model1.cuda()
    model2 = model2.cuda()
    
    # Update the criterion initialization in main()
    criterion = [
        nn.CrossEntropyLoss(), 
        lambda pred, target: dice_loss(torch.softmax(pred, dim=1), target),
        lambda pred, target: hausdorff_loss(torch.softmax(pred, dim=1), target)
    ]

    model = train_val(config, model1, model2, train_loader, val_loader, criterion)
    test(config, model, best_model_dir, test_loader, criterion)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
def flatten_features(features):
    # Flatten tất cả các feature maps trong danh sách
    return [f.view(f.size(0), -1) for f in features]  # Chuyển từ [batch_size, channels, height, width] -> [batch_size, channels*height*width]

def calculate_cosine_similarity(features_1, features_2):
    # Flatten các feature maps
    flattened_1 = flatten_features(features_1)
    flattened_2 = flatten_features(features_2)
    
    cosine_similarities = []

    # Tính cosine similarity cho từng cặp feature map
    for f1, f2 in zip(flattened_1, flattened_2):
        cos_sim = F.cosine_similarity(f1, f2, dim=1, eps=1e-6)  # Tính cosine similarity giữa 2 tensor đã flatten
        cosine_similarities.append(cos_sim)
    
    # Trung bình các cosine similarity
    return torch.stack(cosine_similarities).mean()
# =======================================================================================================
def train_val(config, model1, model2, train_loader, val_loader, criterion):
    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
        optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_score = -float('inf')  # Track combined score
    best_epoch = 0
    
    # Thêm weights cho việc kết hợp metrics
    w_dice = 0.5  # Weight cho Dice score 
    w_hd = 0.5    # Weight cho HD score
    
    model = model1
    # create training data loading iteration
    
    torch.save(model.state_dict(), best_model_dir)
    for epoch in range(epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model1.train()
        model2.train()
        dice_train_sum_1 = 0 
        iou_train_sum_1 = 0
        hd_train_sum_1 = 0
        
        dice_train_sum_2 = 0
        iou_train_sum_2 = 0
        hd_train_sum_2 = 0
        
        loss_train_sum = 0
        num_train = 0
        iter = 0
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        
        # Add tqdm progress bar
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        for idx, (batch, batch_w_s) in enumerate(train_loop):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            weak_batch = batch_w_s['img_w'].cuda().float()
            
            sup_batch_len = img.shape[0]
            unsup_batch_len = weak_batch.shape[0]
            
            # Model 1 forward pass
            output1, lb_features_1 = model1(img, return_features=True)
            output1 = torch.softmax(output1, dim=1)  # Changed from sigmoid to softmax
            
            # Calculate supervised losses for model 1
            losses_l1 = []
            for function in criterion:
                losses_l1.append(function(output1, label))
            sup_loss_1 = sum(w * l for w, l in zip(loss_weights, losses_l1))
            
            # Model 2 forward pass
            output2, lb_features_2 = model2(img, return_features=True)
            output2 = torch.softmax(output2, dim=1)  # Changed from sigmoid to softmax
            
            # Calculate supervised losses for model 2
            losses_l2 = []
            for function in criterion:
                losses_l2.append(function(output2, label))
            sup_loss_2 = sum(w * l for w, l in zip(loss_weights, losses_l2))
            
            # CPS updates for multi-class
            outputs_u1, ulb_features_1 = model1(weak_batch, return_features=True)
            outputs_u1 = torch.softmax(outputs_u1, dim=1)
            pseudo_mask_u1 = (outputs_u1.max(dim=1)[0] > config.semi.conf_thresh).float().unsqueeze(1)
            outputs_weak_masked_u1 = outputs_u1 * pseudo_mask_u1
            # Chuyển pseudo labels về one-hot encoding
            pseudo_u1 = torch.zeros_like(outputs_u1)
            pseudo_u1.scatter_(1, outputs_weak_masked_u1.argmax(dim=1, keepdim=True), 1)

            outputs_u2, ulb_features_2 = model2(weak_batch, return_features=True)
            outputs_u2 = torch.softmax(outputs_u2, dim=1)
            pseudo_mask_u2 = (outputs_u2.max(dim=1)[0] > config.semi.conf_thresh).float().unsqueeze(1)
            outputs_weak_masked_u2 = outputs_u2 * pseudo_mask_u2
            # Chuyển pseudo labels về one-hot encoding
            pseudo_u2 = torch.zeros_like(outputs_u2)
            pseudo_u2.scatter_(1, outputs_weak_masked_u2.argmax(dim=1, keepdim=True), 1)
            
            # Calculate unsupervised losses
            losses_u1 = []
            for function in criterion:
                losses_u1.append(function(outputs_u1, pseudo_u2))
            unsup_loss_1 = sum(w * l for w, l in zip(loss_weights, losses_u1))
                
            losses_u2 = []
            for function in criterion:
                losses_u2.append(function(outputs_u2, pseudo_u1))
            unsup_loss_2 = sum(w * l for w, l in zip(loss_weights, losses_u2))
            
            # Calculate consistency weight
            consistency_weight = get_current_consistency_weight(iter // 150)
            
            # Combine losses
            loss_1 = sup_loss_1 + unsup_loss_1 * consistency_weight * (sup_batch_len / unsup_batch_len)
            loss_2 = sup_loss_2 + unsup_loss_2 * consistency_weight * (sup_batch_len / unsup_batch_len)
            
            # Total loss
            loss = loss_1 + loss_2 
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            
            loss_train_sum += loss.item() * sup_batch_len
            
            # Calculate metrics and update progress bar
            with torch.no_grad():
                output1_np = output1.argmax(dim=1).cpu().numpy()
                output2_np = output2.argmax(dim=1).cpu().numpy()
                label_np = label.argmax(dim=1).cpu().numpy()

                # Calculate metrics per class and average
                dice_train_1 = np.mean([calc_dice(output1_np == i, label_np == i) 
                                      for i in range(output1.shape[1])])
                iou_train_1 = np.mean([calc_iou(output1_np == i, label_np == i) 
                                     for i in range(output1.shape[1])])
                hd_train_1 = np.mean([calc_hd(output1_np == i, label_np == i) 
                                    for i in range(output1.shape[1])])
                
                dice_train_2 = np.mean([calc_dice(output2_np == i, label_np == i) 
                                      for i in range(output2.shape[1])])
                iou_train_2 = np.mean([calc_iou(output2_np == i, label_np == i) 
                                     for i in range(output2.shape[1])])
                hd_train_2 = np.mean([calc_hd(output2_np == i, label_np == i) 
                                    for i in range(output2.shape[1])])

                # Update metric sums
                dice_train_sum_1 += dice_train_1 * sup_batch_len
                iou_train_sum_1 += iou_train_1 * sup_batch_len
                hd_train_sum_1 += hd_train_1 * sup_batch_len
                
                dice_train_sum_2 += dice_train_2 * sup_batch_len
                iou_train_sum_2 += iou_train_2 * sup_batch_len
                hd_train_sum_2 += hd_train_2 * sup_batch_len

                # Update progress bar with simplified metrics
                train_loop.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice1': f'{dice_train_1:.4f}',
                    'Dice2': f'{dice_train_2:.4f}',
                    'HD1': f'{hd_train_1:.4f}',
                    'HD2': f'{hd_train_2:.4f}'
                })

            # Log detailed loss components
            file_log.write(f'Epoch {epoch}, iter {iter}:\n')
            file_log.write(f'Model 1 - Sup Loss: {sup_loss_1:.4f}, Unsup Loss: {unsup_loss_1:.4f}\n')
            file_log.write(f'Model 2 - Sup Loss: {sup_loss_2:.4f}, Unsup Loss: {unsup_loss_2:.4f}\n')
            file_log.flush()

            num_train += sup_batch_len
            iter += 1
            if config.debug: break
                

        # Print epoch summary with simplified metrics
        print(f'Epoch {epoch}, Training || '
              f'Loss: {loss_train_sum/num_train:.4f}, '
              f'Dice1: {dice_train_sum_1/num_train:.4f}, '
              f'Dice2: {dice_train_sum_2/num_train:.4f}, '
              f'HD1: {hd_train_sum_1/num_train:.4f}, '
              f'HD2: {hd_train_sum_2/num_train:.4f}, '
              f'IoU1: {iou_train_sum_1/num_train:.4f}, '
              f'IoU2: {iou_train_sum_2/num_train:.4f}')

        # -----------------------------------------------------------------
        # validate 1
        # ----------------------------------------------------------------
        model1.eval()
        
        dice_val_sum = 0
        iou_val_sum = 0
        loss_val_sum = 0
        hd_val_sum = 0  # Thêm HD sum
        num_val = 0

        # Add tqdm progress bar for validation
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch} Validation Model 1', leave=False)
        for batch in val_loop:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]

            with torch.no_grad():
                output = model1(img)
                output = torch.softmax(output, dim=1)

                # Calculate loss
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss = sum(w * l for w, l in zip(loss_weights, losses))
                loss_val_sum += loss.item() * batch_len

                # Calculate metrics
                output_np = output.argmax(dim=1).cpu().numpy()
                label_np = label.argmax(dim=1).cpu().numpy()
                
                # Calculate per-class metrics
                dice_scores = [calc_dice(output_np == i, label_np == i) 
                             for i in range(output.shape[1])]
                iou_scores = [calc_iou(output_np == i, label_np == i) 
                            for i in range(output.shape[1])]
                hd_scores = [calc_hd(output_np == i, label_np == i) 
                           for i in range(output.shape[1])]
                
                dice_val = np.mean(dice_scores)
                iou_val = np.mean(iou_scores)
                hd_val = np.mean(hd_scores)
                
                dice_val_sum += dice_val * batch_len
                iou_val_sum += iou_val * batch_len
                hd_val_sum += hd_val * batch_len

                # Update progress bar
                val_loop.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice_val:.4f}',
                    'HD': f'{hd_val:.4f}'
                })

                num_val += batch_len
                if config.debug: break

        # Calculate epoch metrics for model 1
        loss_val_epoch_1 = loss_val_sum/num_val
        dice_val_epoch_1 = dice_val_sum/num_val
        iou_val_epoch_1 = iou_val_sum/num_val
        hd_val_epoch_1 = hd_val_sum/num_val

        # Calculate combined score for model 1
        hd_norm_1 = np.exp(-hd_val_epoch_1/100)  # Normalize HD score
        combined_score_1 = w_dice * dice_val_epoch_1 + w_hd * hd_norm_1

        # Log validation metrics for model 1
        file_log.write(f'Epoch {epoch}, Model 1 Validation || '
                      f'Loss: {loss_val_epoch_1:.4f}, '
                      f'Dice: {dice_val_epoch_1:.4f}, '
                      f'HD: {hd_val_epoch_1:.4f}, '
                      f'Combined Score: {combined_score_1:.4f}, '
                      f'IoU: {iou_val_epoch_1:.4f}')
        file_log.flush()

        # -----------------------------------------------------------------
        # validate 2
        # ----------------------------------------------------------------
        model2.eval()
        
        dice_val_sum = 0
        iou_val_sum = 0
        loss_val_sum = 0
        hd_val_sum = 0  # Thêm HD sum
        num_val = 0

        # Add tqdm progress bar for validation
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch} Validation Model 2', leave=False)
        for batch in val_loop:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]

            with torch.no_grad():
                output = model2(img)
                output = torch.softmax(output, dim=1)

                # Calculate loss
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss = sum(w * l for w, l in zip(loss_weights, losses))
                loss_val_sum += loss.item() * batch_len

                # Calculate metrics
                output_np = output.argmax(dim=1).cpu().numpy()
                label_np = label.argmax(dim=1).cpu().numpy()
                
                # Calculate per-class metrics
                dice_scores = [calc_dice(output_np == i, label_np == i) 
                             for i in range(output.shape[1])]
                iou_scores = [calc_iou(output_np == i, label_np == i) 
                            for i in range(output.shape[1])]
                hd_scores = [calc_hd(output_np == i, label_np == i) 
                           for i in range(output.shape[1])]
                
                dice_val = np.mean(dice_scores)
                iou_val = np.mean(iou_scores)
                hd_val = np.mean(hd_scores)
                
                dice_val_sum += dice_val * batch_len
                iou_val_sum += iou_val * batch_len
                hd_val_sum += hd_val * batch_len

                # Update progress bar
                val_loop.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice_val:.4f}',
                    'HD': f'{hd_val:.4f}'
                })

                num_val += batch_len
                if config.debug: break

        # Calculate epoch metrics for model 2
        loss_val_epoch_2 = loss_val_sum/num_val
        dice_val_epoch_2 = dice_val_sum/num_val
        iou_val_epoch_2 = iou_val_sum/num_val
        hd_val_epoch_2 = hd_val_sum/num_val

        # Calculate combined score for model 2
        hd_norm_2 = np.exp(-hd_val_epoch_2/100)  # Normalize HD score
        combined_score_2 = w_dice * dice_val_epoch_2 + w_hd * hd_norm_2

        # Log validation metrics for model 2
        file_log.write(f'Epoch {epoch}, Model 2 Validation || '
                      f'Loss: {loss_val_epoch_2:.4f}, '
                      f'Dice: {dice_val_epoch_2:.4f}, '
                      f'HD: {hd_val_epoch_2:.4f}, '
                      f'Combined Score: {combined_score_2:.4f}, '
                      f'IoU: {iou_val_epoch_2:.4f}')
        file_log.flush()

        # After validating both models, compare and save the better one
        if combined_score_1 > combined_score_2:
            current_score = combined_score_1
            current_model = model1
            current_dice = dice_val_epoch_1
            current_hd = hd_val_epoch_1
        else:
            current_score = combined_score_2
            current_model = model2
            current_dice = dice_val_epoch_2
            current_hd = hd_val_epoch_2

        # Save model if current score improves
        if current_score > max_score:
            max_score = current_score
            best_epoch = epoch
            model = current_model
            torch.save(model.state_dict(), best_model_dir)
            
            message = (f'New best epoch {epoch}! '
                      f'Combined score improved to {current_score:.4f} ==========\n'
                      f'Dice: {current_dice:.4f}, HD: {current_hd:.4f} ==========')
            
            file_log.write(message + '\n')
            file_log.flush()
            print(message)

        # Update learning rate schedulers
        scheduler1.step()
        scheduler2.step()

        end = time.time()
        time_elapsed = end-start
        file_log.write('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
        file_log.flush()
        print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))

        # end one epoch
        if config.debug: return
    
    file_log.write('Complete training ---------------------------------------------------- \n The best epoch is {}\n'.format(best_epoch))
    file_log.flush()
    print('Complete training ---------------------------------------------------- \n The best epoch is {}'.format(best_epoch))

    return model




# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_sum = 0
    iou_test_sum = 0
    hd_test_sum = 0
    loss_test_sum = 0
    num_test = 0

    test_loop = tqdm(test_loader, desc='Testing', leave=True)
    for batch in test_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
            
        with torch.no_grad():
            output = model(img)
            output = torch.softmax(output, dim=1)

            # Calculate loss
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            loss = sum(w * l for w, l in zip(loss_weights, losses))
            loss_test_sum += loss.item() * batch_len

            # Calculate metrics
            output_np = output.argmax(dim=1).cpu().numpy()
            label_np = label.argmax(dim=1).cpu().numpy()
            
            # Calculate per-class metrics and average
            dice_scores = [calc_dice(output_np == i, label_np == i) 
                         for i in range(output.shape[1])]
            iou_scores = [calc_iou(output_np == i, label_np == i) 
                        for i in range(output.shape[1])]
            hd_scores = [calc_hd(output_np == i, label_np == i) 
                       for i in range(output.shape[1])]
            
            dice_test = np.mean(dice_scores)
            iou_test = np.mean(iou_scores)
            hd_test = np.mean(hd_scores)
            
            dice_test_sum += dice_test * batch_len
            iou_test_sum += iou_test * batch_len
            hd_test_sum += hd_test * batch_len

            # Update progress bar with simplified metrics
            test_loop.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice_test:.4f}',
                'HD': f'{hd_test:.4f}'
            })

        num_test += batch_len
        if config.debug: break

    # Calculate final averages with simplified metrics
    metrics_dict = {
        'Loss': loss_test_sum/num_test,
        'Dice': dice_test_sum/num_test,
        'IoU': iou_test_sum/num_test,
        'HD': hd_test_sum/num_test
    }

    # Save and print results
    with open(test_results_dir, 'w') as f:
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics_dict.items()])
        f.write(metrics_str)

    print('\n' + '='*80)
    print('Test Results:')
    for metric, value in metrics_dict.items():
        print(f'{metric}: {value:.4f}')
    print('='*80)
    
    # Log to file
    file_log.write('\n' + '='*80 + '\n')
    file_log.write('Test Results:\n')
    for metric, value in metrics_dict.items():
        file_log.write(f'{metric}: {value:.4f}\n')
    file_log.write('='*80 + '\n')
    file_log.flush()

    return




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