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

from tqdm import tqdm

import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics
import numpy as np
import torch.nn.functional as F
from Datasets.create_dataset import *
from Datasets.transform import normalize
from Utils.losses import dice_loss, hausdorff_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Utils.metrics import calc_metrics, calc_hd, calc_dice, calc_iou
from itertools import cycle
from Models.unetCCT import UNet
torch.cuda.empty_cache()

loss_weights = [0.3, 0.5, 0.2]  # Same weights as in train_sup.py and ccvc.py

def main(config):
    
    # dataset = get_dataset(config, img_size=config.data.img_size, 
    #                                                 supervised_ratio=config.data.supervised_ratio, 
    #                                                 train_aug=config.data.train_aug,
    #                                                 k=config.fold,
    #                                                 lb_dataset=SkinDataset2,
    #                                                 ulb_dataset=StrongWeakAugment2)
    dataset = get_dataset_without_full_label(config, img_size=config.data.img_size,
                                                train_aug=config.data.train_aug,
                                                k=config.fold,
                                                lb_dataset=SkinDataset2)

    l_train_loader = torch.utils.data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    u_train_loader = torch.utils.data.DataLoader(dataset['ulb_dataset'],
                                                batch_size=config.train.u_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    train_loader = {'l_loader':l_train_loader, 'u_loader':u_train_loader}
    print(len(u_train_loader), len(l_train_loader))

    
    model = UNet(in_chns=3, class_num=3)




    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    

    model = model.cuda()
    

    criterion = [
        nn.CrossEntropyLoss(), 
        lambda pred, target: dice_loss(torch.softmax(pred, dim=1), target),
        lambda pred, target: hausdorff_loss(torch.softmax(pred, dim=1), target)
    ]


    # only test
    if config.test.only_test == True:
        test(config, model, best_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
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

# =======================================================================================================
def train_val(config, model, train_loader, val_loader, criterion):
    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.train.optimizer.adamw.lr),
            weight_decay=float(config.train.optimizer.adamw.weight_decay)
        )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    # max_iou = 0 # use for record best model
    max_score = -float('inf')
    w_dice = 0.5  # Weight cho Dice score 
    w_hd = 0.5    # Weight cho HD score
    
    # create training data loading iteration
    torch.save(model.state_dict(), best_model_dir)

    for epoch in range(epochs):
        start = time.time()
        
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model.train()
        dice_train_sum = 0
        iou_train_sum = 0
        loss_train_sum = 0
        acc_train_sum = 0
        se_train_sum = 0
        sp_train_sum = 0
        num_train = 0
        iter = 0
        warmup_epochs = 20  
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        
        for idx, (batch, batch_w_s) in enumerate(source_dataset):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            weak_batch = batch_w_s['img_w'].cuda().float()
            strong_batch = batch_w_s['img_s'].cuda().float()

            sup_batch_len = img.shape[0]
            unsup_batch_len = weak_batch.shape[0]
            
            output = model(img)
            output = torch.softmax(output, dim=1)
            
            # calculate loss
            assert (output.shape == label.shape)
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            
            # FixMatch
            #======================================================================================================
            if epoch < warmup_epochs:
                # Only supervised loss during warmup phase
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                # Use weighted sum of losses
                loss = sum(w * l for w, l in zip(loss_weights, losses))
            else:
                # Post-warmup phase, apply FixMatch
                outputs_weak = model(weak_batch)
                outputs_weak = torch.softmax(outputs_weak, dim=1)
                outputs_strong = model(strong_batch)
                outputs_strong = torch.softmax(outputs_strong, dim=1)

                # Generate pseudo-labels using confidence threshold
                max_probs, pseudo_labels = torch.max(outputs_weak, dim=1)
                mask = max_probs.ge(config.semi.conf_thresh).float()
                
                # Convert pseudo-labels to one-hot
                pseudo_labels_one_hot = F.one_hot(pseudo_labels, num_classes=3).permute(0, 3, 1, 2).float()
                
                # Apply mask
                pseudo_labels_masked = pseudo_labels_one_hot * mask.unsqueeze(1)
                
                # Calculate supervised losses
                sup_losses = []
                for function in criterion:
                    sup_losses.append(function(output, label))
                sup_loss = sum(w * l for w, l in zip(loss_weights, sup_losses))
                
                # Calculate unsupervised losses
                unsup_losses = []
                for function in criterion:
                    unsup_losses.append(function(outputs_strong, pseudo_labels_masked))
                unsup_loss = sum(w * l for w, l in zip(loss_weights, unsup_losses))
                
                # Combine losses with consistency weight
                consistency_weight = get_current_consistency_weight(iter // 150)
                loss = sup_loss + unsup_loss * consistency_weight * (sup_batch_len / unsup_batch_len)
            #======================================================================================================
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train_sum += loss.item() * sup_batch_len
            
            # calculate metrics
            with torch.no_grad():
                pred = output.argmax(dim=1).cpu().numpy()
                label_np = label.argmax(dim=1).cpu().numpy()
                
                # Calculate per-class metrics
                dice_scores = []
                iou_scores = []
                for class_idx in range(3):
                    pred_class = (pred == class_idx)
                    label_class = (label_np == class_idx)
                    
                    # Skip background class (index 0)
                    if class_idx > 0:
                        dice_scores.append(calc_dice(pred_class, label_class))
                        iou_scores.append(calc_iou(pred_class, label_class))
                
                # Average only over non-background classes
                dice_train = np.mean(dice_scores) if dice_scores else 0
                iou_train = np.mean(iou_scores) if iou_scores else 0
                
                dice_train_sum += dice_train * sup_batch_len
                iou_train_sum += iou_train * sup_batch_len

            if epoch < warmup_epochs:
                file_log.write('Epoch {}, iter {}, Weighted Loss: {}, Components: BCE: {}, Dice: {}, HD: {}\n'.format(
                    epoch, iter + 1, 
                    round(loss.item(), 5),
                    round(losses[0].item(), 5),
                    round(losses[1].item(), 5),
                    round(losses[2].item(), 5)
                ))
            else:
                file_log.write('Epoch {}, iter {}, Total Loss: {}, Sup Loss: {}, Unsup Loss: {}\n'.format(
                    epoch, iter + 1,
                    round(loss.item(), 5),
                    round(sup_loss.item(), 5),
                    round(unsup_loss.item(), 5)
                ))
                
                # print('Epoch {}, iter {}, Dice Sup Loss: {}, Dice Unsup Loss: {}, BCE Sup Loss: {}, BCE UnSup Loss: {}'.format(
                #     epoch, iter + 1, round(losses[1].item(), 5), round(losses[3].item(), 5), round(losses[0].item(), 5), round(losses[2].item(), 5)
                # ))

            num_train += sup_batch_len
            iter += 1
            
            # end one test batch
            if config.debug: break

                

        # # print
        # file_log.write('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}\n'.format(epoch, 
        #                                                                                               iter, 
        #                                                                                               round(loss_train_sum / num_train,5), 
        #                                                                                               round(dice_train_sum/num_train,4), 
        #                                                                                               round(iou_train_sum/num_train,4)))
        # file_log.flush()
        # print('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}'.format(epoch, 
        #                                                                                               iter, 
        #                                                                                               round(loss_train_sum / num_train,5), 
        #                                                                                               round(dice_train_sum/num_train,4), 
        #                                                                                               round(iou_train_sum/num_train,4)))
        file_log.write('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}\n'.format(
            epoch, 
            iter, 
            round(loss_train_sum / num_train, 5), 
            round(dice_train_sum / num_train, 4), 
            round(iou_train_sum / num_train, 4)
        ))
        file_log.flush()
        print('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}'.format(
            epoch, 
            iter, 
            round(loss_train_sum / num_train, 5), 
            round(dice_train_sum / num_train, 4), 
            round(iou_train_sum / num_train, 4)
        ))  

        # -----------------------------------------------------------------
        # validate
        # ----------------------------------------------------------------
        model.eval()
        
        dice_val_sum = 0
        iou_val_sum = 0
        hd_val_sum = 0  
        loss_val_sum = 0
        acc_val_sum = 0
        se_val_sum = 0
        sp_val_sum = 0
        num_val = 0

        for batch in val_loader:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]

            with torch.no_grad():
                output = model(img)
                output = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).cpu().numpy()
                label_np = label.argmax(dim=1).cpu().numpy()
                
                # Calculate per-class metrics
                dice_scores = []
                iou_scores = []
                for class_idx in range(3):
                    pred_class = (pred == class_idx)
                    label_class = (label_np == class_idx)
                    
                    # Skip background class (index 0)
                    if class_idx > 0:
                        dice_scores.append(calc_dice(pred_class, label_class))
                        iou_scores.append(calc_iou(pred_class, label_class))
                
                # Average only over non-background classes
                dice_val = np.mean(dice_scores) if dice_scores else 0
                iou_val = np.mean(iou_scores) if iou_scores else 0
                
                dice_val_sum += dice_val * batch_len
                iou_val_sum += iou_val * batch_len

                # Calculate HD
                hd = calc_hd(pred, label_np)
                hd_val_sum += hd * batch_len

                # Calculate loss
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss_val_sum += sum(losses) * batch_len / len(criterion)

                # Calculate other metrics
                acc, se, sp = calc_metrics(output, label)
                acc_val_sum += acc * batch_len
                se_val_sum += se * batch_len
                sp_val_sum += sp * batch_len

                num_val += batch_len
                if config.debug: break

        # Calculate epoch metrics
        loss_val_epoch = loss_val_sum/num_val
        dice_val_epoch = dice_val_sum/num_val
        iou_val_epoch = iou_val_sum/num_val
        hd_val_epoch = hd_val_sum/num_val

        # Save results to file with all metrics
        with open(test_results_dir, 'w') as f:
            f.write(f'Loss: {round(loss_val_epoch.item(),4)}, '
                    f'Dice: {round(dice_val_epoch,4)}, '
                    f'IOU: {round(iou_val_epoch,4)}, '
                    f'HD: {round(hd_val_epoch,4)}')

        # Print and log results
        print('========================================================================================')
        print(f'Test || Loss: {round(loss_val_epoch.item(),4)}, '
              f'Dice: {round(dice_val_epoch,4)}, '
              f'IOU: {round(iou_val_epoch,4)}, '
              f'HD: {round(hd_val_epoch,4)}')
        
        file_log.write('========================================================================================\n')
        file_log.write(f'Test || Loss: {round(loss_val_epoch.item(),4)}, '
                       f'Dice: {round(dice_val_epoch,4)}, '
                       f'IOU: {round(iou_val_epoch,4)}, '
                       f'HD: {round(hd_val_epoch,4)}\n')
        file_log.flush()

        # scheduler step, record lr
        scheduler.step()

        # Calculate epoch metrics
        dice_val_epoch = dice_val_sum/num_val
        hd_val_epoch = hd_val_sum/num_val
        
        # Calculate combined score
        hd_norm = np.exp(-hd_val_epoch/100)
        combined_score = w_dice * dice_val_epoch + w_hd * hd_norm
        
        # Save model if combined score improves
        if combined_score > max_score:
            max_score = combined_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)
            
            message = (f'New best epoch {epoch}\n'
                      f'Combined score improved to {combined_score:.4f}\n'
                      f'Dice: {dice_val_epoch:.4f}, HD: {hd_val_epoch:.4f}')
            
            file_log.write(message + '\n')
            file_log.flush()
            print(message)
        
        end = time.time()
        time_elapsed = end-start
        file_log.write('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
        file_log.flush()
        print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))

        # end one epoch
        if config.debug: return
        # if acc_val_epoch > max_acc and se_val_epoch > max_se:
        #     torch.save(model.state_dict(), best_model_dir)
        #     max_acc = acc_val_epoch
        #     max_se = se_val_epoch
        #     best_epoch = epoch
        #     file_log.write('New best epoch {} based on ACC and SE!===============================\n'.format(epoch))
        #     file_log.flush()
        #     print('New best epoch {} based on ACC and SE!==============================='.format(epoch))
        
        # end = time.time()
        # time_elapsed = end-start
        # file_log.write('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
        #             format(epoch, time_elapsed // 60, time_elapsed % 60))
        # file_log.flush()
        # print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s'.
        #     format(epoch, time_elapsed // 60, time_elapsed % 60))

        # # end one epoch
        # if config.debug: return
    file_log.write('Complete training ---------------------------------------------------- \n The best epoch is {}\n'.format(best_epoch))
    file_log.flush()
    print('Complete training ---------------------------------------------------- \n The best epoch is {}'.format(best_epoch))

    return 


# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_sum = 0
    iou_test_sum = 0
    loss_test_sum = 0
    acc_test_sum = 0
    se_test_sum = 0
    sp_test_sum = 0
    hd_test_sum = 0
    num_test = 0

    test_loop = tqdm(test_loader, desc='Testing', leave=True)
    for batch in test_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        batch_len = img.shape[0]
            
        with torch.no_grad():
            output = model(img)
            output = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            # Tính loss
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            loss_test_sum += sum(losses) * batch_len
            
            # Tính metrics
            pred = pred.cpu().numpy()
            label = label.argmax(dim=1).cpu().numpy()
            
            # Calculate per-class metrics
            dice_scores = []
            iou_scores = []
            for class_idx in range(3):
                pred_class = (pred == class_idx)
                label_class = (label == class_idx)
                
                # Skip background class (index 0)
                if class_idx > 0:  # Skip background
                    dice_scores.append(calc_dice(pred_class, label_class))
                    iou_scores.append(calc_iou(pred_class, label_class))
            
            dice_test = np.mean(dice_scores) if dice_scores else 0
            iou_test = np.mean(iou_scores) if iou_scores else 0
            
            dice_test_sum += dice_test * batch_len
            iou_test_sum += iou_test * batch_len

            # Calculate HD
            hd = calc_hd(pred, label)
            hd_test_sum += hd * batch_len
            
            # Calculate ACC, SE, SP
            acc, se, sp = calc_metrics(output, label)
            acc_test_sum += acc * batch_len
            se_test_sum += se * batch_len
            sp_test_sum += sp * batch_len
            
            # Update progress bar
            test_loop.set_postfix({
                'Loss': f'{(sum(losses)).item():.4f}',
                'Dice': f'{dice_test:.4f}',
                'HD': f'{hd:.4f}'
            })

            num_test += batch_len
            if config.debug: break

    # Calculate average metrics
    loss_test_epoch = loss_test_sum/num_test
    dice_test_epoch = dice_test_sum/num_test
    iou_test_epoch = iou_test_sum/num_test
    acc_test_epoch = acc_test_sum/num_test
    se_test_epoch = se_test_sum/num_test
    sp_test_epoch = sp_test_sum/num_test
    hd_test_epoch = hd_test_sum/num_test

    # Save results to file with all metrics
    with open(test_results_dir, 'w') as f:
        f.write(f'Loss: {round(loss_test_epoch.item(),4)}, '
                f'Dice: {round(dice_test_epoch,4)}, '
                f'IOU: {round(iou_test_epoch,4)}, '
                f'HD: {round(hd_test_epoch,4)}')

    # Print and log results with all metrics
    print('========================================================================================')
    print(f'Test || Loss: {round(loss_test_epoch.item(),4)}, '
          f'Dice: {round(dice_test_epoch,4)}, '
          f'IOU: {round(iou_test_epoch,4)}, '
          f'HD: {round(hd_test_epoch,4)}')
    
    file_log.write('========================================================================================\n')
    file_log.write(f'Test || Loss: {round(loss_test_epoch.item(),4)}, '
                   f'Dice: {round(dice_test_epoch,4)}, '
                   f'IOU: {round(iou_test_epoch,4)}, '
                   f'HD: {round(hd_test_epoch,4)}\n')
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
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
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
    
    folds_to_train = [5]
    
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