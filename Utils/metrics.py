from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from hausdorff import hausdorff_distance
from scipy.ndimage import binary_erosion

def calc_metrics(output, label):
    """
    Calculate metrics for multi-class segmentation
    
    Args:
        output: Predicted segmentation mask [B, C, H, W] - PyTorch tensor or numpy array
        label: Ground truth segmentation mask [B, C, H, W] - PyTorch tensor or numpy array
        
    Returns:
        acc: Average accuracy across all classes
        se: Average sensitivity across all classes
        sp: Average specificity across all classes
    """
    # Convert to numpy if input is PyTorch tensor
    if torch.is_tensor(output):
        output = output.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()
    
    # Ensure output and label have correct shape
    if len(output.shape) == 3:  # If [B, H, W]
        output = output[:, None, :, :]  # Add channel dimension
    if len(label.shape) == 3:  # If [B, H, W]
        label = label[:, None, :, :]  # Add channel dimension
        
    # Get predictions as class indices
    if output.shape[1] > 1:  # If output has multiple channels
        output = np.argmax(output, axis=1)  # [B, H, W]
    else:
        output = output.squeeze(1)  # Remove channel dim if single channel
        
    if label.shape[1] > 1:  # If label has multiple channels
        label = np.argmax(label, axis=1)  # [B, H, W]
    else:
        label = label.squeeze(1)  # Remove channel dim if single channel
    
    num_classes = 3  # background + 2 objects
    
    # Initialize metrics for each class
    acc_per_class = []
    se_per_class = []
    sp_per_class = []
    
    for class_idx in range(num_classes):
        # Convert to binary problem for each class
        output_binary = (output == class_idx)  # [B, H, W]
        label_binary = (label == class_idx)    # [B, H, W]
        
        # Ensure shapes match
        assert output_binary.shape == label_binary.shape, \
            f"Shape mismatch: output {output_binary.shape} vs label {label_binary.shape}"
        
        TP = ((output_binary == 1) & (label_binary == 1)).sum()
        TN = ((output_binary == 0) & (label_binary == 0)).sum()
        FP = ((output_binary == 1) & (label_binary == 0)).sum()
        FN = ((output_binary == 0) & (label_binary == 1)).sum()
        
        # Accuracy for this class
        if (TP + TN + FP + FN) > 0:
            acc = (TP + TN) / (TP + TN + FP + FN)
        else:
            acc = 0
        
        # Sensitivity for this class
        if (TP + FN) > 0:
            se = TP / (TP + FN)
        else:
            se = 0
            
        # Specificity for this class
        if (TN + FP) > 0:
            sp = TN / (TN + FP)
        else:
            sp = 0
            
        acc_per_class.append(acc)
        se_per_class.append(se)
        sp_per_class.append(sp)
    
    # Return average metrics across all classes
    return np.mean(acc_per_class), np.mean(se_per_class), np.mean(sp_per_class)

def calc_auc(output, label):
    """
    Calculate AUC for multi-class segmentation using one-vs-rest approach
    
    Args:
        output: Model predictions [B, C, H, W] - PyTorch tensor or numpy array
        label: Ground truth segmentation mask [B, C, H, W] or [B, H, W] - PyTorch tensor or numpy array
        
    Returns:
        Average AUC across all classes
    """
    # Convert to numpy if input is PyTorch tensor
    if torch.is_tensor(output):
        output = output.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()
    
    # Ensure label has correct shape
    if len(label.shape) == 3:  # If [B, H, W]
        label = label[:, None, :, :]  # Add channel dimension
    
    # If label is one-hot encoded, convert to class indices
    if label.shape[1] > 1:
        label = np.argmax(label, axis=1)
    else:
        label = label.squeeze(1)
        
    num_classes = output.shape[1]
    auc_scores = []
    
    # Calculate AUC for each class using one-vs-rest approach
    for class_idx in range(num_classes):
        # Get probabilities for current class
        class_probs = output[:, class_idx, ...].reshape(-1)
        # Convert to binary labels for current class
        class_labels = (label.reshape(-1) == class_idx).astype(int)
        
        try:
            auc = roc_auc_score(class_labels, class_probs)
            auc_scores.append(auc)
        except ValueError:
            # Skip if class is not present in ground truth
            continue
    
    # Return average AUC if we have any valid scores
    if auc_scores:
        return np.mean(auc_scores)
    else:
        return 0.0

def calc_hd(pred, target):
    """
    Calculate Hausdorff Distance for multi-class segmentation using distance transform
    
    Args:
        pred: Predicted segmentation mask [B, H, W] - numpy array
        target: Ground truth segmentation mask [B, H, W] - numpy array
        
    Returns:
        Average modified Hausdorff Distance across all classes
    """
    num_classes = 3
    hd_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx)
        target_class = (target == class_idx)
        
        if np.any(pred_class) and np.any(target_class):
            # Calculate distance transforms
            pred_dist = distance_transform_edt(~pred_class)
            target_dist = distance_transform_edt(~target_class)

            # Get surface points
            pred_surface = pred_class & ~binary_erosion(pred_class)
            target_surface = target_class & ~binary_erosion(target_class)

            # Calculate HD
            pred_to_target = np.max(pred_dist * target_surface)
            target_to_pred = np.max(target_dist * pred_surface)
            
            hd = max(pred_to_target, target_to_pred)
            hd_scores.append(hd)
            
    return np.mean(hd_scores) if hd_scores else float('inf')

def calc_dice(pred, target):
    """
    Calculate Dice coefficient for multi-class segmentation
    
    Args:
        pred: Predicted segmentation mask [B, H, W] - numpy array 
        target: Ground truth mask [B, H, W] - numpy array
        
    Returns:
        Average Dice coefficient across all classes
    """
    num_classes = 3
    dice_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx)
        target_class = (target == class_idx)
        
        # Calculate intersection and sums
        intersection = np.logical_and(pred_class, target_class).sum()
        pred_sum = pred_class.sum()
        target_sum = target_class.sum()
        
        # Calculate Dice
        if pred_sum + target_sum == 0:
            # If both prediction and target are empty, consider it perfect match
            dice = 1.0
        else:
            dice = (2.0 * intersection) / (pred_sum + target_sum)
            
        dice_scores.append(dice)
    
    return np.mean(dice_scores)

def calc_iou(pred, target):
    """
    Calculate IoU (Jaccard) coefficient for multi-class segmentation
    
    Args:
        pred: Predicted segmentation mask [B, H, W] - numpy array 
        target: Ground truth mask [B, H, W] - numpy array
        
    Returns:
        Average IoU coefficient across all classes
    """
    num_classes = 3
    iou_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx)
        target_class = (target == class_idx)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()
        
        # Calculate IoU
        if union == 0:
            # If both prediction and target are empty, consider it perfect match
            iou = 1.0
        else:
            iou = intersection / union
            
        iou_scores.append(iou)
    
    return np.mean(iou_scores)
