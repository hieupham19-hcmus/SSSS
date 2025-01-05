import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import binary_erosion
from scipy.ndimage import distance_transform_edt, binary_erosion

    

def dice_loss(score, target):
    """
    Compute dice loss for multi-class segmentation
    
    Args:
        score: Model predictions after sigmoid/softmax, shape [B, C, H, W] 
        target: One-hot encoded ground truth, shape [B, C, H, W]
        
    Returns:
        Mean dice loss across all classes and batches
    """
    target = target.float()
    smooth = 1e-5
    
    # Calculate dice loss for each class separately
    num_classes = score.shape[1]
    total_loss = 0
    
    for cls in range(num_classes):
        intersect = torch.sum(score[:, cls, ...] * target[:, cls, ...])
        y_sum = torch.sum(target[:, cls, ...] * target[:, cls, ...]) 
        z_sum = torch.sum(score[:, cls, ...] * score[:, cls, ...])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        total_loss += (1 - loss)
        
    # Average over number of classes
    mean_loss = total_loss / num_classes
    
    return mean_loss

def dice_loss1(score, target):
    """
    Compute non-squared dice loss for multi-class segmentation
    
    Args:
        score: Model predictions after sigmoid/softmax, shape [B, C, H, W]
        target: One-hot encoded ground truth, shape [B, C, H, W]
        
    Returns:
        Mean dice loss across all classes and batches
    """
    target = target.float()
    smooth = 1e-5
    
    # Calculate dice loss for each class separately  
    num_classes = score.shape[1]
    total_loss = 0
    
    for cls in range(num_classes):
        intersect = torch.sum(score[:, cls, ...] * target[:, cls, ...])
        y_sum = torch.sum(target[:, cls, ...])
        z_sum = torch.sum(score[:, cls, ...])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        total_loss += (1 - loss)
        
    # Average over number of classes
    mean_loss = total_loss / num_classes
    
    return mean_loss


def iou_loss(score, target):
    target = target.float()
    smooth = 1e-5
    tp_sum = torch.sum(score * target)
    fp_sum = torch.sum(score * (1 - target))
    fn_sum = torch.sum((1 - score) * target)
    loss = (tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    ## p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(
        np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def compute_sdf01(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)

    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            if np.max(posmask) == 0:
                continue
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.uint8)
            sdf = negdis / np.max(negdis) / 2 - posdis / np.max(
                posdis) / 2 + 0.5
            sdf[boundary > 0] = 0.5
            normalized_sdf[b][c] = sdf
    return normalized_sdf


def compute_sdf1_1(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            if np.max(posmask) == 0:
                continue
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.uint8)
            sdf = negdis / np.max(negdis) - posdis / np.max(posdis)
            sdf[boundary > 0] = 0
            normalized_sdf[b][c] = sdf
    return normalized_sdf

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # Detach tensors before converting to numpy
        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float()

        # Move tensors to same device as input
        pred_dt = pred_dt.to(pred.device)
        target_dt = target_dt.to(target.device)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


def compute_fore_dist(segmentation):
    """
    compute the foreground of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            posdis = distance(posmask)
            normalized_sdf[b][c] = posdis / np.max(posdis)
    return normalized_sdf


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def AAAI_sdf_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        gt_sdm_npy = compute_sdf1_1(y_onehot.cpu().numpy())
        if net_output.device.type == "cuda":
            gt_sdm = torch.from_numpy(gt_sdm_npy).float().cuda(
                net_output.device.index)
        else:
            gt_sdm = torch.from_numpy(gt_sdm_npy).float()
    intersect = sum_tensor(net_output * gt_sdm, axes, keepdim=False)
    pd_sum = sum_tensor(net_output**2, axes, keepdim=False)
    gt_sum = sum_tensor(gt_sdm**2, axes, keepdim=False)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF_AAAI = -L_product.mean() + torch.norm(net_output - gt_sdm,
                                                1) / torch.numel(net_output)

    return L_SDF_AAAI


def sdf_kl_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        # print('y_onehot.shape', y_onehot.shape)
        gt_sdf_npy = compute_sdf(y_onehot.cpu().numpy())
        gt_sdf = torch.from_numpy(gt_sdf_npy + smooth).float().cuda(
            net_output.device.index)
    # print('net_output, gt_sdf', net_output.shape, gt_sdf.shape)
    # exit()
    sdf_kl_loss = F.kl_div(net_output,
                           gt_sdf[:, 1:2, ...],
                           reduction='batchmean')

    return sdf_kl_loss


# don't put the sample itself into the Positive set
class Supervised_Contrastive_Loss(torch.nn.Module):
    '''
    from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    https://blog.csdn.net/wf19971210/article/details/116715880
    Treat samples in the same labels as the positive samples, others as negative samples
    '''
    def __init__(self, temperature=0.1, device='cpu'):
        super(Supervised_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, projections, targets, attribute=None):
        # projections (bs, dim), targets (bs)
        # similarity matrix/T
        # dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
        # print(dot_product_tempered)
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        # exp_dot_tempered = (
        #     torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        # )
        exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-5
        # a matrix, same labels are true, others are false
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        # a matrix, diagonal are zeros, others are ones
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
        mask_nonsimilar_class = ~mask_similar_class
        # mask_nonsimilar_attr = ~mask_similar_attr
        # a matrix, same labels are 1, others are 0, and diagonal are zeros
        mask_combined = mask_similar_class * mask_anchor_out
        # num of similar samples for sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr)
        # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
        if attribute != None:
            mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
       
        else:
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
        supervised_contrastive_loss = torch.sum(log_prob * mask_combined)/(torch.sum(cardinality_per_samples)+1e-5)

        
        return supervised_contrastive_loss


# class Supervised_Contrastive_Loss(torch.nn.Module):
#     '''
#     from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
#     https://blog.csdn.net/wf19971210/article/details/116715880
#     Treat samples in the same labels as the positive samples (including itself), others as negative samples
#     '''
#     def __init__(self, temperature=0.1, device='cpu'):
#         super(Supervised_Contrastive_Loss, self).__init__()
#         self.temperature = temperature
#         self.device = device
    
#     def forward(self, projections, targets, attribute=None):
#         # projections (bs, dim), targets (bs)
#         # similarity matrix/T
#         # dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
#         dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
#         # print(dot_product_tempered)
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         # exp_dot_tempered = (
#         #     torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#         # )
#         exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-6
#         # a matrix, same labels are true, others are false
#         mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
#         # a matrix, diagonal are zeros, others are ones
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
#         mask_nonsimilar_class = ~mask_similar_class
#         # mask_nonsimilar_attr = ~mask_similar_attr
#         # a matrix, same labels are 1, others are 0, and diagonal are zeros
#         mask_combined = mask_similar_class * mask_anchor_out
#         # num of similar samples for sample
#         cardinality_per_samples = torch.sum(mask_similar_class, dim=1)
#         # print(exp_dot_tempered * mask_nonsimilar_class)
#         # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
#         if attribute != None:
#             mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
#             log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-6))
       
#         else:
#             log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-6))
#         supervised_contrastive_loss = torch.sum(log_prob * mask_similar_class)/(torch.sum(cardinality_per_samples)+1e-6)

        
#         return supervised_contrastive_loss


def hausdorff_dt_loss(pred, target):
    """
    Compute Hausdorff distance loss between predicted and target segmentation masks
    using distance transform.
    
    Args:
        pred: Predicted segmentation mask (after sigmoid)
        target: Ground truth segmentation mask
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    
    # Convert tensors to numpy arrays
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    # Compute distance transforms
    pred_dt = distance_transform_edt(pred)
    target_dt = distance_transform_edt(target)
    
    # Compute Hausdorff loss
    pred_dt = torch.from_numpy(pred_dt).cuda().float()
    target_dt = torch.from_numpy(target_dt).cuda().float()
    
    return torch.mean(torch.abs(pred_dt - target_dt))


def hausdorff_loss(pred, target):
    """
    Calculate Hausdorff Distance loss
    Args:
        pred: [B, C, H, W] prediction after softmax
        target: [B, C, H, W] one-hot encoded target
    """

    # Convert to numpy for distance transform
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    B, C, H, W = pred.shape
    total_hd_loss = 0
    
    for b in range(B):
        batch_loss = 0
        # Skip background (start from 1)
        for c in range(1, C):
            pred_mask = pred[b, c] > 0.5
            target_mask = target[b, c] > 0.5
            
            if not (np.any(pred_mask) and np.any(target_mask)):
                continue
                
            # Calculate distance transforms
            pred_dt = distance_transform_edt(~pred_mask)
            target_dt = distance_transform_edt(~target_mask)
            
            # Get surface points
            pred_border = pred_mask ^ binary_erosion(pred_mask)
            target_border = target_mask ^ binary_erosion(target_mask)
            
            # Calculate HD components
            pred_to_target = np.mean(pred_dt[target_border])
            target_to_pred = np.mean(target_dt[pred_border])
            
            batch_loss += max(pred_to_target, target_to_pred)
            
        if C > 1:  # If we found valid classes
            total_hd_loss += batch_loss / (C-1)  # Average over classes
            
    return torch.tensor(total_hd_loss / B, requires_grad=True).cuda()


if __name__ == '__main__':

    # # check supervised contrastive loss
    # loss_func = Supervised_Contrastive_Loss()
    # # a,b = torch.tensor([[0.,0,0,0,1,1,1,1,1,1]]), torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # a,b  = torch.ones((3,7)), torch.ones(3,7)
    # # a,b = a.repeat((3,1)), b.repeat((3,1))
    # # a = torch.tensor([[0.,0,1,1]])
    # # a= a.repeat((6,1))
    # # a = torch.randn(3,10)
    # # b = torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # # x = torch.cat((a,b),dim=0)
    # x = torch.randn(6,10)

    # y = torch.tensor([1,2,3,4,5,6])
    # # z = torch.tensor([2,3,3,2,3,3])
    # loss = loss_func(x, y)
    # print(loss)

    a = torch.tensor([0.0,1.0,0.0,1.0])
    b = torch.tensor([0.0,0.0,0.0,1.0])
    # print(a)
    # print(b)
    dice = dice_per_img(a,b)
    dice_all = dice_loss(a,b)
    print(dice.shape)
    print(dice)
    print(dice_all)