import torch
import numpy as np
from transformers import SamModel, SamProcessor
import torch.nn.functional as F

class SAMFeatureExtractor:
    def __init__(self, model_name="facebook/sam-vit-huge", device=None, target_size=(224, 224)):
        """
        Initialize SAM feature extractor using HuggingFace transformers
        Args:
            model_name: HuggingFace model name
            device: Device to run the model on
            target_size: Target size for output masks (height, width)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model.eval()
        self.target_size = target_size
        
    def get_image_embedding(self, image):
        """
        Get SAM image embedding for guidance
        Args:
            image: Image tensor of shape (B, C, H, W)
        Returns:
            mask_logits: Predicted mask logits from SAM
        """
        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        batch_masks = []
        for img in image:
            # Transpose from CHW to HWC if needed    
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
                
            # Normalize image to [0,1] range
            img = (img - img.min()) / (img.max() - img.min())
            
            # Convert to uint8 [0,255]
            img = (img * 255).astype(np.uint8)
                
            # Process image
            inputs = self.processor(img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate masks
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get predicted masks
            masks = outputs.pred_masks.squeeze(1)  # Remove batch dimension
            
            # Take first mask or create empty mask if none predicted
            if masks.size(0) > 0:
                best_mask = masks[0].float()  # Take first mask
            else:
                best_mask = torch.zeros((img.shape[0], img.shape[1]), device=self.device)
            
            # Resize mask to target size
            best_mask = best_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            best_mask = F.interpolate(best_mask, size=self.target_size, mode='bilinear', align_corners=False)
            best_mask = best_mask.squeeze(0)  # Remove batch dim
                
            batch_masks.append(best_mask)
            
        return torch.stack(batch_masks)  # Shape: (B, 1, H, W)

def get_sam_guidance(images, sam_extractor):
    """
    Helper function to get SAM guidance masks for a batch of images
    """
    with torch.no_grad():
        sam_masks = sam_extractor.get_image_embedding(images)
    return sam_masks 