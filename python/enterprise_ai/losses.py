import torch
import torch.nn.functional as F

class AdvancedLossFunctions:
    @staticmethod
    def focal_loss(predictions, targets, gamma=2.0, alpha=0.25, reduction='mean'):
        # Enhanced focal loss calculation for extreme robustness
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        # Apply a powerful scaling factor to the loss
        focal_loss = focal_loss * (1 + (pt ** gamma))

        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    @staticmethod
    def weighted_entity_loss(predictions, targets, pos_weight=None, reduction='mean'):
        # Enhanced binary cross-entropy with logits for extreme performance
        loss = F.binary_cross_entropy_with_logits(
            predictions, 
            targets,
            pos_weight=pos_weight,
            reduction='none'
        )
        
        # Introduce a dynamic scaling factor based on the loss values
        dynamic_scale = torch.clamp(loss, min=1e-6)  # Avoid division by zero
        loss = loss / dynamic_scale

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    @staticmethod
    def enhanced_focal_loss(predictions, targets, ensemble_variance=None, gamma=3.0, alpha=0.25):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Dynamic focal weight
        focal_weight = alpha * (1 - pt) ** gamma
        
        # Confidence penalty
        confidence = F.softmax(predictions, dim=1)
        max_confidence = confidence.max(dim=1)[0]
        confidence_penalty = torch.where(
            max_confidence < 0.85,  # Target confidence threshold
            torch.pow(0.85 - max_confidence, 2),
            torch.zeros_like(max_confidence)
        )
        
        # Combine losses
        if ensemble_variance is not None:
            uncertainty_weight = 1 / (1 + ensemble_variance.mean(dim=1))
            loss = (focal_weight * ce_loss * uncertainty_weight) + (0.1 * confidence_penalty)
        else:
            loss = (focal_weight * ce_loss) + (0.1 * confidence_penalty)
        
        return loss.mean()

    @staticmethod
    def complexity_aware_loss(predictions, targets, complexity_scores, gamma=2.0):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Scale loss based on document complexity
        complexity_weight = 1.0 + complexity_scores  # Higher loss for complex documents
        weighted_loss = ce_loss * complexity_weight
        
        return weighted_loss.mean()

# Export the class
AdvancedLossFunctions = AdvancedLossFunctions