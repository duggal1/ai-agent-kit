import torch
from typing import Dict, Any, Optional, List
import numpy as np
from .model import EnterpriseAIModel
import logging
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced configuration validation
        self._validate_config()
        
        # Advanced model initialization with more sophisticated architecture
        self.model = self._create_advanced_model()
        
        # Multi-objective optimizer with adaptive techniques
        self.optimizer = self._create_adaptive_optimizer()
        
        # Advanced learning rate scheduling with multiple strategies
        self.scheduler = self._create_sophisticated_scheduler()
        
        # Enhanced precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Advanced regularization techniques
        self.best_model = None
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 5),
            delta=config.get('early_stopping_delta', 1e-4)
        )
        
        # Experiment tracking
        self._initialize_experiment_tracking()
    
    def _validate_config(self):
        """Enhanced configuration validation with more comprehensive checks."""
        required_params = [
            'model_type', 'learning_rate', 'weight_decay', 'batch_size',
            'num_document_classes', 'num_entity_types', 'time_series_features',
            'forecast_horizon', 'dropout_rate', 'embedding_dim'
        ]
        
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing critical config parameter: {param}")
        
        # Additional sophisticated validation
        assert self.config['num_document_classes'] > 0, "Invalid document classes"
        assert self.config['learning_rate'] > 0, "Learning rate must be positive"
        assert 0 <= self.config.get('dropout_rate', 0) < 1, "Dropout rate must be between 0 and 1"
    
    def _create_advanced_model(self) -> EnterpriseAIModel:
        """Create a more sophisticated model with advanced architectural considerations."""
        model = EnterpriseAIModel(
            model_type=self.config['model_type'],
            config={
                **self.config,
                # Advanced model configuration
                'use_layer_normalization': True,
                'use_residual_connections': True,
                'attention_heads': self.config.get('attention_heads', 8),
                'advanced_regularization': True
            }
        ).to(self.device)
        
        return model
    
    def _create_adaptive_optimizer(self):
        """Create an adaptive multi-objective optimizer."""
        optimizer = torch.optim.AdamW(
            [
                {'params': self.model.base_layers.parameters(), 'weight_decay': self.config['weight_decay']},
                {'params': self.model.task_specific_layers.parameters(), 'weight_decay': self.config['weight_decay'] * 0.5},
            ],
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return optimizer
    
    def _create_sophisticated_scheduler(self):
        """Advanced learning rate scheduling with multiple strategies."""
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # Initial restart period
            T_mult=2,  # Multiplicative factor for restart periods
            eta_min=1e-6  # Minimum learning rate
        )
        return scheduler
    
    def _initialize_experiment_tracking(self):
        """Enhanced experiment tracking with more comprehensive logging."""
        try:
            wandb.init(
                project="enterprise-ai-advanced",
                config=self.config,
                tags=['multi-task', 'advanced-training'],
                notes="Sophisticated multi-objective model training"
            )
        except Exception as e:
            logger.warning(f"Experiment tracking initialization failed: {e}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        """Advanced training method with multiple sophisticated techniques."""
        logger.info(f"Advanced training on {self.device}")
        
        for epoch in range(epochs):
            # Train phase with advanced techniques
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase with comprehensive evaluation
            val_metrics = self._validate_epoch(val_loader)
            
            # Combine metrics for holistic assessment
            combined_metrics = self._combine_metrics(train_metrics, val_metrics)
            
            # Logging and tracking
            self._log_epoch_metrics(epoch, combined_metrics)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping mechanism
            if self.early_stopping(val_metrics['loss']):
                logger.info("Early stopping triggered")
                break
        
        # Restore best model
        if self.best_model:
            self.model.load_state_dict(self.best_model)
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Sophisticated training epoch with advanced techniques."""
        self.model.train()
        total_loss = 0
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Advanced multi-task learning with auxiliary losses
                outputs = self.model(batch)
                primary_loss = self.model._compute_loss(outputs, batch)
                auxiliary_loss = self._compute_auxiliary_losses(outputs, batch)
                
                loss = primary_loss + 0.3 * auxiliary_loss
                loss = loss / gradient_accumulation_steps
            
            # Advanced gradient handling
            if self.scaler:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        return {'loss': total_loss / len(train_loader)}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Comprehensive validation with advanced metrics."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(batch)
                    loss = self.model._compute_loss(outputs, batch)
                    total_loss += loss.item()
                    
                    # Collect predictions for advanced metrics
                    preds = torch.argmax(outputs['document_class'], dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['document_class'].cpu().numpy())
        
        # Advanced performance metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'loss': total_loss / len(val_loader),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _compute_auxiliary_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute auxiliary losses for multi-task learning."""
        # Example of an auxiliary loss - this could be modified based on specific requirements
        auxiliary_loss = F.kl_div(
            F.log_softmax(outputs['document_class'], dim=1),
            F.softmax(batch.get('document_class_soft_targets', outputs['document_class']), dim=1),
            reduction='batchmean'
        )
        return auxiliary_loss
    
    def _combine_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> Dict[str, float]:
        """Combine training and validation metrics for holistic assessment."""
        combined_metrics = {**train_metrics, **val_metrics}
        combined_metrics['combined_score'] = (
            val_metrics['f1_score'] - 0.5 * train_metrics['loss']
        )
        return combined_metrics
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Sophisticated logging of epoch metrics."""
        logger.info(f"Epoch {epoch + 1} Metrics: {metrics}")
        
        try:
            wandb.log({
                'epoch': epoch,
                **metrics
            })
        except Exception as e:
            logger.warning(f"Metrics logging failed: {e}")

class EarlyStopping:
    """Sophisticated early stopping mechanism."""
    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
            return False
        
        self.counter += 1
        return self.counter >= self.patience