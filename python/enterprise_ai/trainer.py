from typing import Dict, Any, Optional, List
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import copy
from transformers import get_cosine_schedule_with_warmup
from .model import EnterpriseAIModel
from .losses import AdvancedLossFunctions
import torch.nn as nn
from torch.optim import AdamW
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """Dataset class for handling text and labels."""
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'labels': self.labels[idx]
        }

class EnhancedTrainer:
    def __init__(self, config: Dict[str, Any]):
        # Fixed configuration parameters
        self.config = {
            "model_type": config.get("model_type", "document_processor"),
            "learning_rate": config.get("learning_rate", 0.001),
            "weight_decay": 0.01,
            "batch_size": config.get("batch_size", 16),
            "num_classes": config.get("num_document_classes", 3),
            "num_entity_types": config.get("num_entity_types", 5),
            "num_semantic_classes": config.get("num_semantic_classes", 3),
            "hidden_size": 1024,
            "dropout_rate": 0.1,
            "num_patterns": 100,
            "num_epochs": config.get("num_epochs", 10),
            "early_stopping_patience": config.get("early_stopping_patience", 5),
            "early_stopping_delta": config.get("early_stopping_delta", 1e-4)
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Initialize the model with proper config
            self.model = EnterpriseAIModel(
                model_type=self.config['model_type'],
                config=self.config
            ).to(self.device)
            
            # Initialize component-specific optimizers
            self.optimizers = {
                'base': AdamW(
                    self.model.base_model.parameters(),
                    lr=2e-5,
                    weight_decay=0.01
                ),
                'document': AdamW(
                    self.model.document_analyzer.parameters(),
                    lr=1e-4
                ),
                'entity': AdamW(
                    self.model.entity_recognizer.parameters(),
                    lr=1e-4
                ),
                'semantic': AdamW(
                    self.model.semantic_analyzer.parameters(),
                    lr=1e-4
                )
            }
            
            # Initialize schedulers for each optimizer
            self.schedulers = {
                name: get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=1000,
                    num_training_steps=50000,
                    num_cycles=2
                ) for name, optimizer in self.optimizers.items()
            }
            
            # Initialize loss functions with correct number of classes
            self.loss_fns = {
                'classification': nn.CrossEntropyLoss(
                    weight=torch.ones(self.config['num_classes']).to(self.device),
                    label_smoothing=0.1
                ),
                'entity': nn.BCEWithLogitsLoss(),
                'semantic': nn.KLDivLoss(reduction='batchmean')
            }
            
            # Initialize mixed precision training
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            # Initialize early stopping
            self.early_stopping = EarlyStopping(
                patience=config.get('early_stopping_patience', 5),
                delta=config.get('early_stopping_delta', 1e-4)
            )
            
            # Initialize best model tracking
            self.best_model = None
            self.best_loss = float('inf')
            
            # Initialize experiment tracking
            self._initialize_experiment_tracking()
            
        except Exception as e:
            logger.error(f"Trainer initialization error: {str(e)}")
            raise

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
    
    def _prepare_dataset(self, texts: List[str], labels: List[int]) -> SimpleDataset:
        """Prepare dataset from texts and labels"""
        return SimpleDataset(texts=texts, labels=labels)

    def train(self, training_data: Dict[str, Any], validation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the model with the given data
        Args:
            training_data: Dictionary containing texts and labels
            validation_data: Optional dictionary containing validation data
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            # Prepare datasets
            train_dataset = self._prepare_dataset(
                texts=training_data['texts'],
                labels=training_data['labels']
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'],
                shuffle=True
            )

            # Initialize metrics tracking
            metrics = {
                'train_loss': [],
                'learning_rates': {}
            }

            # Training loop
            for epoch in range(self.config['num_epochs']):
                self.model.train()
                total_loss = 0
                
                # Use tqdm for progress bar
                with tqdm(train_loader, desc=f'Epoch {epoch + 1}') as pbar:
                    for batch in pbar:
                        # Move batch to device
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = self.model(batch)
                        loss = self._calculate_loss(outputs, batch)
                        
                        # Backward pass
                        for optimizer in self.optimizers.values():
                            optimizer.zero_grad()
                        
                        loss.backward()
                        
                        for optimizer in self.optimizers.values():
                            optimizer.step()
                        
                        total_loss += loss.item()
                        pbar.set_postfix({'loss': loss.item()})
                        
                        # Log batch metrics
                        self._log_batch_metrics(loss.item(), outputs, batch)
                
                # Calculate average loss
                avg_loss = total_loss / len(train_loader)
                
                # Update training state
                self._update_training_state(avg_loss, metrics, epoch)
                
                # Early stopping check
                if self.early_stopping(avg_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Prepare and return training results
            return self._prepare_training_results(metrics, epoch)
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted combination of all losses"""
        total_loss = 0
        
        if 'document_class' in outputs:
            # Extract logits from the document class output dictionary
            doc_logits = outputs['document_class'].get('logits', outputs['document_class'])
            if isinstance(doc_logits, dict):
                doc_logits = doc_logits.get('logits', torch.zeros(1, self.config['num_classes']).to(self.device))
            total_loss += self.loss_fns['classification'](doc_logits, batch['labels'])
        
        if 'entities' in outputs:
            entity_outputs = outputs['entities']
            if isinstance(entity_outputs, dict):
                entity_outputs = entity_outputs.get('logits', torch.zeros(1, self.config['num_entity_types']).to(self.device))
            if 'entity_labels' in batch:
                total_loss += 0.3 * self.loss_fns['entity'](entity_outputs, batch['entity_labels'])
        
        if 'semantic_features' in outputs:
            semantic_outputs = outputs['semantic_features']
            if isinstance(semantic_outputs, dict):
                semantic_outputs = semantic_outputs.get('logits', torch.zeros(1, self.config['num_semantic_classes']).to(self.device))
            if 'semantic_labels' in batch:
                total_loss += 0.2 * self.loss_fns['semantic'](
                    F.log_softmax(semantic_outputs, dim=-1),
                    batch['semantic_labels']
                )
        
        return total_loss

    def _update_training_state(self, avg_loss: float, metrics: Dict[str, Any], epoch: int):
        """Update training state with current metrics and learning rates"""
        try:
            # Update metrics
            metrics['train_loss'].append(avg_loss)
            
            # Update learning rates for each optimizer
            for name, optimizer in self.optimizers.items():
                current_lr = optimizer.param_groups[0]['lr']
                if 'learning_rates' not in metrics:
                    metrics['learning_rates'] = {}
                if name not in metrics['learning_rates']:
                    metrics['learning_rates'][name] = []
                metrics['learning_rates'][name].append(current_lr)
            
            # Step schedulers
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'avg_loss': avg_loss,
                **{f'lr_{name}': metrics['learning_rates'][name][-1] 
                   for name in self.optimizers.keys()}
            })
            
            # Update best model if needed
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                
        except Exception as e:
            logger.error(f"Error updating training state: {str(e)}")
            # Continue training even if logging fails
            pass

    def _prepare_training_results(self, metrics: Dict[str, Any], final_epoch: int) -> Dict[str, Any]:
        """Prepare final training results and metrics"""
        try:
            return {
                'success': True,
                'final_loss': metrics['train_loss'][-1],
                'best_loss': self.best_loss,
                'epochs_completed': final_epoch + 1,
                'learning_rates': metrics['learning_rates'],
                'loss_history': metrics['train_loss'],
                'training_time': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error preparing training results: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _log_batch_metrics(self, loss: float, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Log metrics for the current batch"""
        try:
            wandb.log({
                'batch_loss': loss,
                'document_loss': self.loss_fns['classification'](
                    outputs.get('document_class', torch.tensor(0)), 
                    batch['labels']
                ).item() if 'document_class' in outputs else 0,
                'entity_loss': self.loss_fns['entity'](
                    outputs.get('entities', torch.tensor(0)), 
                    batch.get('entity_labels', torch.tensor(0))
                ).item() if 'entities' in outputs else 0,
                'semantic_loss': self.loss_fns['semantic'](
                    F.log_softmax(outputs.get('semantic_features', torch.tensor(0)), dim=-1),
                    batch.get('semantic_labels', torch.tensor(0))
                ).item() if 'semantic_features' in outputs else 0
            })
        except Exception as e:
            logger.warning(f"Error logging batch metrics: {str(e)}")
            # Continue training even if logging fails
            pass

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