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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    def __init__(self, config: Dict[str, Any]):
        # Fixed configuration parameters
        self.config = {
            "model_type": config.get("model_type", "document_processor"),
            "learning_rate": config.get("learning_rate", 0.001),
            "weight_decay": 0.01,
            "batch_size": 16,
            "num_document_classes": config.get("num_document_classes", 2),
            "num_entity_types": config.get("num_entity_types", 5),
            "num_semantic_classes": config.get("num_semantic_classes", 3),
            "time_series_features": config.get("time_series_features", 3),
            "forecast_horizon": config.get("forecast_horizon", 1),
            "dropout_rate": 0.1,
            "embedding_dim": 1024
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the model
        self.model = EnterpriseAIModel(
            model_type=self.config['model_type'],
            config=self.config
        ).to(self.device)
        
        # Advanced optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=1000,
            num_training_steps=config.get('total_steps', 10000)
        )

        # Initialize losses
        self.losses = AdvancedLossFunctions()
        
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
    
    def train(self, train_data: Dict[str, Any], epochs: int = 10) -> Dict[str, Any]:
        """
        Train the model with the provided data
        """
        logger.info("Starting training process...")
        
        try:
            # Create dataset from training data
            train_dataset = self._prepare_dataset(train_data)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )

            # Training loop
            best_loss = float('inf')
            metrics = {
                'train_loss': [],
                'learning_rates': []
            }

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                    try:
                        # Move batch to device
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Forward pass with gradient computation
                        self.optimizer.zero_grad()
                        
                        # Use automatic mixed precision if available
                        if self.scaler:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(batch)
                                loss = self.model._compute_loss(outputs, batch)
                            
                            # Scale loss and perform backward pass
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            outputs = self.model(batch)
                            loss = self.model._compute_loss(outputs, batch)
                            loss.backward()
                            self.optimizer.step()
                        
                        total_loss += loss.item()
                    except Exception as e:
                        logger.error(f"Error in batch processing: {str(e)}")
                        continue
                
                avg_loss = total_loss / len(train_loader)
                metrics['train_loss'].append(avg_loss)
                metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step(avg_loss)
                
                # Early stopping check
                if self.early_stopping(avg_loss):
                    logger.info("Early stopping triggered")
                    break
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.best_model = copy.deepcopy(self.model)
                    logger.info(f"New best model saved with loss: {best_loss:.4f}")
            
            return {
                "status": "success",
                "final_loss": best_loss,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise

    def _prepare_dataset(self, data: Dict[str, Any]) -> Dataset:
        """
        Convert raw data into a PyTorch Dataset
        """
        class SimpleDataset(Dataset):
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
        
        return SimpleDataset(data['texts'], data['labels'])

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