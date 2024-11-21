import torch
from typing import Dict, Any
import numpy as np
from .model import EnterpriseAIModel
import logging
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        required_params = [
            'model_type', 'learning_rate', 'weight_decay', 'batch_size',
            'num_document_classes', 'num_entity_types', 'time_series_features',
            'forecast_horizon'
        ]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required config parameter: {param}")
        
        if 'num_document_classes' not in config or config['num_document_classes'] <= 0:
            raise ValueError("Invalid configuration: 'num_document_classes' must be specified and greater than 0.")
        
        self.model = EnterpriseAIModel(
            model_type=config['model_type'],
            config=config
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Implement a learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            steps_per_epoch=1,
            epochs=config.get('epochs', 10),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Enable mixed precision training for speed and efficiency
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        try:
            wandb.init(
                project="enterprise-ai",
                config=config
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
        
    def train(self, train_data: Dict[str, Any], val_data: Dict[str, Any], epochs: int = 10):
        logger.info(f"Starting training on {self.device}")
        
        train_tensors = self._prepare_data(train_data)
        val_tensors = self._prepare_data(val_data)
        
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            for i in tqdm(range(gradient_accumulation_steps), desc=f"Epoch {epoch + 1}/{epochs}"):
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(train_tensors)
                    loss = self.model._compute_loss(outputs, train_tensors)
                    loss = loss / gradient_accumulation_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / gradient_accumulation_steps
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            wandb.log({'epoch': epoch, 'train_loss': avg_loss})
            
            self.scheduler.step()
            
            with torch.no_grad():
                self.model.eval()
                val_outputs = self.model(val_tensors)
                val_loss = self.model._compute_loss(val_outputs, val_tensors)
                logger.info(f"Validation Loss: {val_loss.item():.4f}")
                wandb.log({'val_loss': val_loss.item()})
        
        logger.info("Training completed")
    
    def _prepare_data(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tensors = {}
        if 'text' in data:
            tensors['text'] = data['text']
        if 'time_series' in data:
            tensors['time_series'] = torch.tensor(
                data['time_series'], 
                dtype=torch.float32
            ).unsqueeze(1).to(self.device)
        if 'document_class' in data:
            tensors['document_class'] = torch.tensor(
                data['document_class'],
                dtype=torch.long
            ).to(self.device)
        return tensors
    
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(batch)
                    loss = self.model._compute_loss(outputs, batch)
                    total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        # learning Rate Scheduling: Added OneCycleLR scheduler for dynamic learning rate adjustment, which is effective for stable training for a long time 
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        if hasattr(self, 'optimizer') and checkpoint['optimizer_state']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f"Loaded checkpoint from {path}")