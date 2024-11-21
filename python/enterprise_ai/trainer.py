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
        
        # Ensure required config parameters are present
        required_params = [
            'model_type', 'learning_rate', 'weight_decay', 'batch_size',
            'num_document_classes', 'num_entity_types', 'time_series_features',
            'forecast_horizon'
        ]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required config parameter: {param}")
        
        # Forcefully ensure 'document_class' is included in the config
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
        
        # Initialize wandb for experiment tracking
        try:
            wandb.init(
                project="enterprise-ai",
                config=config
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
        
    def train(self, train_data: Dict[str, Any], val_data: Dict[str, Any], epochs: int = 10):
        logger.info(f"Starting training on {self.device}")
        
        # Convert training data to tensors
        train_tensors = self._prepare_data(train_data)
        val_tensors = self._prepare_data(val_data)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            outputs = self.model(train_tensors)
            loss = self.model._compute_loss(outputs, train_tensors)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            wandb.log({
                'epoch': epoch,
                'train_loss': loss.item()
            })
            
            # Validation phase
            with torch.no_grad():
                self.model.eval()
                val_outputs = self.model(val_tensors)
                val_loss = self.model._compute_loss(val_outputs, val_tensors)
                logger.info(f"Validation Loss: {val_loss.item():.4f}")
                wandb.log({'val_loss': val_loss.item()})
        
        logger.info("Training completed")
    
    def _prepare_data(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert input data to tensors"""
        tensors = {}
        if 'text' in data:
            tensors['text'] = data['text']  # Keep as list for tokenizer
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
        
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        if hasattr(self, 'optimizer') and checkpoint['optimizer_state']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f"Loaded checkpoint from {path}")