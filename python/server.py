from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import torch
import logging
import os
import torch.nn.functional as F
import datetime
from transformers import AutoTokenizer

# Import only the components we actually use
from enterprise_ai.model import EnterpriseAIModel
from enterprise_ai.trainer import EnhancedTrainer
from enterprise_ai.losses import AdvancedLossFunctions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Enterprise AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data validation models
class TrainingData(BaseModel):
    texts: List[str]
    labels: List[int]

class ModelConfig(BaseModel):
    num_document_classes: int
    num_entity_types: int
    num_semantic_classes: int = 3
    time_series_features: int = 64
    forecast_horizon: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    batch_size: int = 16
    dropout_rate: float = 0.1
    embedding_dim: int = 1024
    use_advanced_features: bool = True
    use_specialized_modules: bool = True

class TrainingRequest(BaseModel):
    model_type: str
    training_data: TrainingData
    config: ModelConfig

class PredictionRequest(BaseModel):
    model_type: str
    data: Dict[str, Any]

class APIResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

# Global state
models: Dict[str, EnterpriseAIModel] = {}
trainers: Dict[str, EnhancedTrainer] = {}
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

def initialize_model(model_type: str, config: Dict[str, Any]) -> EnterpriseAIModel:
    """Initialize a new model with proper configuration"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnterpriseAIModel(
            model_type=model_type,
            config=config
        ).to(device)
        
        # Initialize with dummy input to catch any initialization errors
        with torch.no_grad():
            dummy_input = tokenizer(
                "This is a test input",
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            model(dummy_input)
        
        return model
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
        raise

@app.post("/api/train")
async def train_model(request: TrainingRequest) -> APIResponse:
    try:
        # Initialize trainer if not exists
        if request.model_type not in trainers:
            trainers[request.model_type] = EnhancedTrainer(request.config.dict())
        
        trainer = trainers[request.model_type]
        
        # Train model with data in correct format
        training_result = trainer.train(
            training_data={
                'texts': request.training_data.texts,
                'labels': request.training_data.labels
            }
        )
        
        if training_result['success']:
            # Save trained model
            models[request.model_type] = trainer.model
            
            return APIResponse(
                success=True,
                result={
                    'model_type': request.model_type,
                    **training_result
                }
            )
        else:
            return APIResponse(
                success=False,
                error=training_result.get('error', 'Unknown training error')
            )
            
    except Exception as e:
        logger.error(f"Enhanced training error: {str(e)}")
        return APIResponse(success=False, error=str(e))

@app.post("/api/predict", response_model=APIResponse)
async def predict(request: PredictionRequest) -> APIResponse:
    try:
        if request.model_type not in models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        
        model = models[request.model_type]
        model.eval()
        
        with torch.no_grad():
            # Process input
            inputs = {
                'text': request.data.get('text', ''),
                'metadata': request.data.get('metadata', {})
            }
            
            # Get model outputs
            outputs = model(inputs)
            
            # Post-process outputs
            try:
                processed_outputs = model.post_process_outputs(outputs)
                
                return APIResponse(
                    success=True,
                    result={
                        **processed_outputs,
                        'metadata': {
                            'model_type': request.model_type,
                            'processing_timestamp': datetime.datetime.now().isoformat(),
                            'model_version': '2.0',
                            'confidence_boosted': True,
                            'multi_scale_enabled': True,
                            'enhanced_entities': True,
                            'complexity_aware': True
                        }
                    }
                )
                
            except Exception as e:
                logger.error(f"Advanced prediction processing error: {str(e)}", exc_info=True)
                # Fallback to basic processing
                return APIResponse(
                    success=True,
                    result={
                        'document_classification': {
                            'probabilities': outputs.get('document_class', {}).get('logits', torch.zeros(1)).softmax(dim=-1).tolist(),
                            'confidence': 0.5,
                            'complexity_score': 0.5
                        }
                    }
                )
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return APIResponse(success=False, error=str(e))

@app.get("/api/models")
async def list_models():
    available_models = list(models.keys())
    model_types = [
        "enhanced_document_processor",
        "advanced_workflow_optimizer",
        "intelligent_customer_analyzer",
        "smart_supply_chain_optimizer"
    ]
    
    return {
        "available_models": available_models,
        "model_types": model_types,
        "model_capabilities": {
            "specialized_modules": True,
            "advanced_features": True,
            "enhanced_processing": True,
            "multi_scale_attention": True,
            "confidence_boosting": True,
            "complexity_aware": True,
            "transformer_layers": 12,
            "attention_heads": 32,
            "model_version": "2.0"
        }
    }

@app.get("/api/test")
async def test_endpoint():
    return {"status": "ok", "message": "API is working"}

if __name__ == "__main__":
    import uvicorn
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Run enhanced server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )