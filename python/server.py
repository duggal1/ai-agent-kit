from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import torch
from enterprise_ai.model import EnterpriseAIModel
from enterprise_ai.trainer import ModelTrainer
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
models: Dict[str, EnterpriseAIModel] = {}
trainers: Dict[str, ModelTrainer] = {}

class ModelRequest(BaseModel):
    model_type: str
    data: Dict[str, Any]

class TrainingRequest(BaseModel):
    model_type: str
    training_data: Dict[str, Any]
    config: Dict[str, Any]

@app.post("/api/train")
async def train_model(request: TrainingRequest):
    try:
        # Add model_type to config if not present
        config = request.config
        config['model_type'] = request.model_type
        
        # Initialize the local trainer
        trainer = ModelTrainer(config)
        
        # Train the model locally
        trainer.train(
            train_data=request.training_data,
            val_data=request.training_data,  # Using the same data for validation temporarily
            epochs=10
        )
        
        # Store the trained model
        models[request.model_type] = trainer.model
        trainers[request.model_type] = trainer
        
        return {
            "status": "success",
            "message": "Training completed successfully",
            "model_type": request.model_type
        }
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(request: ModelRequest):
    try:
        if request.model_type not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Use the local model for prediction
        model = models[request.model_type]
        model.eval()
        
        with torch.no_grad():
            # Prepare input data
            input_tensors = trainers[request.model_type]._prepare_data(request.data)
            outputs = model(input_tensors)
            
            # Convert outputs to Python native types
            prediction = {
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }
            
            return {
                "status": "success",
                "prediction": prediction
            }
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    return {
        "available_models": list(models.keys()),
        "model_types": [
            "document_processor",
            "workflow_optimizer",
            "customer_intelligence",
            "supply_chain_optimizer"
        ]
    }

@app.get("/api/test")
async def test_setup():
    """Test endpoint to verify the server setup"""
    return {
        "status": "ok",
        "available_models": list(models.keys()),
        "available_trainers": list(trainers.keys()),
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)